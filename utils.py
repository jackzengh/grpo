from datetime import timedelta
import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import wandb
from datasets import Dataset
from deepspeed import DeepSpeedEngine
from transformers import AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

def prepare_model_inputs(
    query_token_ids: List[List[int]],
    response_token_ids: List[List[int]],
    advantages: List[List[int]],
    device: torch.device,
):
    max_seq_len = max(len(q) + len(r) for q, r in zip(query_token_ids, response_token_ids)) # set the max length to be the longest prompt + response we've got 
    inputs = {"input_ids": [], "attention_mask": [], "labels": [], "advantages":[], "labels_mask":[]}

    """
    Shape of input_ids: 
    [
        [12, 88, 5, 0, 0],     <- sequence 1 (padded with 0s at the end)
        [4, 9, 31, 7, 0],      <- sequence 2
        [1, 2, 3, 4, 5],       <- sequence 3
    ]
    """

    pad_token_id = 0
    ignore_index = -100

    # unpack each unique prompt and its completions
    for query, response, advantage in zip(query_token_ids,response_token_ids, advantages):
        # the model reads the full sequence
        combined_ids = query + response 
        seq_len = len(combined_ids)

        ###################################
        # PADDING MASK
        ###################################
        
        # for a single sequence prepare inputs + labels + advantages and append to respective array in the *inputs* dict
        # create padded sequence
        input_ids = combined_ids + [pad_token_id] * (max_seq_len-seq_len) # pad to max_tokens
        attention_mask = [1] * seq_len + [0] * (max_seq_len-seq_len) # mask out the padding tokens
        
        # labels is just response_token_ids but with the query masked out + taking only [1:seq_len] to compare with generated tokens
        # we mask out query because we don't want to train the model to predict the prompt
        labels = [ignore_index] * len(query) + response + [ignore_index] * (max_seq_len-seq_len) # mask the query + pad tokens

        ###################################
        # LOSS + ADVANTAGE MASK
        ###################################
        """
        WTF IS LABELS?
        
        Labels are what we'll use in the compute_token_log_probs function 
        to compare the completion made by vLLM (actor) to the completion made by 
        our policy_model (learner). That's why we need to mask labels, we don't 
        want our policy_model learning to generate 'query' tokens
        """

        labels_mask = [0] * len(query) + [1] * len(response) + [0] * (max_seq_len-seq_len)
        advantages_seq = [0] * len(query) + advantage + [0.0] * (max_seq_len-seq_len)

        assert len(input_ids) == max_seq_len
        assert len(attention_mask) == max_seq_len
        assert len(labels) == max_seq_len
        assert len(advantages_seq) == max_seq_len
        assert len(labels_mask) == max_seq_len

        inputs["input_ids"].append(input_ids)
        inputs["attention_mask"].append(attention_mask)
        inputs["labels"].append(labels)
        inputs["advantages"].append(advantages_seq)
        inputs["labels_mask"].append(labels_mask)

    return {
        # recreate the inputs array but in a dict, converting the arrays to tensors
        k: torch.tensor(v, dtype=torch.long if k != "advantages" else torch.float, device=device) for k, v in inputs.items() # always need .items() for a dict
    }


@torch.compile(dynamic=True)
def log_softmax_and_gather(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Copied from https://github.com/allenai/open-instruct/blob/main/open_instruct/model_utils.py#L425
    
    See https://github.com/allenai/open-instruct/pull/584
    """
    logprobs = logits.log_softmax(dim=-1) # perform softmax operation on the last dimension which is (B, T, V) where V is vocab_size
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1) # from each sequence we just pluck out the index (i.e. token id) we care about
    # torch.gather allows us to index into the desired sequence then pluck out the index we want
    # this index is the next_token index from our labels array - which is token_ids


def compute_token_log_probs(
    model: DeepSpeedEngine | PreTrainedModel,
    inputs: Dict[str, torch.Tensor],
    temperature: float,
):
    # we pass our input_ids into the model again so that it produces logits for each token
    # output is (B,T,V) where V is vocab_size
    # recall that in attention, you produce logits for every token passed in (due to attention masking)
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
        use_cache=False,
    )
    # divide by temperature which recall in softmax becomes e^{logits/temperature}
    logits = outputs.logits / temperature  # Shape: [batch_size, seq_len, vocab_size]
    
    # shift logits left
    shift_logits = logits[..., :-1, :]  # Shape: [batch_size, seq_len-1, vocab_size] remove final logit since we have no label for it
    
    # shift labels right, 
    shift_labels = inputs["labels"][..., 1:]  # Shape: [batch_size, seq_len-1] now indices of logits and compared labels are matched up
    
    # RECALL: labels_mask = [0] * len(query) + [1] * len(response) + [0] * (max_seq_len-seq_len)
    shift_labels_mask = inputs["labels_mask"][..., 1:]  # Shape: [batch_size, seq_len-1], take only the tokens at t+1, which is what you're matching the completion against

    # Now zero out everything that isn't '1' in the mask
    # Stops us attending to -100 - the ignore_index - in labels
    shift_labels[~(shift_labels_mask.bool())] = 0  # Shape: [batch_size, seq_len-1] set padding labels to zero

    # Calculate log probabilities
    log_probs = log_softmax_and_gather(shift_logits, shift_labels)  # Shape: [batch_size, seq_len-1]
    log_probs = log_probs * shift_labels_mask  # Shape: [batch_size, seq_len-1]

    return log_probs

# Extract solution from MATH solution dict
def extract_boxed_answer(solution: str): 

	idx = solution.rfind(r"\boxed") # reversed find

	if idx == -1: 
		return None # didn't find solution, return -1
	
	i = idx+len(r"\boxed")
	while i < len(solution) and solution[i] != "{":
		i += 1 # example solution: \boxed{0}$. We are just trying to skip whitespacae
	if i >= len(solution): 
		return None
	num_open_left_brace = 0
	start = i + 1 
	for j in range(i, len(solution)): 
		if solution[j] == "{":
			num_open_left_brace += 1 # counting the open brackets, ensuring they are closed for cases like \boxed   {\dfrac{1}{2}}
		if solution[j] == "}":
			num_open_left_brace -= 1
			if num_open_left_brace == 0:
				return solution[start:j].strip()
	return ""

def find_last_checkpoint(exp_dir: Path):
    checkpoint_dir = exp_dir / "checkpoints" # where checkpoints are located
    checkpoints = list(checkpoint_dir.glob("ckpt_*")) # find all folder starting with ckpt

    checkpoints = [ckpt for ckpt in checkpoints if (ckpt / "deepspeed").exists()]
    if not checkpoints:
        return None, None
    ckpt_path = max(checkpoints, key=lambda x: int(x.stem.split("_")[-1])) # pick the checkpoint with teh biggest number
    ckpt_iter = int(ckpt_path.stem.split("_")[-1]) # what iteration is this?
    return ckpt_path, ckpt_iter

def evaluate_on_test_set(
    inference_engine: LLM,
    test_dataset: Dataset,
    tokenizer: AutoTokenizer,
    eos_token: str,
    eval_sampling_params: SamplingParams,
    reward_func: Callable[[str, Dict[str, Any]], Tuple[float, Dict[str, float]]], # generic reward function that takes function as input with parameters of (completion, sample)
    num_eval_samples: int = 500,
):
    """
    We'll need to sample from the test dataset, pass the prompt into the 
    inference engine, then compare it to the answer. Evaluating the answer
    using the reward_func. 

    The inference engine uses the the tokenizer passed in + sampling params 
    to determine temperature + max_tokens

    """
    ### PLUCK OUT num_eval_samples from the eval set
    # Subsample the test set so eval is fast (mirrors the training-loop sampling idiom).
    # `size=` is np.random.choice's count kwarg; replace=False so prompts are unique.
    indices = np.random.choice(len(test_dataset), size=num_eval_samples, replace=False)
    eval_samples = test_dataset.select(indices)

    # generations will returns an array of arrays with the completions for each respective prompt
    # vLLM is built for batching prompts and simultaneous generation
    #
    # NOTE (vLLM API): generate() no longer takes a top-level prompt_token_ids= kwarg.
    # Pre-tokenized input is passed as the first positional arg, as a list of dicts
    # each shaped {"prompt_token_ids": [...]} (vLLM's "TokensPrompt" form).
    generations = inference_engine.generate(
        [{"prompt_token_ids": ids} for ids in eval_samples["input_ids"]],
        sampling_params=eval_sampling_params,
    ) # controlled generation with temperature, max_tokens setting

    metrics = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
    }

    all_query_token_ids = [] # prompt tokens
    all_response_token_ids = [] # answer tokens

    for i, sample in enumerate(eval_samples): # iterate the SAME subsample we generated for, so generations[i] lines up

        query_token_ids = sample["input_ids"]
        response_token_ids = generations[i].outputs[0].token_ids 
        finish_reason = generations[i].outputs[0].finish_reason

        response = tokenizer.decode(response_token_ids, skip_special_tokens=False) # decode reward to pass into reward_func
        reward, reward_components = reward_func(response, sample) # compute reward

        all_query_token_ids.append(query_token_ids)
        all_response_token_ids.append(response_token_ids)

        metrics["rewards"].append(reward)
        metrics["non_stop_rate"].append(finish_reason != "stop")
        metrics["response_lengths"].append(len(response_token_ids))

        for k, v in reward_components.items(): # .items() lets you iterate over dict in (k,v) pairs
            # dictionary.setdefault(key, defaultvalue) retrieves or inserts key with defaultvalue
            metrics.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = { # contains array of arrays of prompt tokens and the completions associated
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_response_token_ids,
    }

    return episodes, metrics

def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def dump_episodes(
    episodes: Dict[str, Any],
    episodes_stats: Dict[str, Any],
    exp_dir: Path,
    tokenizer: AutoTokenizer,
    iteration: int,
    is_eval: bool = False,
    do_save: bool = True,
):

    """
    Save episodes to wandb and logging

    exp_dir/                        ← the one folder you pass in
        ├── checkpoints/                ← model snapshots (ckpt_100/, ckpt_200/, …)
        │   └── ckpt_200/deepspeed/
        ├── episodes/                   ← training experience saved here
        └── eval_episodes/              ← evaluation experience saved here
    """
    query_token_ids = episodes["all_query_token_ids"]
    response_token_ids = episodes["all_response_token_ids"]
    rewards = episodes_stats["rewards"]
    response_lengths = episodes_stats["response_lengths"]

    query_texts = tokenizer.batch_decode(
        query_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    response_texts = tokenizer.batch_decode(
        response_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )

    # distributed training or no?
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0 # rank of GPU

    if not is_eval and rank == 0:
        print(f"########## Example 1 (Reward: {rewards[0]}, Response Length: {response_lengths[0]})")
        print(f"#### Query:\n`{query_texts[0]}`")
        print(f"#### Response:\n`{response_texts[0]}`\n\n")

        print(f"########## Example 2 (Reward: {rewards[1]}, Response Length: {response_lengths[1]})")
        print(f"#### Query:\n`{query_texts[1]}`")
        print(f"#### Response:\n`{response_texts[1]}`\n\n")
    
    # SCRATCH folder where checkpoints, logging, eval results, episodes are saved
    if is_eval:
        # save to 
        episodes_dir = exp_dir / "eval_episodes"
    else: 
        episodes_dir = exp_dir / "episodes"
    
    if dist.is_initialized():
        episodes_dir = episodes_dir / f"rank_{rank:02d}"  # each GPU saves its episodes in their respective folder
    episodes_dir.mkdir(parents=True, exist_ok=True) # create dir for this these episodes - that is exp_dir / "..."

    # wandb logging
    table = wandb.Table(columns=["query", "response", "reward", "response_length"])
    for i in range(len(query_texts)):
        table.add_data(query_texts[i], response_texts[i], rewards[i], response_lengths[i])

    # should we save to memory? 
    if not do_save:
        return table

    # inside exp_dir/episodes/eps_1 (if iteration 1)
    with open(episodes_dir / f"eps_{iteration:06d}.json", "w") as f:
        json.dump( # writing to a json file
            [
                {
                    "query": query_texts[i],
                    "response": response_texts[i],
                    "reward": rewards[i],
                }
                for i in range(len(query_texts))
            ],
            f, # where to write the json dump
        )
    
    return table

def load_model_into_vllm(model: Union[DeepSpeedEngine, PreTrainedModel], llm: LLM) -> None:
    """
    Load weights from a HuggingFace model (either wrapped in DeepSpeed or not) into a vLLM inference engine.

    This function transfers the weights from a training model to a vLLM inference engine,
    allowing for efficient inference using the updated model weights.

    Args:
        model (Union[DeepSpeedEngine, PreTrainedModel]): The source model to copy weights from.
            Can be either a DeepSpeed-wrapped model or a regular HuggingFace PreTrainedModel.
        vllm (LLM): The target vLLM inference engine to load the weights into.
            Must be already initialized and ready to accept new weights.

    Returns:
        None
    """
    state_dict = model.module.state_dict() if isinstance(model, DeepSpeedEngine) else model.state_dict()
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())
