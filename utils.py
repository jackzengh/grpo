from datetime import timedelta
import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import wandb
from datasets import Dataset
from deepspeed import DeepSpeedEngine
from transformers import AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

@torch.compile(dynamic=True)
def log_softmax_and_gather(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Copied from https://github.com/allenai/open-instruct/blob/main/open_instruct/model_utils.py#L425
    
    See https://github.com/allenai/open-instruct/pull/584
    """
    logprobs = logits.log_softmax(dim=-1) # perform softmax operation on the last dimension
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1) # from each sequence we just pluck out the index we care about
    # torch.gather allows us to index into the desired sequence then pluck out the index we want
    # this happens to be the index in label, 


def compute_token_log_probs(
    model: Union[DeepSpeedEngine, PreTrainedModel],
    inputs: Dict[str, torch.Tensor],
    temperature: float,
):
   
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
        use_cache=False,
    )

    logits = outputs.logits / temperature  # Shape: [batch_size, seq_len, vocab_size]
    shift_logits = logits[..., :-1, :]  # Shape: [batch_size, seq_len-1, vocab_size] remove final logit since we have no label for it
    shift_labels = inputs["labels"][..., 1:]  # Shape: [batch_size, seq_len-1] now indices of logits and compared labels are matched up
    shift_labels_mask = inputs["labels_mask"][..., 1:]  # Shape: [batch_size, seq_len-1] just shifts label mask

    # Create mask for valid labels
    shift_labels[~(shift_labels_mask.bool())] = 0  # Shape: [batch_size, seq_len-1] set padding labels to zero

    # Calculate log probabilities
    log_probs = log_softmax_and_gather(shift_logits, shift_labels)  # Shape: [batch_size, seq_len-1]
    log_probs = log_probs * shift_labels_mask  # Shape: [batch_size, seq_len-1]

    return log_probs

