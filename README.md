# GRPO from Scratch — DeepSeek R1-Zero style RL on MATH

This project trains a language model to **reason through math problems** using
**GRPO** (Group Relative Policy Optimization) — the reinforcement-learning method
behind DeepSeek's R1-Zero. It's a from-scratch, single-notebook implementation
written to learn how the algorithm actually works under the hood, not a wrapper
around a high-level library like TRL.

The model learns to write out its reasoning inside `<think>...</think>` tags and
then give a final answer, and it gets *rewarded* when (a) the answer is correct
and (b) it follows the required format.

---

## The 30-second mental model

Imagine you give a student the same hard math problem **4 times** and they write
4 different solutions. Some are right, some are wrong. Instead of needing a
"perfect answer key" to copy, you just tell the student: *"Of your 4 attempts,
these were better than your average — do more of that; these were worse — do less
of that."*

That comparison-within-a-group is the entire trick. There's no separate "critic"
network grading each answer (which is what older methods like PPO need). The group
of answers grades itself relative to its own average. That's the "Group Relative"
in GRPO.

The loop, per training step:

1. **Generate** — for a batch of math problems, sample several answers each (fast,
   using vLLM).
2. **Reward** — score every answer: is the final answer mathematically correct?
   Did it use the right format?
3. **Advantage** — within each group of answers to the same problem, measure how
   much better/worse than the group average each one was.
4. **Learn** — nudge the model to make above-average answers more likely and
   below-average ones less likely, while a "KL penalty" keeps it from drifting too
   far from where it started (so it doesn't forget how to write coherent text).

Repeat for many iterations and the model gets measurably better at math.

---

## What's in here

| File | What it is |
|------|-----------|
| `grpo.ipynb` | The main notebook — the whole training pipeline, top to bottom. Run this on a GPU. |
| `utils.py` | Helper functions: input tensor prep, log-prob computation, answer extraction, checkpointing, evaluation, and logging. |
| `requirements-gpu.txt` | The **full** training stack (vLLM + DeepSpeed). NVIDIA GPU / Linux only. |
| `requirements-dev.txt` | A Mac-installable subset for **reading and editing** the code locally (no vLLM/DeepSpeed). |
| `requirements.txt` | Loose, unpinned superset of the dependencies. |

### Key pieces inside the notebook / `utils.py`

- **`prepare_model_inputs`** (`utils.py`) — pads prompt+response sequences into
  equal-length tensors and builds the `labels`, `labels_mask`, and `advantages`
  arrays. The query tokens are masked out so the model is only trained on the
  tokens *it generated*, not on re-predicting the prompt.
- **`compute_token_log_probs`** (`utils.py`) — re-runs the model over each
  sequence to get the log-probability it assigns to each generated token. This is
  the core quantity GRPO optimizes.
- **`format_reward_fn`** + math verification (notebook) — the reward signal.
  Format reward checks the `<think>...</think>` structure; correctness is checked
  with the `math_verify` library, which parses and compares math expressions
  symbolically (so `1/2` and `0.5` count as equal).
- **`calculate_advantage`** (notebook) — turns raw rewards into group-relative
  advantages (subtract the group mean, divide by the group std).
- **`compute_loss`** (notebook) — the GRPO objective: the policy-gradient term
  weighted by advantage, plus the KL-divergence penalty against the frozen
  reference model.
- **`evaluate_on_test_set`** / **`dump_episodes`** (`utils.py`) — periodic eval on
  held-out problems and logging of example generations to Weights & Biases.

---

## The setup at a glance

- **Base model:** `Qwen/Qwen2.5-3B` (3-billion-parameter model). The `-Instruct`
  variant's tokenizer is used for its chat template.
- **Dataset:** `EleutherAI/hendrycks_math` (the Hendrycks MATH benchmark —
  competition-level math problems with `\boxed{}` answers).
- **Two models in play:**
  - **Policy model** — the one being trained (with DeepSpeed ZeRO-2).
  - **Reference model** — a frozen copy of the starting model, used only to compute
    the KL penalty so the policy doesn't drift too far.
- **Inference engine:** vLLM generates the candidate answers quickly in batches.
- **Logging:** Weights & Biases (`wandb`).

### Default hyperparameters (top of the notebook)

| Setting | Value | Meaning |
|---------|-------|---------|
| `NUM_ITERATIONS` | 1000 | total training steps |
| `EPISODES_PER_ITERATION` | 64 | answers collected per step |
| `GENERATIONS_PER_SAMPLE` | 4 | answers per problem (the "group size") |
| `KL_COEFFICIENT` | 0.001 | strength of the don't-drift penalty |
| `PER_DEVICE_BATCH_SIZE` | 4 | sequences per GPU per micro-step |
| `LEARNING_RATE` | 1e-6 | how big each update is |
| `MAX_RESPONSE_TOKENS` | 1024 | max length of a generated answer |
| `TEMPERATURE` | 1.0 | sampling randomness during training |

---

## Running it

> **Important:** the full training stack (vLLM + DeepSpeed) only runs on **Linux
> with an NVIDIA GPU (CUDA)**. It will *not* install on macOS. Use a cloud GPU box
> (RunPod, Lambda, Vast.ai, Colab, etc.).

### On a GPU box (training)

The first cell of `grpo.ipynb` does the setup:

```bash
git clone https://github.com/jackzengh/grpo.git
cd grpo
pip install -r requirements-gpu.txt
```

Then run the notebook cells top to bottom. You'll want a Weights & Biases account
(`wandb login`) for logging. Checkpoints, eval results, and saved episodes are
written under `~/scratch/deepseek_r1z_hackathon/<RUN_NAME>/`, and training
**auto-resumes** from the latest checkpoint if one exists.

### On a Mac (reading / editing only)

You can install the lightweight subset to get type-checking and editor support
without the GPU-only packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

This is enough to read and edit `utils.py`, but you **cannot run training**
locally — `vllm` and `deepspeed` are not installed (and won't build on macOS).

---

## How the output is organized

```
~/scratch/deepseek_r1z_hackathon/<RUN_NAME>/
├── checkpoints/          ← model snapshots (ckpt_100/, ckpt_200/, …)
│   └── ckpt_200/deepspeed/
├── episodes/             ← training experience (generated answers) saved as JSON
└── eval_episodes/        ← evaluation experience saved as JSON
```

---

## Why this exists

This is a learning project: a transparent, heavily-commented re-implementation of
the GRPO algorithm so the math and the data flow are visible at every step, rather
than hidden behind a library call. The comments in `utils.py` and the notebook
walk through *why* each tensor is shaped the way it is.
