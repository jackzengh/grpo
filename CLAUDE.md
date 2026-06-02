# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

A **from-scratch implementation of GRPO** (Group Relative Policy Optimization), the
RL algorithm behind DeepSeek R1-Zero. It trains `Qwen/Qwen2.5-3B` on the Hendrycks
MATH dataset to reason through math problems inside `<think>...</think>` tags and
produce a correct final answer. This is a learning/research project — the priority
is *clarity and correctness*, not production polish. The code is intentionally
heavily commented to expose the algorithm's mechanics.

See `README.md` for the full conceptual walkthrough.

## Layout

- `grpo.ipynb` — the entire training pipeline (config, dataset, reward function,
  advantage calc, loss, training loop). This is the main artifact.
- `utils.py` — supporting functions: `prepare_model_inputs`,
  `compute_token_log_probs`, `log_softmax_and_gather`, `extract_boxed_answer`,
  `find_last_checkpoint`, `evaluate_on_test_set`, `dump_episodes`.
- `requirements-gpu.txt` — full training stack (vLLM + DeepSpeed), **pinned**,
  Linux + NVIDIA GPU only.
- `requirements-dev.txt` — Mac-installable subset for editing locally (no
  vLLM/DeepSpeed).
- `requirements.txt` — loose unpinned superset.

## Critical environment constraint

**The training stack does NOT run on macOS.** `vllm` and `deepspeed` are
NVIDIA-GPU / Linux-only and will not install or build on a Mac. The user develops
on a Mac and runs training on a cloud GPU box (RunPod / Lambda / Vast.ai / Colab).

Consequences for you:
- Do **not** try to run the notebook or import `vllm`/`deepspeed` locally — it will
  fail. Don't suggest "just run it to check."
- For local sanity checks, only the `requirements-dev.txt` packages are available.
- When editing `utils.py`, remember it imports `vllm` and `deepspeed` at module
  top level, so the whole module is un-importable on Mac. Verify changes by reading,
  not by executing.

## Conventions in this codebase

- Tab indentation appears in the notebook cells and parts of `utils.py`; match the
  surrounding style of whatever file you edit.
- Comments explain the *why* and often the tensor **shapes** at each step (e.g.
  `# Shape: [batch_size, seq_len-1]`). Preserve and update these when you change
  shapes — they're the main teaching tool here.
- Padding convention: `pad_token_id = 0`, `ignore_index = -100`. Query tokens are
  masked out of `labels`/`labels_mask` so only generated tokens are trained on.
- Two models exist at runtime: the **policy** (trained) and the frozen
  **reference** (KL penalty only). Keep that distinction clear in any RL-logic edits.

## Working agreements (from the user's global preferences)

- Explain things in plain language, assuming no ML/programming jargon — define
  terms, use simple analogies. The user is learning this material.
- Tech stack is Python + PyTorch.
- **Never** add "Co-Authored-By: Claude" or any Claude attribution to git commits.
- Only commit or push when explicitly asked.

## Gotchas

- There's a known variable-name inconsistency in `utils.py`'s
  `evaluate_on_test_set` (`all_responses_token_ids` declared vs.
  `all_response_token_ids` appended). If you touch that function, reconcile the
  names rather than leaving the latent bug.
- Output (checkpoints, episodes, eval) is written under
  `~/scratch/deepseek_r1z_hackathon/<RUN_NAME>/`, and training auto-resumes from
  the latest checkpoint via `find_last_checkpoint`.
