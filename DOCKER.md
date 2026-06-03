# Running GRPO training in Docker (RunPod / Lambda / Vast.ai)

This guide explains, step by step and in plain language, how to run the training
notebook in a reproducible Docker container — so the CUDA/torch/vLLM version
mismatches you hit on Colab (the `libcudart.so.13` and `cumem allocator not
supported` errors) cannot happen.

## What's in this repo for Docker

- **`Dockerfile`** — the recipe. Starts from vLLM's official image
  (`vllm/vllm-openai:v0.22.0-cu129-ubuntu2404`), which already contains a matched
  set: **vLLM 0.22.0 + PyTorch 2.11 + CUDA 12.9**. It then installs only the extra
  libraries (deepspeed, transformers, etc.) and copies your code in.
- **`.dockerignore`** — tells Docker which local files to skip (your `.venv`, git
  history, caches) so the image stays small.

The key idea: **torch and vllm are NOT reinstalled** — they're inherited from the
base image, already built to match CUDA. That's what kills the whole class of
version errors.

---

## Option A — RunPod (easiest, recommended for a first time)

RunPod lets you boot a GPU machine that already has Docker + NVIDIA drivers set up,
so you don't install anything on your own computer.

### 1. Make the image available to RunPod
RunPod pulls images from a registry (like Docker Hub). Two ways:

**Simplest — skip building entirely:** RunPod can run the **base image directly**.
When deploying a pod, set the custom image to:
```
vllm/vllm-openai:v0.22.0-cu129-ubuntu2404
```
Then install the extra packages and clone your repo from the pod's terminal (steps
below). You don't even need to build the Dockerfile for this path.

**Full image — build & push (do this once you want everything baked in):**
On any machine with Docker (or RunPod's own build feature):
```bash
docker build -t YOUR_DOCKERHUB_USERNAME/grpo:latest .
docker push YOUR_DOCKERHUB_USERNAME/grpo:latest
```
Then point the pod at `YOUR_DOCKERHUB_USERNAME/grpo:latest`.

### 2. Deploy the pod
- On runpod.io → **Deploy** → pick a GPU. An **A100 80GB** makes the memory
  problems disappear; an A100 40GB also works with the optimizer-offload already in
  the notebook.
- Set the **container image** to one of the two above.
- Set a **port** to expose for Jupyter: `8888`.
- Deploy.

### 3. Get into it and run
- Click the pod's **Jupyter** or **Web Terminal** button.
- If you used the *base* image (not the full build), set up your code once:
  ```bash
  cd /workspace
  git clone https://github.com/jackzengh/grpo.git
  cd grpo
  pip install deepspeed "transformers>=4.56.0" datasets numpy tqdm wandb math_verify
  ```
  (torch + vllm are already present — do not reinstall them.)
- Open `grpo.ipynb` and run cells top-to-bottom.

### 4. Sanity check before training
In a cell or terminal:
```bash
python -c "import torch, vllm; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| vllm', vllm.__version__)"
```
You should see torch 2.11 / cuda 12.9 / vllm 0.22.0 — one coherent set, no mismatch.

---

## Option B — Build and run on your own GPU machine

If you have a Linux box with an NVIDIA GPU, Docker, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
# from inside this repo
docker build -t grpo .                 # build the image (one time)
docker run --gpus all -p 8888:8888 -it grpo
```
`--gpus all` passes the GPU into the container. Then open the JupyterLab URL it
prints (http://localhost:8888) and run `grpo.ipynb`.

---

## Notes & gotchas

- **vLLM sleep mode should work here.** The `cumem allocator not supported` error was
  caused by Colab's CUDA mismatch. With this matched image, `enable_sleep_mode=True`
  and the `.sleep()` / `.wake_up()` calls in the training loop should function. If it
  still errors on a given box, the fallback is to disable sleep mode and give vLLM a
  small fixed `gpu_memory_utilization` slice.
- **Weights & Biases login:** the notebook calls `wandb.init`. Either run
  `wandb login` in the terminal first, or set `WANDB_MODE=offline` (or pass
  `mode="disabled"`) if you don't want logging.
- **Hugging Face downloads:** the model and dataset download on first run. For higher
  rate limits set a token: `export HF_TOKEN=...` before launching.
- **Persisting checkpoints:** containers are ephemeral. On RunPod, attach a
  **Network Volume** (or use `/workspace` if the pod has persistent storage) so your
  saved checkpoints survive a pod restart. With plain `docker run`, mount a folder:
  `docker run --gpus all -v $(pwd)/checkpoints:/workspace/checkpoints ... grpo`.
- **Updating the image when you change code:** rebuild (`docker build`) — or, faster
  while iterating, just `git pull` your latest code inside the running container.
