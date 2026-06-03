# Dockerfile — a reproducible GPU environment for the GRPO training notebook.
#
# WHY THIS EXISTS
#   On Colab the CUDA/torch/vllm versions don't always agree (e.g. torch built for
#   CUDA 12.8 next to a CUDA 13.0 driver and a stray cuda_runtime 13.3 wheel). That
#   mismatch is what produced the "libcudart.so.13: cannot open shared object file"
#   and "cumem allocator is not supported on current platform" errors.
#
#   This file pins the WHOLE environment to a single, verified base image where vLLM,
#   PyTorch, and CUDA are already built to match. Build it once, run it anywhere with
#   an NVIDIA GPU (RunPod / Lambda / Vast.ai), and those version errors cannot occur.
#
# HOW TO USE (on a GPU box with Docker + the NVIDIA Container Toolkit installed):
#   1. Build the image (one time, ~5-10 min the first build):
#        docker build -t grpo .
#   2. Run it with the GPU passed through and a Jupyter server exposed:
#        docker run --gpus all -p 8888:8888 -it grpo
#      Then open the Jupyter URL it prints and run grpo.ipynb top-to-bottom.
#   (On RunPod/Lambda you can instead set this image as the pod's "custom image"
#    and use their built-in Jupyter — see DOCKER.md for the click-by-click version.)

# ---------------------------------------------------------------------------
# 1) BASE IMAGE — the foundation. This is the official vLLM image, version-matched:
#    vLLM 0.22.0 + PyTorch 2.11 + CUDA 12.9, all built together by the vLLM team.
#    Because torch and vllm come baked in here (and matched), we must NOT pip-install
#    them again below — doing so would risk pip "upgrading" them and re-breaking the
#    match. We only add the lighter libraries on top.
# ---------------------------------------------------------------------------
FROM vllm/vllm-openai:v0.22.0-cu129-ubuntu2404 

#  vllm/vllm-openai : v0.22.0 - cu129 - ubuntu2404
# └─ who/what ───┘   └ vLLM ┘  └CUDA┘  └─ OS ──┘
#       0.22.0    12.9    Ubuntu 24.04

# ---------------------------------------------------------------------------
# 2) WORKDIR — the folder inside the container where our code lives and commands run.
#    Created automatically. Think of it as a persistent `cd /workspace`.
# ---------------------------------------------------------------------------
WORKDIR /workspace

# ---------------------------------------------------------------------------
# 3) PYTHON DEPENDENCIES — only the packages NOT already in the base image.
#    Deliberately omits torch and vllm (inherited from the base, already matched).
#    These are the same libs as requirements-gpu.txt minus torch/vllm.
#      --no-cache-dir : don't keep pip's download cache, keeps the image smaller.
#    jupyter is added so you can open and run the notebook inside the container.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    deepspeed \
    "transformers>=4.56.0" \
    datasets \
    numpy \
    tqdm \
    wandb \
    math_verify \
    jupyterlab

# ---------------------------------------------------------------------------
# 4) COPY THE PROJECT IN — bring our code from the build machine into the image.
#    `./` means "into the current WORKDIR (/workspace)". utils.py is imported by
#    the notebook, so it must be present alongside it.
# ---------------------------------------------------------------------------
COPY . .

# ---------------------------------------------------------------------------
# 5) EXPOSE + DEFAULT COMMAND — start a JupyterLab server when the container runs.
#    EXPOSE documents the port; the actual mapping happens in `docker run -p`.
#      --ip=0.0.0.0    : listen on all interfaces so it's reachable from outside the
#                        container (required, otherwise you can't connect to it).
#      --no-browser    : don't try to open a browser inside the container.
#      --allow-root    : the base image runs as root; Jupyter needs this flag to allow it.
#      --NotebookApp.token='' : disable the login token for convenience on a private box.
#                               (Remove this on any shared/public machine for safety.)
# ---------------------------------------------------------------------------
EXPOSE 8888
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''"]
