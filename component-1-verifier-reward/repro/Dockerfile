# Full-scale DDPO reproduction, RunPod-ready (A100-class GPU).
# Same code as the toy CPU repro in src/ — only the launch configuration
# (model size, T, dataset, reward model) scales up, via runpod_launch.sh /
# CLI flags. Not built or run in the CPU-only sandbox that produced logs/.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
# torch/torchvision already present in the base image with CUDA support;
# only add the extra full-scale deps here (kept separate from the CPU
# requirements.txt on purpose, see requirements-fullscale.txt).
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements-fullscale.txt /workspace/requirements-fullscale.txt
RUN pip install --no-cache-dir -r requirements-fullscale.txt

COPY src/ /workspace/src/
COPY runpod_launch.sh /workspace/runpod_launch.sh
RUN chmod +x /workspace/runpod_launch.sh

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/workspace/runpod_launch.sh"]
