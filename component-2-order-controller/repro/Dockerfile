# Full-scale MaskGIT reproduction, RunPod-ready (A100-class GPU).
# Same core mechanism as the toy CPU repro in src/ (cosine masked-token
# training + confidence-ranked parallel decoding) -- only the tokenizer,
# dataset and model size scale up. Not built or run in the CPU-only sandbox
# that produced logs/.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements-fullscale.txt /workspace/requirements-fullscale.txt
RUN pip install --no-cache-dir -r requirements-fullscale.txt

COPY src/ /workspace/src/
COPY runpod_launch.sh /workspace/runpod_launch.sh
RUN chmod +x /workspace/runpod_launch.sh

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/workspace/runpod_launch.sh"]
