# RunPod-ready scaffold for a FULL-SCALE rerun of this reproduction:
# a real Stable-Diffusion-scale UNet, CIFAR-10/ImageNet, and a much higher
# starting NFE (e.g. 1024/512 -> ... -> 1, matching the paper's own settings).
# Not built or executed in this (CPU-only) session -- provided for the
# applicant to launch on RunPod A100 hardware themselves.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /workspace/src/
COPY runpod_launch.sh /workspace/runpod_launch.sh
RUN chmod +x /workspace/runpod_launch.sh

# Mount or bake in a bigger checkpoint / dataset cache as needed; this image
# only ships the code. See runpod_launch.sh for the intended full-scale
# command line (bigger UNet, bigger dataset, higher starting NFE).
ENTRYPOINT ["/workspace/runpod_launch.sh"]
