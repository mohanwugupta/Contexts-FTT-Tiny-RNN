FROM python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b
COPY requirements.txt /runtime/requirements.txt
RUN python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir -r /runtime/requirements.txt && python -m pip check
# Worker sources and all scientific evidence are mounted read-only at runtime.
# No source/evidence/credential directory is copied into this image.
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
