FROM python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b
COPY requirements.txt /runtime/requirements.txt
RUN --mount=type=secret,id=proxy_ca,required=true PIP_CONFIG_FILE=/dev/null PIP_CERT=/run/secrets/proxy_ca REQUESTS_CA_BUNDLE=/run/secrets/proxy_ca SSL_CERT_FILE=/run/secrets/proxy_ca python -m pip install --no-cache-dir --disable-pip-version-check torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=secret,id=proxy_ca,required=true PIP_CONFIG_FILE=/dev/null PIP_CERT=/run/secrets/proxy_ca REQUESTS_CA_BUNDLE=/run/secrets/proxy_ca SSL_CERT_FILE=/run/secrets/proxy_ca python -m pip install --no-cache-dir --disable-pip-version-check -r /runtime/requirements.txt && python -m pip check
RUN python -c "import os; assert not os.path.exists('/run/secrets/proxy_ca'); assert not any(k in os.environ for k in ['PIP_CERT','REQUESTS_CA_BUNDLE','SSL_CERT_FILE'])"
# The CA is only a BuildKit mount. No CA bytes or certificate ENV is persisted.
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
