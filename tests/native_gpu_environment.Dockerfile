# CPU-only bootstrap fixture. No NVIDIA runtime, checkpoint or model download.
FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285
RUN python -m pip install --disable-pip-version-check torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
# Disable only the disposable image's ensurepip, reproducing the Colab failure.
RUN mv /usr/local/lib/python3.13/ensurepip /usr/local/lib/python3.13/ensurepip-disabled-for-rotquant-test
WORKDIR /workspace
CMD ["python", "-u", "scripts/check_native_gpu_environment.py", "--work-dir", "/tmp/rq3-bootstrap-test", "--output-dir", "/workspace/build/native-environment-ci"]
