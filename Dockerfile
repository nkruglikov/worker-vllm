FROM runpod/worker-vllm:stable-cuda12.1.0
COPY src /src
CMD ["python3", "/src/handler.py"]
