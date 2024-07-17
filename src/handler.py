import os
import gc
import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine
from constants import DEFAULT_MAX_CONCURRENCY

vllm_engine = vLLMEngine()
openai_engine = OpenAIvLLMEngine(vllm_engine)

async def handler(job):
    global vllm_engine
    global openai_engine

    print("input", job["input"])

    current_model = vllm_engine.config["model"]
    requested_model = job["input"].get("openai_input", {}).get("model", current_model)
    if requested_model != current_model:
        print(f"{requested_model} != {current_model}, reallocating...")
        del vllm_engine
        del openai_engine
        gc.collect()
        vllm_engine = vLLMEngine(model=requested_model)
        openai_engine = OpenAIvLLMEngine(vllm_engine)
        print("reallocated!")
    else:
        print(f"same model: {requested_model}, continue")

    job_input = JobInput(job["input"])
    engine = openai_engine if job_input.openai_route else vllm_engine
    results_generator = engine.generate(job_input)
    async for batch in results_generator:
        yield batch

concurrency_modifier = int(os.getenv("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))
runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: concurrency_modifier,
        "return_aggregate_stream": True,
    }
)
