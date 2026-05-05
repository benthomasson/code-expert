"""Model invocation for code expert."""

import asyncio
import os
import shutil

# Model CLI commands - extend this dict to add new models
# Note: gemini requires empty string after -p to read prompt from stdin
MODEL_COMMANDS: dict[str, list[str]] = {
    "claude": ["claude", "-p"],
    "gemini": ["gemini", "-p", ""],
}

DEFAULT_TIMEOUT = 300


def check_model_available(model: str) -> bool:
    """Check if a model's CLI is available."""
    if model not in MODEL_COMMANDS:
        return False
    cmd = MODEL_COMMANDS[model][0]
    return shutil.which(cmd) is not None


async def invoke(prompt: str, model: str = "claude", timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Invoke model via CLI, piping prompt through stdin.

    Args:
        prompt: Full prompt text to send
        model: Model name (must be in MODEL_COMMANDS)
        timeout: Timeout in seconds

    Returns:
        Model's response text

    Raises:
        ValueError: If model not supported
        TimeoutError: If model doesn't respond in time
        RuntimeError: If model invocation fails
    """
    if model not in MODEL_COMMANDS:
        raise ValueError(f"Unknown model: {model}. Available: {list(MODEL_COMMANDS.keys())}")

    cmd = MODEL_COMMANDS[model]

    # Remove CLAUDECODE env var to allow nested claude invocation
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(prompt.encode()),
            timeout=timeout,
        )
    except TimeoutError:
        proc.kill()
        raise TimeoutError(f"Model {model} timed out after {timeout}s") from None

    if proc.returncode != 0:
        raise RuntimeError(f"Model {model} failed: {stderr.decode()}")

    return stdout.decode()


def invoke_sync(prompt: str, model: str = "claude", timeout: int = DEFAULT_TIMEOUT) -> str:
    """Synchronous wrapper around invoke."""
    return asyncio.run(invoke(prompt, model, timeout))


async def invoke_concurrent(
    prompts: list[str],
    model: str = "claude",
    timeout: int = DEFAULT_TIMEOUT,
    max_concurrent: int = 3,
) -> list[str | Exception]:
    """Invoke model on multiple prompts concurrently with bounded parallelism.

    Returns a list parallel to prompts — each element is either the response
    string or the Exception that occurred for that prompt.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _guarded(prompt: str) -> str:
        async with sem:
            return await invoke(prompt, model, timeout)

    return await asyncio.gather(
        *[_guarded(p) for p in prompts],
        return_exceptions=True,
    )


def invoke_concurrent_sync(
    prompts: list[str],
    model: str = "claude",
    timeout: int = DEFAULT_TIMEOUT,
    max_concurrent: int = 3,
) -> list[str | Exception]:
    """Synchronous wrapper around invoke_concurrent."""
    return asyncio.run(invoke_concurrent(prompts, model, timeout, max_concurrent))
