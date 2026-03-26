"""
Observation tools for code explanation.

Instead of dumping entire repo contents into the prompt, the model
requests specific observations (grep, file reads, etc.) and gets
targeted results back. This keeps prompts focused and fast.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from pathlib import Path
from typing import Any


async def grep(pattern: str, repo_path: str, glob: str = "*.py", max_results: int = 30) -> dict[str, Any]:
    """
    Search for a pattern in the codebase.

    Args:
        pattern: Regex pattern to search for
        repo_path: Repository path
        glob: File glob pattern (default: *.py)
        max_results: Maximum number of results

    Returns:
        Dict with matching files and lines
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "grep", "-Ern", f"--include={glob}", pattern, repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            return {"error": "grep timed out", "pattern": pattern}

        matches = []
        for line in stdout.decode().strip().split("\n"):
            if line and ":" in line:
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    matches.append({
                        "file": parts[0].replace(repo_path + "/", ""),
                        "line": int(parts[1]),
                        "text": parts[2].strip()[:120],
                    })

        return {
            "pattern": pattern,
            "matches": matches[:max_results],
            "total_count": len(matches),
        }
    except Exception as e:
        return {"error": str(e), "pattern": pattern}


async def read_file(file_path: str, repo_path: str, start_line: int = 0, max_lines: int = 200) -> dict[str, Any]:
    """
    Read a file's contents.

    Args:
        file_path: Path to the file (relative to repo)
        repo_path: Repository root
        start_line: Starting line (0-indexed)
        max_lines: Maximum lines to return

    Returns:
        Dict with file content and metadata
    """
    try:
        full_path = Path(repo_path) / file_path
        if not full_path.is_file():
            return {"error": f"File not found: {file_path}", "file": file_path}

        content = full_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        total_lines = len(lines)

        selected = lines[start_line:start_line + max_lines]

        return {
            "file": file_path,
            "total_lines": total_lines,
            "start_line": start_line,
            "lines_returned": len(selected),
            "content": "\n".join(selected),
        }
    except Exception as e:
        return {"error": str(e), "file": file_path}


async def list_directory(dir_path: str, repo_path: str, max_depth: int = 2) -> dict[str, Any]:
    """
    List contents of a directory.

    Args:
        dir_path: Directory path (relative to repo)
        repo_path: Repository root
        max_depth: Max depth to traverse

    Returns:
        Dict with directory listing
    """
    try:
        full_path = Path(repo_path) / dir_path
        if not full_path.is_dir():
            return {"error": f"Directory not found: {dir_path}", "dir": dir_path}

        skip_dirs = {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            ".tox", ".eggs", "dist", "build", ".mypy_cache",
        }

        entries = []

        def _walk(p: Path, depth: int, prefix: str):
            if depth > max_depth:
                return
            try:
                items = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
            except PermissionError:
                return
            for item in items:
                if item.name in skip_dirs or item.name.startswith("."):
                    continue
                rel = str(item.relative_to(Path(repo_path)))
                if item.is_dir():
                    entries.append({"path": rel + "/", "type": "dir"})
                    _walk(item, depth + 1, prefix)
                else:
                    entries.append({"path": rel, "type": "file"})

        _walk(full_path, 1, "")

        return {
            "dir": dir_path,
            "entries": entries[:100],
            "total_entries": len(entries),
        }
    except Exception as e:
        return {"error": str(e), "dir": dir_path}


async def find_symbol(symbol: str, repo_path: str) -> dict[str, Any]:
    """
    Find where a symbol (class, function, variable) is defined.

    Args:
        symbol: Symbol name to find
        repo_path: Repository path

    Returns:
        Dict with definition locations
    """
    try:
        # Search for definitions
        patterns = [
            f"^(class|def|async def) {symbol}[(:  ]",
            f"^{symbol}\\s*=",
        ]

        definitions = []
        for pattern in patterns:
            proc = await asyncio.create_subprocess_exec(
                "grep", "-Ern", "--include=*.py", pattern, repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            except asyncio.TimeoutError:
                proc.kill()
                continue

            for line in stdout.decode().strip().split("\n"):
                if line and ":" in line:
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        definitions.append({
                            "file": parts[0].replace(repo_path + "/", ""),
                            "line": int(parts[1]),
                            "text": parts[2].strip()[:120],
                        })

        return {
            "symbol": symbol,
            "definitions": definitions[:20],
            "count": len(definitions),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


async def find_usages(symbol: str, repo_path: str) -> dict[str, Any]:
    """
    Find where a symbol is used (imported, called, referenced).

    Args:
        symbol: Symbol to search for
        repo_path: Repository path

    Returns:
        Dict with usage locations
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "grep", "-Frn", "--include=*.py", symbol, repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            return {"error": "grep timed out", "symbol": symbol}

        usages = []
        for line in stdout.decode().strip().split("\n"):
            if line and ":" in line:
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    usages.append({
                        "file": parts[0].replace(repo_path + "/", ""),
                        "line": int(parts[1]),
                        "text": parts[2].strip()[:120],
                    })

        return {
            "symbol": symbol,
            "usages": usages[:30],
            "total_count": len(usages),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


async def file_imports(file_path: str, repo_path: str) -> dict[str, Any]:
    """
    Extract import statements from a file.

    Args:
        file_path: Path to the file (relative to repo)
        repo_path: Repository root

    Returns:
        Dict with imports
    """
    import ast

    try:
        full_path = Path(repo_path) / file_path
        source = full_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        imports = []
        from_imports = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                from_imports.append({"module": module, "names": names})

        return {
            "file": file_path,
            "imports": imports,
            "from_imports": from_imports,
        }
    except Exception as e:
        return {"error": str(e), "file": file_path}


# Registry of all observation tools
OBSERVATION_TOOLS: dict[str, Any] = {
    "grep": grep,
    "read_file": read_file,
    "list_directory": list_directory,
    "find_symbol": find_symbol,
    "find_usages": find_usages,
    "file_imports": file_imports,
}


async def run_observation(name: str, tool: str, params: dict[str, Any], repo_path: str) -> dict[str, Any]:
    """Run a single observation tool."""
    if tool not in OBSERVATION_TOOLS:
        return {"name": name, "result": {"error": f"Unknown tool: {tool}"}}

    tool_func = OBSERVATION_TOOLS[tool]

    sig = inspect.signature(tool_func)
    valid_params = set(sig.parameters.keys())

    filtered_params = {k: v for k, v in params.items() if k in valid_params}

    if "repo_path" in valid_params and "repo_path" not in filtered_params:
        filtered_params["repo_path"] = repo_path

    try:
        result = await tool_func(**filtered_params)
    except TypeError as e:
        result = {"error": f"Parameter error: {e}"}

    return {"name": name, "tool": tool, "result": result}


async def run_observations(observations: list[dict[str, Any]], repo_path: str) -> dict[str, Any]:
    """Run multiple observations in parallel."""
    results = {}
    tasks = []
    task_names = []

    for obs in observations:
        name = obs.get("name", "unnamed")
        tool = obs.get("tool")
        params = obs.get("params", {})

        if not tool:
            results[name] = {"error": "No tool specified"}
            continue

        tasks.append(run_observation(name, tool, params, repo_path))
        task_names.append(name)

    if tasks:
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        for name, result in zip(task_names, task_results):
            if isinstance(result, Exception):
                results[name] = {"error": str(result)}
            else:
                results[name] = result["result"]

    return results


# --- Parsing observation requests from model output ---

def parse_observation_requests(response: str) -> list[dict]:
    """Parse observation requests from model response."""
    json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    try:
        result = json.loads(response.strip())
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    return []
