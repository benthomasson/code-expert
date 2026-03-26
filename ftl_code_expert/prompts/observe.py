"""Observation-gathering prompt for explanation."""

OBSERVE_PROMPT = """You are preparing to explain code to a developer who is new to this codebase.

## Question

The developer wants to understand: **{question}**

## Repository Structure (top-level)

```
{tree}
```

## Your Task

Identify what specific information you need to answer this question well.
Do NOT answer the question yet. Only request observations.

## Available Observation Tools

| Tool | Purpose | Params |
|------|---------|--------|
| `grep` | Search for a pattern in the codebase | `pattern`, `glob` (default: *.py) |
| `read_file` | Read a file's contents | `file_path`, `start_line`, `max_lines` |
| `list_directory` | List contents of a directory | `dir_path`, `max_depth` |
| `find_symbol` | Find where a class/function is defined | `symbol` |
| `find_usages` | Find where a symbol is used | `symbol` |
| `file_imports` | Extract imports from a file | `file_path` |

## Output Format

Output a JSON array of observation requests:

```json
[
  {{"name": "descriptive_name", "tool": "tool_name", "params": {{"param": "value"}}}},
  ...
]
```

## Guidelines

- Request 3-8 observations. Be targeted, not exhaustive.
- Start with `find_symbol` or `grep` to locate relevant code, then `read_file` to read it.
- If you can identify the right file from the tree, go straight to `read_file`.
- For conceptual questions, use `grep` to find where the concept appears in code.

Now output your observation requests as JSON:
"""


def build_observe_prompt(question: str, tree: str) -> str:
    """Build the observation-gathering prompt."""
    return OBSERVE_PROMPT.format(question=question, tree=tree)
