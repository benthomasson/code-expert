"""Prompt template for repo scanning."""

SCAN_PROMPT = """\
You are a senior software architect performing initial reconnaissance on a codebase.
Your goal is to identify the most important files and components to understand.

## Repository Structure

```
{tree}
```

{config_section}

{readme_section}

{entry_points_section}

## Instructions

Analyze this repository structure and identify:

1. **Critical Files** (5-15): The files that are most important to understand first.
   For each, explain WHY it matters (entry point? core logic? shared types? config?).

2. **Architecture Sketch**: In 3-5 sentences, what is the likely architecture?

3. **Module Map**: Group the top-level directories/packages by responsibility.

4. **Exploration Strategy**: What order should these files be explored in?
   Start with entry points and shared types, then work outward.

For each critical file, output a topic using this format so they can be queued:

## Topics to Explore

- [file] `path/to/file.py` — Why this file matters
- [function] `path/to/file.py:main_function` — Why this function matters
- [general] `concept-name` — Why this concept matters

Aim for 8-15 topics that form a coherent exploration path through the codebase.
"""


def build_scan_prompt(
    tree: str,
    config_content: str | None = None,
    readme_content: str | None = None,
    entry_points: list[str] | None = None,
) -> str:
    """Build the scan prompt."""
    config_section = ""
    if config_content:
        config_section = f"## Project Configuration\n\n```\n{config_content}\n```"

    readme_section = ""
    if readme_content:
        readme_section = f"## README\n\n{readme_content}"

    entry_points_section = ""
    if entry_points:
        entry_points_section = "## Entry Points\n\n" + "\n".join(f"- {ep}" for ep in entry_points)

    return SCAN_PROMPT.format(
        tree=tree,
        config_section=config_section,
        readme_section=readme_section,
        entry_points_section=entry_points_section,
    )
