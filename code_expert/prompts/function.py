"""Prompt template for function/class explanation."""

from .common import BELIEFS_INSTRUCTIONS, TOPICS_INSTRUCTIONS


def build_function_prompt(
    file_path: str,
    symbol_name: str,
    symbol_source: str,
    full_file_content: str | None = None,
    related_tests: list[str] | None = None,
) -> str:
    """
    Build prompt for explaining a specific function or class.

    Args:
        file_path: Path to the source file
        symbol_name: Name of the function or class
        symbol_source: Extracted source code of the symbol
        full_file_content: Full file for additional context
        related_tests: Paths to related test files
    """
    sections = [
        "You are a senior software engineer explaining code to a colleague.",
        f"Explain the following symbol `{symbol_name}` from `{file_path}`.",
        "",
        "## Source Code",
        "",
        "```python",
        symbol_source,
        "```",
        "",
    ]

    if full_file_content:
        sections.extend([
            "## Full File Context",
            "",
            f"The symbol is defined in `{file_path}`. Here is the full file for context:",
            "",
            "```python",
            full_file_content,
            "```",
            "",
        ])

    if related_tests:
        sections.extend([
            "## Related Tests",
            "",
        ])
        for test in related_tests:
            sections.append(f"- `{test}`")
        sections.append("")

    sections.extend([
        "## Instructions",
        "",
        "Explain this function/class covering:",
        "",
        "1. **Purpose**: What does it do and why does it exist?",
        "2. **Contract**: What are the preconditions, postconditions, and invariants?",
        "3. **Parameters**: What each parameter means, expected types/values, and edge cases",
        "4. **Return Value**: What it returns, under what conditions, and what the caller must handle",
        "5. **Algorithm**: Step-by-step walkthrough of the logic",
        "6. **Side Effects**: Any mutations, I/O, state changes, or external calls",
        "7. **Error Handling**: What exceptions can be raised, what errors are returned vs swallowed",
        "8. **Usage Patterns**: How this is typically called, and any caller obligations",
        "9. **Dependencies**: What internal/external modules this relies on",
        "",
        "Format your response as markdown.",
        "Be precise — explain the actual logic, not just paraphrase the code.",
        "Identify any assumptions the code makes that are not enforced by the type system.",
        TOPICS_INSTRUCTIONS,
        BELIEFS_INSTRUCTIONS,
    ])

    return "\n".join(sections)
