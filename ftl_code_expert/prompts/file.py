"""Prompt template for file explanation."""

from .common import BELIEFS_INSTRUCTIONS, TOPICS_INSTRUCTIONS


def build_file_prompt(
    file_path: str,
    file_content: str,
    imports: list[str] | None = None,
    imported_by: list[str] | None = None,
    repo_context: str | None = None,
) -> str:
    """
    Build prompt for explaining a single file.

    Args:
        file_path: Relative path to the file
        file_content: Full file content
        imports: List of import statements in the file
        imported_by: List of files that import this file
        repo_context: Brief repo structure context
    """
    sections = [
        "You are a senior software engineer explaining code to a colleague.",
        f"Explain the following file: `{file_path}`",
        "",
    ]

    if repo_context:
        sections.extend([
            "## Repository Context",
            "",
            "```",
            repo_context,
            "```",
            "",
        ])

    sections.extend([
        "## File Content",
        "",
        f"```{_guess_language(file_path)}",
        file_content,
        "```",
        "",
    ])

    if imports:
        sections.extend([
            "## Imports",
            "",
        ])
        for imp in imports:
            sections.append(f"- `{imp}`")
        sections.append("")

    if imported_by:
        sections.extend([
            "## Imported By",
            "",
        ])
        for f in imported_by:
            sections.append(f"- `{f}`")
        sections.append("")

    sections.extend([
        "## Instructions",
        "",
        "Explain this file covering:",
        "",
        "1. **Purpose**: What is this file's role in the project? What responsibility does it own?",
        "2. **Key Components**: Important classes, functions, and constants — with their contracts",
        "3. **Patterns**: Design patterns, idioms, or architectural conventions used",
        "4. **Dependencies**: What it depends on (imports) and what depends on it (imported_by)",
        "5. **Flow**: How the code executes (control flow, data transformations)",
        "6. **Invariants**: Any rules this code enforces — preconditions, validation, ordering constraints",
        "7. **Error Handling**: How errors are produced, propagated, or swallowed",
        "",
        "Format your response as markdown.",
        "Be concrete — reference specific functions, classes, and line-level details.",
        "Focus on what a developer maintaining this code needs to know.",
        TOPICS_INSTRUCTIONS,
        BELIEFS_INSTRUCTIONS,
    ])

    return "\n".join(sections)


def _guess_language(file_path: str) -> str:
    """Guess language from file extension for syntax highlighting."""
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".rb": "ruby",
        ".sh": "bash",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".toml": "toml",
        ".json": "json",
        ".md": "markdown",
        ".sql": "sql",
        ".html": "html",
        ".css": "css",
    }
    for ext, lang in ext_map.items():
        if file_path.endswith(ext):
            return lang
    return ""
