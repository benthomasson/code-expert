"""Prompt template for diff explanation."""

from .common import BELIEFS_INSTRUCTIONS, TOPICS_INSTRUCTIONS


def build_diff_prompt(
    diff_content: str,
    commit_log: str | None = None,
    changed_files_summary: list[str] | None = None,
) -> str:
    """
    Build prompt for explaining a diff (what changed and why).

    Args:
        diff_content: Unified diff output
        commit_log: Commit messages for the changes
        changed_files_summary: List of changed file paths
    """
    sections = [
        "You are a senior software engineer explaining code changes to a colleague.",
        "Explain what changed in this diff and why.",
        "",
    ]

    if commit_log:
        sections.extend([
            "## Commit History",
            "",
            "```",
            commit_log,
            "```",
            "",
        ])

    if changed_files_summary:
        sections.extend([
            "## Changed Files",
            "",
        ])
        for f in changed_files_summary:
            sections.append(f"- `{f}`")
        sections.append("")

    sections.extend([
        "## Diff",
        "",
        "```diff",
        diff_content,
        "```",
        "",
        "## Instructions",
        "",
        "Explain these changes covering:",
        "",
        "1. **Summary**: One-paragraph overview of what changed",
        "2. **Motivation**: Why were these changes made? (infer from commit messages and code)",
        "3. **File-by-File Breakdown**: For each changed file, explain what changed and why",
        "4. **Impact**: What behavior changes as a result?",
        "5. **Risks**: Any potential issues or things to watch out for",
        "",
        "Format your response as markdown.",
        "Focus on the 'why' — don't just describe what lines were added/removed.",
        TOPICS_INSTRUCTIONS,
        BELIEFS_INSTRUCTIONS,
    ])

    return "\n".join(sections)
