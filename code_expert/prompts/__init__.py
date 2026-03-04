"""Prompt templates for code expert."""

from .common import BELIEFS_INSTRUCTIONS, TOPICS_INSTRUCTIONS
from .diff import build_diff_prompt
from .file import build_file_prompt
from .function import build_function_prompt
from .observe import build_observe_prompt
from .propose import PROPOSE_BELIEFS_CODE
from .repo import build_repo_prompt
from .scan import build_scan_prompt

__all__ = [
    "BELIEFS_INSTRUCTIONS",
    "PROPOSE_BELIEFS_CODE",
    "TOPICS_INSTRUCTIONS",
    "build_diff_prompt",
    "build_file_prompt",
    "build_function_prompt",
    "build_observe_prompt",
    "build_repo_prompt",
    "build_scan_prompt",
]
