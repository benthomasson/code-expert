"""Prompt templates for code expert."""

from .common import BELIEFS_INSTRUCTIONS, TOPICS_INSTRUCTIONS
from .diff import build_diff_prompt, build_diff_summary_prompt
from .file import build_file_prompt
from .function import build_function_prompt
from .observe import build_observe_prompt
from .derive import DERIVE_BELIEFS_PROMPT
from .propose import PROPOSE_BELIEFS_CODE
from .review import REVIEW_PROMPT
from .repo import build_repo_prompt
from .scan import build_scan_prompt
from .spec import GENERATE_SPEC_PROMPT

__all__ = [
    "BELIEFS_INSTRUCTIONS",
    "DERIVE_BELIEFS_PROMPT",
    "GENERATE_SPEC_PROMPT",
    "PROPOSE_BELIEFS_CODE",
    "REVIEW_PROMPT",
    "TOPICS_INSTRUCTIONS",
    "build_diff_prompt",
    "build_diff_summary_prompt",
    "build_file_prompt",
    "build_function_prompt",
    "build_observe_prompt",
    "build_repo_prompt",
    "build_scan_prompt",
]
