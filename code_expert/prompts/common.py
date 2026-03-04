"""Shared prompt components."""

# Appended to every prompt so the model surfaces follow-up explorations.
TOPICS_INSTRUCTIONS = """
## Topics to Explore

After your explanation, add a section titled "## Topics to Explore".
List 3-5 things the reader should explore next to deepen their understanding.
Each item MUST use this exact format:

- [kind] `target` — Description

Where:
- **kind** is one of: file, function, repo, diff, general
- **target** is the exploration target:
  - For file: the file path (e.g., `src/auth/client.py`)
  - For function: file:symbol (e.g., `src/auth/client.py:login`)
  - For general: a short label (e.g., `dataverse-integration`)
- **Description** explains why this is worth exploring

Example:
- [file] `src/workflow/executor.py` — Orchestrates the plan-execute-synthesize loop
- [function] `src/router.py:route_request` — Decides which agent handles each request
- [general] `error-handling-strategy` — How failures propagate across agent boundaries
"""

# Appended to every prompt so the model surfaces beliefs.
BELIEFS_INSTRUCTIONS = """
## Beliefs

After your explanation, add a section titled "## Beliefs".
List 2-5 factual claims about this code that a developer should know.
These should be concrete, testable assertions — not opinions or vague observations.

Each item MUST use this exact format:

- `belief-id-in-kebab-case` — One-line factual claim

Good beliefs are about:
- **Architecture invariants**: "All agent requests flow through the router before reaching department agents"
- **API contracts**: "execute_plan returns a PlanResult and never raises; errors are captured in result.errors"
- **Data ownership**: "The Dataverse mart is the single source of truth for department metrics"
- **Dependency rules**: "The synthesizer module imports from agents but agents never import from synthesizer"
- **Configuration constraints**: "MODEL_TIMEOUT must be set before any agent is instantiated"
- **Error semantics**: "A failed observation returns a dict with 'error' key, never raises an exception"

Bad beliefs (avoid):
- Vague: "This module is important"
- Obvious: "This file contains Python code"
- Opinion: "This is well-designed"
- Redundant: Repeating what the code literally says without insight
"""
