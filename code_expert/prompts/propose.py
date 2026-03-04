"""Prompt template for proposing beliefs from code exploration entries."""

PROPOSE_BELIEFS_CODE = """\
You are extracting architectural and behavioral claims from code exploration notes.

These entries were written while studying a codebase. Extract factual claims that a \
developer needs to know to work in this codebase effectively.

For each significant claim, output a proposed belief in this exact format.
The square brackets around ACCEPT or REJECT are REQUIRED — do not omit them:

### [ACCEPT] <belief-id-in-kebab-case>
<one-line factual claim>
- Source: <path to the entry file>

Or to reject:

### [REJECT] <belief-id-in-kebab-case>
<one-line reason for rejection>
- Source: <path to the entry file>

Good beliefs for codebases:
- Architecture invariants: "All HTTP handlers go through the middleware chain before reaching route logic"
- API contracts: "create_agent() requires a valid config dict and returns an Agent or raises ConfigError"
- Data flow rules: "User input is sanitized in the router, not in individual handlers"
- Dependency constraints: "The core module never imports from plugins; plugins import from core"
- Configuration requirements: "DATABASE_URL must be set before import of models module"
- Error semantics: "Failed observations return dicts with 'error' key, never raise exceptions"
- Performance constraints: "Topic queue is O(n) scanned on each pop; acceptable for < 1000 topics"
- Concurrency rules: "Agent execution uses asyncio.gather for parallelism; no threading"

Bad beliefs (avoid):
- Vague generalizations: "This is a well-structured project"
- Obvious facts: "This file uses Python imports"
- Opinions: "This pattern is better than alternatives"
- Unstable details: Line numbers, exact version pins (these change too often)

Rules:
- Each belief should be a single, testable factual claim about the code
- Use kebab-case IDs that are descriptive (e.g., router-dispatches-by-complexity)
- Aim for 3-8 beliefs per entry
- Prefer claims that would break things if violated (invariants) over trivia

---

ENTRIES:

{entries}
"""
