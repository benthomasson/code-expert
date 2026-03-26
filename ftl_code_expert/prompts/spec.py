"""Prompt template for spec generation from beliefs."""

GENERATE_SPEC_PROMPT = """\
You are generating a technical specification from a belief registry.

## Component

**Name:** {component}
**Files:** {source_files}

## Beliefs ({belief_count} total)

These beliefs have been extracted from code analysis entries. Each has an ID, \
status (IN/OUT), type (OBSERVATION/DERIVED), claim text, and source entry.

{beliefs_text}

## Source Code

{source_code}

## Instructions

Generate a specification document in markdown with this structure:

### Header
```
# Specification: {component}

*Generated from belief registry ({belief_count} beliefs)*
*Source: {domain} knowledge base*
```

### Sections

1. **Purpose** — What this component does, 2-3 sentences. Reference the most \
fundamental belief.

2. **Sacred Contract** — MUST behaviors confirmed by code analysis. Group by \
sub-topic (e.g., "Citation Invariants", "Synthesis Behavior"). For each:
   - One-line summary in bold
   - Brief explanation
   - Code snippet if the belief references specific code patterns
   - `> **Belief**: belief-id` attribution

3. **Implementation Details** — HOW it works internally. These may change \
without breaking the contract.

4. **Anti-Patterns** — MUST NOT behaviors, each linked to a belief.

5. **Files** — Table of source files and their roles.

6. **Verification** — Which beliefs are runtime-confirmed vs code-only.

### Rules

- Only include beliefs with status [IN]. Skip [OUT] beliefs entirely.
- Every claim MUST link to a belief ID. No unsourced claims.
- Include code snippets where the belief text references specific functions \
or patterns. Use comments to show the relevant logic, not full code blocks.
- If a belief has type DERIVED and depends on other beliefs, mention the \
dependency chain briefly.
- Keep it concise. Each belief gets 2-5 lines, not a paragraph.
- Do not invent claims. If the beliefs don't cover something, leave it out.
- Output the full specification document directly. Do NOT ask for permission, \
confirmation, or approval. Do NOT say "the spec is ready" — just output it.
"""
