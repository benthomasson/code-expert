"""Prompt template for PRD generation from beliefs and specs."""

GENERATE_PRD_PROMPT = """\
You are generating a Product Requirements Document (PRD) from verified \
technical specifications and beliefs about a software product.

## Product

**Name:** {product_name}
**Domain:** {domain}

## Derived Beliefs (verified high-level conclusions)

These are the highest-level claims about the product, each backed by a \
chain of verified code observations:

{derived_beliefs}

## Specifications

{specs_text}

## Instructions

Generate a Product Requirements Document in markdown with this structure:

### Header
```
# PRD: {product_name}

*Generated from belief registry ({belief_count} beliefs, {spec_count} specs)*
*Source: {domain} knowledge base*
```

### Sections

1. **Executive Summary** — 2-3 paragraphs. What is {product_name}, who is it \
for, why does it exist. Ground every claim in a belief ID.

2. **Target Users** — Define 3-4 user personas with their pain points and how \
{product_name} addresses them. Use beliefs for evidence.

3. **Product Capabilities** — Group into major capability areas. For each:
   - Capability name and one-line description
   - User stories (As a [persona], I want [capability], so that [outcome])
   - Acceptance criteria derived from sacred contract beliefs
   - Belief IDs that verify this capability

4. **Architecture Overview** — High-level system diagram (as ASCII art). Show \
how the major components interact. Reference the specs.

5. **Competitive Positioning** — Comparison table if competitor beliefs exist. \
Be specific about what's better, equivalent, and missing.

6. **Requirements Hierarchy** — Show the belief dependency tree as a \
requirements hierarchy. Top-level requirements (depth 5-6 derived beliefs) \
depend on sub-requirements (depth 3-4) which depend on foundational \
capabilities (depth 1-2 observations). This IS the requirements traceability \
matrix.

7. **Risks and Constraints** — Known limitations from anti-pattern beliefs, \
design gaps that remain open, platform dependencies.

8. **Success Metrics** — Derived from verification tables in the specs. \
What's runtime-confirmed vs code-analysis-only.

### Rules

- Every claim MUST reference a belief ID. No unsourced claims.
- Write for a product manager / technical decision maker, not a developer.
- Be specific about what the product does and doesn't do.
- Use the derived belief chains to show WHY capabilities matter together.
- Output the full document directly. Do NOT ask for permission, \
confirmation, or approval. Do NOT say "the PRD is ready" — just output it.
- Do not invent claims. If the beliefs don't cover something, leave it out.
"""
