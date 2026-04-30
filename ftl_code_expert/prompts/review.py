"""Prompt template for LLM-based review of proposed beliefs."""

REVIEW_PROMPT = """\
You are reviewing proposed beliefs extracted from code exploration notes. Your job is to filter \
out low-quality proposals that should NOT be added to the belief network.

## Rejection Criteria

Reject a proposal if it matches ANY of these categories:

1. **Meta** — About the belief network itself, not the codebase. Examples: node counts, cascade \
analyses, compaction strategies, retraction history, knowledge base statistics.
2. **Duplicate** — Same claim already exists in the current beliefs (listed below), or is a \
trivial rewording of an existing belief.
3. **Ephemeral** — Point-in-time snapshots that expire immediately. Examples: "file has N lines", \
"module has N functions", "currently N items" (specific counts that change with any edit).
4. **Speculative** — Cascade risk analyses, estimates, editorial judgments about what "should" \
happen, predictions. Not grounded in observable code facts.
5. **Trivial** — Obvious facts any developer can see from imports, naming, or file structure. \
Examples: "module X imports Y", "class Z has a constructor", "file uses Python 3 syntax".

## What to KEEP

Keep proposals that are:
- Architecture invariants ("All requests go through the middleware chain")
- API contracts ("create_agent() requires a config dict and returns Agent or raises ConfigError")
- Data flow rules ("User input is sanitized in the router, not in handlers")
- Error semantics ("Failed observations return dicts with 'error' key, never raise")
- Concurrency rules ("Agent execution uses asyncio.gather; no threading")
- Dependency constraints ("Core module never imports from plugins")
- Configuration requirements ("DATABASE_URL must be set before models import")
- Security properties ("Host key verification is enabled by default")
- Durable enough to remain true for at least a week

## Existing Beliefs

{existing_beliefs}

## Proposals to Review

{proposals}

## Output Format

For EACH proposal, output exactly one line:

ACCEPT belief-id
or
REJECT belief-id reason-category: brief explanation

Categories: meta, duplicate, ephemeral, speculative, trivial

Review every proposal — do not skip any. Output nothing else.
"""
