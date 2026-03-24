"""Prompt template for deriving deeper reasoning chains from existing beliefs."""

DERIVE_BELIEFS_PROMPT = """\
You are a reasoning architect analyzing a belief network about a codebase. Your task is to \
identify opportunities for deeper derived conclusions by combining existing beliefs.

## Background

A Reason Maintenance System (RMS) tracks beliefs with justifications and automatic retraction \
cascades. There are three kinds of nodes:

1. **Base premises** (depth-0): Observable code facts with no justifications
2. **Derived conclusions** (depth-1+): Justified by antecedents via SL (support-list) rules
3. **Outlist-gated conclusions**: Justified by antecedents UNLESS certain nodes are IN \
   (the conclusion is OUT while the outlist node is IN, and flips IN when it goes OUT)

When a base premise is retracted, all derived conclusions that depend on it cascade OUT \
automatically. This is the key value — maintaining consistency without manual intervention.

## Your Task

Given the existing beliefs and derived conclusions below, propose NEW derived conclusions that:

1. **Combine existing conclusions** into higher-level claims (depth N+1 from depth N)
2. **Group related base beliefs** into thematic conclusions (new depth-1)
3. **Connect positive and negative chains** via outlist semantics — where a positive claim \
   should only hold when a negative claim (bug/issue/gap) is OUT

## Rules

- Each proposed conclusion must have at least 2 antecedents
- Antecedents must be existing belief IDs from the list below
- Prefer combining existing derived beliefs (deeper chains) over just grouping base beliefs
- For outlist-gated beliefs: the antecedent should be a positive claim, the unless should be \
  a negative claim (bug, gap, issue, fragility)
- Don't propose conclusions that merely restate a single antecedent
- Don't propose conclusions whose antecedents are unrelated (no forced connections)
- Each conclusion should represent a genuine emergent property or architectural insight

## Output Format

For each proposed conclusion, output EXACTLY this format:

### DERIVE <belief-id-in-kebab-case>
<one-line claim text>
- Antecedents: <comma-separated list of existing belief IDs>
- Label: <brief justification rationale>

For outlist-gated conclusions:

### GATE <belief-id-in-kebab-case>
<one-line claim text>
- Antecedents: <comma-separated list of existing belief IDs>
- Unless: <comma-separated list of belief IDs that must be OUT>
- Label: <brief justification rationale>

---

## Existing Beliefs

{beliefs_section}

## Existing Derived Conclusions

{derived_section}

## Statistics

- Total IN beliefs: {total_in}
- Existing derived: {total_derived}
- Max depth: {max_depth}
"""
