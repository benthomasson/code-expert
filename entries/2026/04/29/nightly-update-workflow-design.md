# Nightly Update Workflow Design

**Date:** 2026-04-29
**Time:** 16:12

## Problem

Every night before logging off, I want to run a single command that brings the knowledge base current with the day's commits and produces a morning summary I can read over coffee. Today this requires running 5+ commands manually and reviewing intermediate output. The goal is a fully automated pipeline.

## Desired Workflow

```
code-expert nightly
  1. walk-commits --since-last     # grab all new commits, explore changed files
  2. propose-beliefs --auto        # extract beliefs from new entries, auto-accept all
  3. accept-beliefs                # import into reasons.db
  4. derive --exhaust              # compute all logical consequences repeatedly until stable
  5. generate-summary              # morning report entry
```

One command. No human review in the loop. Results waiting in the morning.

## Summary Report Requirements

The morning summary entry should have these sections:

### New Findings (since last nightly run)
- **New gated OUT beliefs** — conclusions that are now blocked because a negative finding became active. These are the most actionable: something broke or was discovered that invalidates a positive claim.
- **New negative IN beliefs** — bugs, gaps, security issues, fragilities that were just extracted from today's commits. These are the raw problems.

### Critical Watch List (always shown)
- **Highly important gated and negative beliefs** — even if they are not new, critical issues should always be surfaced. Security vulnerabilities, data loss risks, production safety gaps. These should never scroll off the radar just because they were found last week.
- Importance could be determined by keywords (security, injection, data loss, production, authentication) or by explicit priority tags on beliefs.

### Statistics
- Commits processed, files explored, entries created
- Beliefs proposed/accepted, derivations added
- Current belief counts (IN/OUT/STALE)
- Topic queue state

## Current State — What Exists

| Step | Status | Gap |
|------|--------|-----|
| walk-commits --since-last | Works | None |
| explore (within walk-commits) | Works | None — walk-commits calls explore per file |
| propose-beliefs | Works | No --auto flag. Requires manual ACCEPT/REJECT marking |
| accept-beliefs | Works | Depends on proposals being pre-marked |
| derive --auto | Works | Need to verify --exhaust exists or build it |
| file-issues | Works | External side effects — should it be in the nightly? |
| Summary report | Does not exist | Entirely new |
| Orchestrator | Does not exist | Need a top-level `nightly` command |

## Design Questions

### 1. Auto-accept for propose-beliefs
The simplest approach: add `--auto` flag that marks all proposals as ACCEPT before importing. This matches `derive --auto`. No confidence threshold — the human reviews the morning summary and can retract anything that looks wrong. The belief system already supports retraction, so auto-accepting is low-risk.

Alternative: confidence threshold or keyword filtering. But this adds complexity and the RMS (reason maintenance system) already handles corrections gracefully via retraction. Lean toward accept-all.

### 2. derive --exhaust
`derive --auto` adds one round of derivations. But new derivations may enable further derivations. `--exhaust` should loop until no new derivations are proposed — fixed-point computation. Need to check if this exists or needs to be built.

### 3. Command name
`code-expert nightly` reads well. Alternative: `code-expert update` (more general, could be run anytime). Leaning toward `update` since there is nothing inherently nightly about it — you might want to run it mid-day too.

### 4. Importance / severity for the watch list
Options:
- **Keyword matching** — scan belief text for security, injection, vulnerability, data loss, authentication, production, etc. Simple, no schema changes.
- **Priority field** — add a priority to beliefs in reasons.db. More structured but requires schema work.
- **Tags** — beliefs already have IDs; could add tags. Medium effort.

Keyword matching is the pragmatic starting point. Can add structured priority later.

### 5. Should file-issues be part of the nightly?
Probably not by default. Filing GitHub issues is an external side effect — you don't want to wake up to 15 issues you haven't reviewed. Better to have the summary report list what *would* be filed, and let the human run `file-issues` after reviewing the morning summary.

The nightly command could accept `--file-issues` to opt in explicitly.

### 6. Error handling
The pipeline should continue on non-fatal errors. If propose-beliefs fails on one entry, skip it and keep going. Log errors in the summary. Only abort on truly fatal conditions (repo not initialized, no commits to process, reasons.db missing).

## Proposed Implementation Steps

1. **Add `--auto` flag to `propose-beliefs`** — marks all proposals as ACCEPT and calls accept-beliefs inline
2. **Add or verify `--exhaust` flag for `derive`** — loop until fixed point
3. **Build `generate-summary` command** — queries reasons.db for new and critical beliefs, writes a summary entry
4. **Build `update` orchestrator command** — chains the pipeline steps, handles errors, passes through relevant flags
5. **Add importance detection** — keyword-based scanning for critical belief highlighting

## Open Questions

- What format should the summary entry use? Plain markdown sections? Or something more structured that could be parsed later?
- Should the summary include a diff of belief state (what changed IN/OUT since last run)?
- Should there be a `--dry-run` mode for the full pipeline?
