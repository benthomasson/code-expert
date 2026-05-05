---
name: code-expert
description: Build expert knowledge bases from codebases — explore, explain, extract beliefs
argument-hint: "[init|scan|explain|explore|walk-commits|update|topics|propose-beliefs|review-proposals|accept-beliefs|derive|generate-summary|file-issues|status]"
allowed-tools: Bash(code-expert *), Bash(uv run code-expert *), Bash(uvx *ftl-code-expert*), Read, Grep, Glob
---

# Code Expert

Build expert knowledge bases from codebases by combining code exploration with belief extraction.

## How to Run

Try these in order until one works:
1. `code-expert $ARGUMENTS` (if installed via `uv tool install`)
2. `uv run code-expert $ARGUMENTS` (if in the repo with pyproject.toml)
3. `uvx --from git+https://github.com/benthomasson/ftl-code-expert code-expert $ARGUMENTS` (fallback)

## Typical Workflow

```bash
code-expert init ~/git/some-project --domain "Web framework"
code-expert scan                           # identify key files, populate topic queue
code-expert explore                        # explain next topic, create entry
code-expert explore --pick 1,3,8           # explore multiple by index (stable indices)
code-expert explore --skip                 # skip one
code-expert topics                         # see exploration queue
code-expert propose-beliefs                # extract beliefs from entries
code-expert review-proposals               # LLM quality filter
# edit proposed-beliefs.md if needed
code-expert accept-beliefs                 # import accepted beliefs
code-expert derive --auto                  # derive deeper reasoning chains
code-expert status                         # dashboard

# Automated nightly update (does all of the above)
code-expert update --since-last            # full pipeline + morning summary
```

## Commands

- `init <repo-path>` — Bootstrap knowledge base for a codebase
- `scan` — Quick repo scan, identify key files, populate topic queue
- `explain file <path>` — Explain a file, create entry
- `explain function <file:symbol>` — Explain a function/class, create entry
- `explain repo [path]` — Repo architecture overview entry
- `explain diff [--branch B]` — Explain changes, create entry
- `explore [--skip] [--pick N[,N,...]] [--loop N]` — Work through topic queue (--loop N explores up to N topics continuously)
- `walk-commits --since DATE|--since-commit SHA|--since-last [--dry-run]` — Walk commits and explore each changed file
- `topics [--all]` — Show exploration queue
- `propose-beliefs [--auto] [--since DATE]` — Extract beliefs from entries (`--auto` accepts all without review; `--since` filters by entry date)
- `review-proposals [--batch-size N]` — LLM quality filter: rejects meta, duplicate, ephemeral, speculative, trivial proposals
- `accept-beliefs` — Import accepted beliefs (uses `reasons` if installed, falls back to `beliefs`)
- `derive [--auto] [--exhaust] [--dry-run]` — Propose deeper reasoning chains (`--exhaust` loops until no new derivations; delegates to `reasons derive`)
- `generate-summary` — Morning summary entry: new gated OUT beliefs, new negative IN beliefs, critical watch list
- `update --since-last|--since DATE [--file-issues]` — Full automated pipeline: walk-commits → propose-beliefs → review-proposals → accept-beliefs → derive --exhaust → generate-summary
- `file-issues [--dry-run] [--repo OWNER/REPO] [--label L]` — File issues from gated beliefs with active blockers (GitHub/GitLab)
- `status` — Dashboard (shows reasons.db stats if available)

## Natural Language

If the user says:
- "study this codebase" → `code-expert init <path> && code-expert scan`
- "what should I look at next" → `code-expert explore`
- "explain this file" → `code-expert explain file <path>`
- "extract what we've learned" → `code-expert propose-beliefs`
- "review the proposals" / "filter proposals" → `code-expert review-proposals`
- "build deeper chains" / "derive conclusions" → `code-expert derive`
- "derive everything" / "exhaust derivations" → `code-expert derive --exhaust`
- "file issues for blockers" / "what's blocking features" → `code-expert file-issues --dry-run`
- "walk through recent commits" / "explore what changed this week" → `code-expert walk-commits --since "1 week ago"`
- "catch up on changes" / "nightly update" / "update the knowledge base" → `code-expert update --since-last`
- "morning summary" / "what's the status" → `code-expert generate-summary`
- "how far along are we" → `code-expert status`

## Integrations

- **sentrux** — If `sentrux` is on PATH, it's available as an observation tool during `explore`. The LLM can request structural quality analysis (modularity, cycles, complexity, god files) when exploring architecture topics.

## Belief Storage

When `ftl-reasons` is installed (`reasons` CLI on PATH), `accept-beliefs` writes directly to `reasons.db` and re-exports `beliefs.md` and `network.json`. When only `ftl-beliefs` is installed, it writes to `beliefs.md` directly. The `init` command sets up whichever store is available.
