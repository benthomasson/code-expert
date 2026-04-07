---
name: code-expert
description: Build expert knowledge bases from codebases — explore, explain, extract beliefs
argument-hint: "[init|scan|explain|explore|walk-commits|topics|propose-beliefs|accept-beliefs|derive|file-issues|status]"
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
# edit proposed-beliefs.md: mark [ACCEPT] or [REJECT]
code-expert accept-beliefs                 # import accepted beliefs
code-expert status                         # dashboard
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
- `propose-beliefs` — Extract beliefs from entries
- `accept-beliefs` — Import accepted beliefs (uses `reasons` if installed, falls back to `beliefs`)
- `derive [--auto] [--dry-run]` — Propose deeper reasoning chains from existing beliefs (requires `reasons`)
- `file-issues [--dry-run] [--repo OWNER/REPO] [--label L]` — File issues from gated beliefs with active blockers (GitHub/GitLab)
- `status` — Dashboard (shows reasons.db stats if available)

## Natural Language

If the user says:
- "study this codebase" → `code-expert init <path> && code-expert scan`
- "what should I look at next" → `code-expert explore`
- "explain this file" → `code-expert explain file <path>`
- "extract what we've learned" → `code-expert propose-beliefs`
- "build deeper chains" / "derive conclusions" → `code-expert derive`
- "file issues for blockers" / "what's blocking features" → `code-expert file-issues --dry-run`
- "walk through recent commits" / "explore what changed this week" → `code-expert walk-commits --since "1 week ago"`
- "how far along are we" → `code-expert status`

## Belief Storage

When `ftl-reasons` is installed (`reasons` CLI on PATH), `accept-beliefs` writes directly to `reasons.db` and re-exports `beliefs.md` and `network.json`. When only `ftl-beliefs` is installed, it writes to `beliefs.md` directly. The `init` command sets up whichever store is available.
