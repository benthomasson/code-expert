# code-expert

Deep code analysis through belief networks. Systematically explores codebases, extracts factual beliefs, builds dependency networks, and surfaces architectural issues that code review, static analysis, and LLM chat miss individually.

**What it finds:** Not just bugs in diffs, but design-level issues — dormant subsystems, security defaults that were never hardened, abstractions that leak across layers. These have no diff that introduced them. They become visible only when you build a theory of the codebase and reason over it.

**How it works:** In one session on a 15k-line infrastructure framework, code-expert explored 152 topics, extracted 785 beliefs into a reason maintenance system, derived logical consequences across the belief network, and surfaced architectural issues that led to 8 merged PRs and 8 closed GitHub issues — including disabled SSH host key verification, command injection vectors, and a dormant policy engine with no tests.

## Install

```bash
uv tool install git+https://github.com/benthomasson/ftl-code-expert
```

Prerequisites — these CLIs must be on your PATH:

- `git`
- [`beliefs`](https://github.com/benthomasson/beliefs) — belief registry management
- [`entry`](https://github.com/benthomasson/entry) — chronological entry creation
- `claude` or `gemini` — at least one LLM CLI

Optional (for semantic belief deduplication):

```bash
uv pip install 'ftl-code-expert[embeddings]'
```

## Quick Start

```bash
# 1. Point code-expert at a repo
code-expert init ~/git/my-project --domain "Orchestration framework"

# 2. Scan for interesting files and topics
code-expert scan

# 3. Explore topics one at a time
code-expert explore          # next topic
code-expert explore --pick 3 # specific topic
code-expert explore --skip   # skip and move on

# 4. Extract beliefs from your exploration entries
code-expert propose-beliefs
# Review proposed-beliefs.md — mark entries ACCEPT or REJECT
code-expert accept-beliefs

# 5. Check progress
code-expert status

# Or run the full automated pipeline in one command
code-expert update --since-last
```

## How It Works

Code-expert follows a **scan → explore → distill → reason → act** pipeline:

```
scan              Lightweight repo analysis → topic queue
  │
  ▼
explore           Pop one topic, explain it deeply, discover new topics
  │                 ├── file     Read + explain a source file
  │                 ├── function Extract + explain a specific symbol
  │                 ├── repo     Explain a directory or full repo
  │                 ├── diff     Explain a branch or commit range
  │                 └── general  Observe (grep, read, find) then explain
  │
walk-commits      Walk commits since a date/SHA, explore each changed file
  │
  ▼
propose-beliefs   Batch-extract factual claims from entries
  │
  ▼
accept-beliefs    Import reviewed claims into beliefs.md / reasons.db
  │
  ▼
derive            Compute logical consequences across the belief network
  │                 ├── DERIVE   Conclusion holds when all premises hold
  │                 ├── GATE     Positive claim holds UNLESS a blocker is IN
  │                 └── --exhaust Loop until no new derivations (fixed-point)
  │
  ▼
generate-summary  Morning report: new gated OUT, new negative IN, critical watch list
  │
  ▼
file-issues       OUT gated beliefs → GitHub/GitLab issues automatically

update            Runs the full pipeline above in one command (--since-last)
```

Each exploration creates a dated entry in `entries/` and may generate new topics, so the queue grows organically as you learn.

### Why this catches things code review doesn't

The pipeline's power comes from five ingredients working together:

1. **Exhaustive exploration** — the topic queue ensures every public function, module boundary, and configuration surface gets examined. Humans skip things.
2. **Belief extraction** — observations become structured, queryable claims with provenance. "known_hosts=None" stops being a detail buried in a file and becomes a tracked fact.
3. **Dependency tracking** — beliefs connect to each other. "SSH layer is production-hardened" depends on "host key verification is enabled" and "command injection is prevented."
4. **Derivation** — the system computes logical consequences. If a premise is OUT, every conclusion that depends on it goes OUT automatically.
5. **Contradiction detection** — OUT derived beliefs map directly to actionable issues. "SSH layer is NOT production-hardened" becomes a GitHub issue with the specific blockers listed.

Remove any one of these and the process breaks down. Without exhaustive exploration, you miss premises. Without belief extraction, observations stay in prose. Without dependency tracking, you can't compute consequences. Without derivation, you see only individual facts. Without contradiction detection, you have a knowledge base but no actionable output.

## Commands

### `code-expert init <repo-path>`

Bootstrap a knowledge base. Creates `.code-expert/` config, `entries/`, `beliefs.md`, and a `CLAUDE.md` skill file.

```bash
code-expert init ~/git/agents-python --domain "AI orchestration framework"
```

### `code-expert scan`

Analyze repo structure, config files, README, and entry points. Produces an overview and populates the topic queue with 8-15 starting points.

```bash
code-expert scan
code-expert topics  # see what was queued
```

### `code-expert explain <kind> <target>`

Explain a specific piece of code without going through the topic queue.

```bash
code-expert explain file src/router.py
code-expert explain function src/router.py:route_request
code-expert explain repo .
code-expert explain repo src/workflows

# Explain what changed on a branch
code-expert explain diff --branch feature-auth

# Explain recent changes by date
code-expert explain diff --since 2026-03-01
code-expert explain diff --since "1 week ago"

# Pick up where you left off
code-expert explain diff --since-last
```

Large diffs (>100K chars) automatically switch to summary mode — commit log and file list only — then queue individual files for `explore`.

### `code-expert explore`

Process the next topic in the queue. Each exploration reads full source, invokes the model, creates an entry, and discovers follow-up topics.

```bash
code-expert explore              # next pending topic
code-expert explore --pick 2     # pick topic #2
code-expert explore --pick 1,3,8 # pick multiple (indices resolved before any are consumed)
code-expert explore --skip       # skip current topic
code-expert explore --loop 10    # continuously explore up to 10 topics
code-expert explore --loop 5     # explore up to 5
```

For `general` topics, explore uses a three-phase process: ask the model what it needs to observe, run those observations (grep, read_file, list_directory, find_symbol, find_usages, file_imports), then explain with the gathered context.

### `code-expert walk-commits`

Walk commits since a date or commit and explore each changed file individually. Creates one entry per file, with commit context. Deduplicates files across commits — each file is explored once using its latest version.

```bash
code-expert walk-commits --since 2026-03-01
code-expert walk-commits --since-commit abc1234
code-expert walk-commits --since-last        # from last diff checkpoint
code-expert walk-commits --since "1 week ago" --dry-run  # preview
```

### `code-expert topics`

View the exploration queue.

```bash
code-expert topics       # pending only
code-expert topics --all # include done and skipped
```

### `code-expert propose-beliefs`

Extract candidate beliefs from exploration entries. Deduplicates against existing beliefs (keyword overlap, or semantic similarity with `fastembed`).

```bash
code-expert propose-beliefs
code-expert propose-beliefs --batch-size 10
code-expert propose-beliefs --entry entries/2026/03/09/diff-since-last.md
code-expert propose-beliefs --all  # re-process everything
code-expert propose-beliefs --auto # extract and accept all without review
```

Output goes to `proposed-beliefs.md`. Each proposal is marked `[ACCEPT]` or `[REJECT]` — review and flip as needed, then import. Use `--auto` to skip review and accept all proposals directly.

### `code-expert accept-beliefs`

Import accepted proposals into `beliefs.md`.

```bash
code-expert accept-beliefs
code-expert accept-beliefs --file my-proposals.md
```

### `code-expert derive`

Analyze the reasons network and propose deeper reasoning chains by combining existing beliefs. Requires `reasons` (ftl-reasons) and `network.json`.

```bash
code-expert derive              # propose derivations → proposed-derivations.md
code-expert derive --auto       # propose and add to reasons automatically
code-expert derive --exhaust    # loop until no new derivations (implies --auto)
code-expert derive --dry-run    # show the prompt without invoking the LLM
code-expert derive --topic auth # only derive from auth-related beliefs
code-expert derive --budget 500 # include more beliefs in prompt
```

Delegates to `reasons derive`, which handles prompt building, LLM invocation, proposal validation (including Jaccard dedup against retracted beliefs), and network updates.

Proposes two types of derived beliefs:

- **DERIVE**: Standard SL justification — conclusion is IN when all antecedents are IN, cascades OUT when any is retracted
- **GATE**: Outlist-gated — positive claim holds UNLESS a negative claim (bug, gap, fragility) is IN. When the negative claim is retracted (problem fixed), the gated conclusion automatically restores

Without `--auto`, proposals are written to `proposed-derivations.md` for review. Use `--exhaust` to loop until the network reaches a fixed point (no new derivations possible).

### `code-expert file-issues`

File GitHub or GitLab issues from gated beliefs with active blockers. Detects the platform from the target repository's git remote.

```bash
code-expert file-issues              # auto-detect repo, file issues
code-expert file-issues --dry-run    # preview without filing
code-expert file-issues --repo owner/repo --label bug
code-expert file-issues --platform gitlab --repo group/project
```

For each GATE belief where the outlist node is IN (blocking the positive conclusion):
- Checks for existing issues to avoid duplicates (fuzzy title matching)
- Creates an issue with the blocker's description, impact (which beliefs it blocks), and resolution instructions
- Adds the `reasons-gate` label automatically

Requires `gh` (GitHub) or `glab` (GitLab) CLI to be installed and authenticated.

### `code-expert update`

Automated pipeline that runs the full workflow in one command: walk commits, extract and accept beliefs, derive all consequences, and generate a morning summary.

```bash
code-expert update --since-last              # from last checkpoint
code-expert update --since "1 week ago"      # from a date
code-expert update --since-last --file-issues # also file GitHub issues
```

The pipeline:
1. `walk-commits --since-last` — explore changed files
2. `propose-beliefs --auto` — extract and accept beliefs without review
3. `derive --exhaust` — compute all logical consequences until fixed point
4. `generate-summary` — morning report entry

Each step continues on non-fatal errors. The summary entry highlights what's new and what's critical.

### `code-expert generate-summary`

Generate a morning summary entry of belief state. The summary includes:

- **New gated OUT beliefs** — conclusions blocked by active negative findings
- **New negative IN beliefs** — bugs, gaps, and security issues just discovered
- **Critical watch list** — high-severity issues always shown regardless of age (security, injection, authentication, data loss, etc.)
- **Statistics** — belief counts, derivation counts, new vs existing

```bash
code-expert generate-summary  # standalone summary
```

When called via `update`, the summary automatically detects which beliefs are new (added during the current run) vs pre-existing.

### `code-expert status`

Dashboard showing entry count, belief counts (IN/STALE), nogoods, topic queue state, diff checkpoint, and proposal progress.

### `code-expert install-skill`

Install the Claude Code skill file so Claude can invoke code-expert commands.

```bash
code-expert install-skill
code-expert install-skill --skill-dir .claude/skills/code-expert
```

## Global Options

| Option | Description |
|--------|-------------|
| `--repo`, `-r` | Repository root (default: from config or cwd) |
| `--model`, `-m` | Model to use: `claude` or `gemini` (default: claude) |
| `--quiet`, `-q` | Suppress explanation output to stdout |
| `--version` | Show version |

The `-r` flag lets you work across repos:

```bash
code-expert scan -r ~/git/other-project
code-expert explore -r ~/git/other-project
```

## Project Layout

After `init`, the repo gets:

```
.code-expert/
├── config.json           # repo path, domain, created date
├── topics.json           # exploration queue
├── last-diff.json        # diff checkpoint for --since-last
├── proposed-entries.json  # tracks which entries have been processed
└── belief-vectors.json   # cached embeddings (if fastembed enabled)

entries/                   # dated exploration entries
├── 2026/03/08/
│   ├── scan-my-project.md
│   ├── file-router.md
│   └── diff-since-last.md

beliefs.md                # belief registry
nogoods.md                # tracked contradictions
proposed-beliefs.md       # proposals awaiting review
CLAUDE.md                 # AI assistant instructions
```

## Supported Models

| Name | CLI Command | Notes |
|------|-------------|-------|
| `claude` | `claude -p` | Default. Requires [Claude Code](https://claude.com/claude-code) |
| `gemini` | `gemini -p ""` | Requires Gemini CLI |

## Tips

- **Start broad, go deep.** `scan` gives you the lay of the land. `explore` digs into specifics. Each explanation surfaces new topics.
- **Use `--since` for ongoing tracking.** After the first `explain diff --since DATE`, subsequent runs with `--since-last` pick up automatically.
- **Review proposals carefully.** The model proposes beliefs — you decide which are worth keeping. Flip `[REJECT]` to `[ACCEPT]` for claims you verify.
- **Run `update --since-last` nightly.** This is the easiest way to keep the knowledge base current — one command walks new commits, extracts beliefs, derives consequences, and writes a summary you can read in the morning.
- **Run `derive --exhaust` after accepting beliefs.** This is where architectural insights emerge — the system connects individual observations into higher-level conclusions and identifies which ones are broken. `--exhaust` loops until no new derivations are possible.
- **Use `file-issues` to close the loop.** OUT gated beliefs become GitHub issues with specific blockers and resolution instructions. Fix the code, retract the blocker in the RMS, and the gated belief automatically restores to IN.
- **Run `status` periodically.** It shows stale beliefs, unexplored commits, and pending proposals.
- **Cross-repo exploration.** Use `-r` to point at any repo without re-initializing.
