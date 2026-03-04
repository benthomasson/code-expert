# code-expert

Build expert knowledge bases from codebases. Combines repo-aware code exploration with a belief extraction pipeline.

## Install

```bash
uv tool install git+https://github.com/benthomasson/code-expert
```

## Quick Start

```bash
code-expert init ~/git/my-project --domain "My Project"
code-expert scan
code-expert explore    # repeat until satisfied
code-expert propose-beliefs
# edit proposed-beliefs.md
code-expert accept-beliefs
code-expert status
```
