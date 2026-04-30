"""Command-line interface for code expert."""

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import click

from .git_utils import (
    commits_since_checkpoint,
    extract_symbol,
    find_related_tests,
    get_commit_log,
    get_diff,
    get_diff_since,
    get_diff_since_commit,
    get_file_content,
    get_imports,
    get_repo_structure,
    list_commits_with_files,
    load_diff_checkpoint,
    save_diff_checkpoint,
)
from .llm import check_model_available, invoke, invoke_sync
from .observations import parse_observation_requests, run_observations
from .prompts import (
    PROPOSE_BELIEFS_CODE,
    build_diff_prompt,
    build_diff_summary_prompt,
    build_file_prompt,
    build_function_prompt,
    build_observe_prompt,
    build_repo_prompt,
    build_scan_prompt,
)
from .topics import (
    add_topics,
    load_queue,
    parse_topics_from_response,
    pending_count,
    pop_at,
    pop_multiple,
    pop_next,
    skip_topic,
)

PROJECT_DIR = ".code-expert"


# --- Config helpers ---


def _load_config() -> dict | None:
    """Load .code-expert/config.json if it exists."""
    config_path = Path.cwd() / PROJECT_DIR / "config.json"
    if config_path.is_file():
        return json.loads(config_path.read_text())
    return None


def _save_config(config: dict) -> None:
    """Save config to .code-expert/config.json."""
    config_dir = Path.cwd() / PROJECT_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps(config, indent=2))


def _get_repo(ctx) -> str:
    """Resolve repo path from context."""
    return ctx.obj.get("repo", os.getcwd())


def _get_project_dir(ctx) -> str:
    """Resolve .code-expert directory relative to repo root."""
    return os.path.join(_get_repo(ctx), PROJECT_DIR)


# --- Output helpers ---


def _sanitize_path_for_filename(path: str) -> str:
    """Convert a file path to a safe filename."""
    name = path.replace("/", "-").replace("\\", "-")
    if "." in name:
        name = name.rsplit(".", 1)[0]
    # Remove leading dashes
    name = name.lstrip("-")
    return name[:80] if name else "unknown"


def _emit(ctx, text: str) -> None:
    """Print to stdout unless --quiet."""
    if not ctx.obj.get("quiet"):
        click.echo(text)


def _create_entry(topic: str, title: str, content: str) -> None:
    """Create an entry via the entry CLI."""
    try:
        result = subprocess.run(
            ["entry", "create", topic, title, "--content", content],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            click.echo(f"Entry: {result.stdout.strip()}", err=True)
        else:
            # Try without --content flag (pipe via stdin)
            result = subprocess.run(
                ["entry", "create", topic, title],
                input=content,
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                click.echo(f"Entry: {result.stdout.strip()}", err=True)
            else:
                click.echo(f"WARN: entry create failed: {result.stderr.strip()}", err=True)
    except FileNotFoundError:
        click.echo("WARN: entry CLI not found. Install with: uv tool install entry", err=True)


def _enqueue_topics(response: str, source: str, project_dir: str | None = None) -> None:
    """Parse topics from model response and add to queue."""
    new_topics = parse_topics_from_response(response, source=source)
    if new_topics:
        added = add_topics(new_topics, project_dir)
        if added:
            total = pending_count(project_dir)
            click.echo(f"Queued {added} new topic(s) ({total} pending)", err=True)


def _has_reasons() -> bool:
    """Check if ftl-reasons CLI is available."""
    return shutil.which("reasons") is not None


def _parse_beliefs_from_response(response: str) -> list[dict]:
    """Parse belief suggestions from model response."""
    section_match = re.search(
        r"#+\s*Beliefs?\s*\n(.*?)(?=\n#|\Z)",
        response, re.DOTALL | re.IGNORECASE,
    )
    if not section_match:
        return []

    beliefs = []
    pattern = re.compile(r"^[-*]\s+`([^`]+)`\s*(?:—|-|:)\s*(.+)$", re.MULTILINE)
    for match in pattern.finditer(section_match.group(1)):
        beliefs.append({
            "id": match.group(1),
            "text": match.group(2).strip(),
        })
    return beliefs


def _report_beliefs(response: str) -> None:
    """Report extracted beliefs to stderr for user awareness."""
    beliefs = _parse_beliefs_from_response(response)
    if beliefs:
        click.echo(f"Surfaced {len(beliefs)} belief(s):", err=True)
        for b in beliefs[:5]:
            click.echo(f"  {b['id']}: {b['text'][:80]}", err=True)


def _find_project_config(repo_path: str) -> tuple[str | None, str | None]:
    """Find and read the project config file."""
    config_files = [
        "pyproject.toml", "package.json", "Cargo.toml",
        "go.mod", "pom.xml", "build.gradle", "Makefile",
    ]
    for config in config_files:
        path = os.path.join(repo_path, config)
        content = get_file_content(path)
        if content is not None:
            return config, content
    return None, None


def _find_entry_points(repo_path: str, config_content: str | None) -> list[str]:
    """Identify likely entry points from config and convention."""
    entry_points = []
    candidates = [
        "src/main.py", "main.py", "app.py", "src/app.py",
        "manage.py", "setup.py", "cli.py",
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(repo_path, candidate)):
            entry_points.append(candidate)

    if config_content and "[project.scripts]" in config_content:
        in_scripts = False
        for line in config_content.split("\n"):
            if "[project.scripts]" in line:
                in_scripts = True
                continue
            if in_scripts:
                if line.startswith("["):
                    break
                if "=" in line:
                    entry_points.append(line.strip())

    return entry_points


# --- CLI ---


@click.group()
@click.version_option(package_name="ftl-code-expert")
@click.option("--quiet", "-q", is_flag=True, default=False,
              help="Suppress explanation output to stdout")
@click.option("--repo", "-r", type=click.Path(exists=True, file_okay=False),
              default=None, help="Repository root (default: from config or cwd)")
@click.option("--model", "-m", default="claude", help="Model to use (default: claude)")
@click.option("--timeout", "-t", default=300, type=int, help="LLM timeout in seconds (default: 300)")
@click.pass_context
def cli(ctx, quiet, repo, model, timeout):
    """Build expert knowledge bases from codebases."""
    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet
    ctx.obj["model"] = model
    ctx.obj["timeout"] = timeout
    if repo:
        ctx.obj["repo"] = os.path.abspath(repo)
    else:
        config = _load_config()
        ctx.obj["repo"] = config.get("repo_path", os.getcwd()) if config else os.getcwd()


# --- init ---


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
@click.option("--domain", "-d", default=None, help="One-line domain description")
def init(repo_path, domain):
    """Bootstrap a code-expert knowledge base for a codebase."""
    abs_repo = os.path.abspath(repo_path)
    repo_name = os.path.basename(abs_repo)

    if not domain:
        domain = repo_name

    # Check prerequisites — reasons OR beliefs required, not both
    for tool in ["git", "entry"]:
        if not shutil.which(tool):
            click.echo(f"Error: {tool} not found on PATH", err=True)
            click.echo(f"Install with: uv tool install {tool}", err=True)
            sys.exit(1)
    if not shutil.which("reasons") and not shutil.which("beliefs"):
        click.echo("Error: neither reasons nor beliefs found on PATH", err=True)
        click.echo("Install with: uv tool install ftl-reasons", err=True)
        sys.exit(1)

    # Create project dir
    project_dir = Path.cwd() / PROJECT_DIR
    project_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    _save_config({
        "repo_path": abs_repo,
        "domain": domain,
        "created": date.today().isoformat(),
    })

    # Create entries dir
    Path("entries").mkdir(exist_ok=True)

    # Init reasons as primary store if available, otherwise beliefs
    if _has_reasons():
        if not Path("reasons.db").exists():
            subprocess.run(["reasons", "init"], capture_output=True)
            subprocess.run(
                ["reasons", "add-repo", repo_name, abs_repo],
                capture_output=True,
            )
            click.echo("Initialized reasons.db")
        # Generate beliefs.md from reasons
        if not Path("beliefs.md").exists():
            _reasons_export()
    elif not Path("beliefs.md").exists():
        subprocess.run(["beliefs", "init"], capture_output=True)
        click.echo("Initialized beliefs.md")

    # Generate CLAUDE.md
    template_path = Path(__file__).parent / "data" / "CLAUDE.md.template"
    if template_path.exists():
        template = template_path.read_text()
        claude_md = template.replace("{{DOMAIN}}", domain).replace("{{REPO_PATH}}", abs_repo)
        Path("CLAUDE.md").write_text(claude_md)
        click.echo("Generated CLAUDE.md")

    click.echo(f"\nInitialized code-expert for {repo_name}")
    click.echo(f"  Repo: {abs_repo}")
    click.echo(f"  Domain: {domain}")
    click.echo(f"\nNext: code-expert scan")


# --- scan ---


@cli.command()
@click.pass_context
def scan(ctx):
    """Scan a repo to identify key files and populate the exploration queue."""
    from .caffeinate import hold as _caffeinate
    _caffeinate()
    repo_path = _get_repo(ctx)
    model = ctx.obj["model"]
    timeout = ctx.obj["timeout"]

    if not check_model_available(model):
        click.echo(f"Error: Model '{model}' CLI not available", err=True)
        sys.exit(1)

    click.echo(f"Scanning {repo_path}...", err=True)

    tree = get_repo_structure(repo_path, max_depth=3)
    _, config_content = _find_project_config(repo_path)
    readme_content = get_file_content(os.path.join(repo_path, "README.md"))
    entry_points = _find_entry_points(repo_path, config_content)

    prompt = build_scan_prompt(
        tree=tree,
        config_content=config_content,
        readme_content=readme_content,
        entry_points=entry_points or None,
    )

    click.echo(f"Running {model}...", err=True)
    try:
        result = asyncio.run(invoke(prompt, model, timeout=timeout))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Create entry
    repo_name = os.path.basename(repo_path)
    _create_entry(f"scan-{repo_name}", f"Scan: {repo_name}", result)

    # Enqueue topics
    _enqueue_topics(result, source=f"scan:{repo_name}", project_dir=_get_project_dir(ctx))

    _emit(ctx, result)


# --- explain group ---


@cli.group()
@click.pass_context
def explain(ctx):
    """Explain files, functions, repos, or diffs."""
    pass


@explain.command("file")
@click.argument("file_path", type=click.Path())
@click.pass_context
def explain_file(ctx, file_path):
    """Explain a file's purpose, structure, and key patterns."""
    model = ctx.obj["model"]
    timeout = ctx.obj["timeout"]
    repo_path = _get_repo(ctx)

    if not check_model_available(model):
        click.echo(f"Error: Model '{model}' CLI not available", err=True)
        sys.exit(1)

    # Resolve path: try as-is first, then relative to repo root
    abs_path = os.path.abspath(file_path)
    if not os.path.isfile(abs_path):
        repo_resolved = os.path.join(os.path.abspath(repo_path), file_path)
        if os.path.isfile(repo_resolved):
            abs_path = repo_resolved
        else:
            click.echo(f"Error: File not found: {file_path}", err=True)
            click.echo(f"  Tried: {abs_path}", err=True)
            click.echo(f"  Tried: {repo_resolved}", err=True)
            sys.exit(1)
    content = get_file_content(abs_path)
    if content is None:
        click.echo(f"Error: Cannot read file: {file_path}", err=True)
        sys.exit(1)

    click.echo(f"Explaining {file_path}...", err=True)

    rel_path = os.path.relpath(abs_path, os.path.abspath(repo_path))
    import_info = get_imports(abs_path, os.path.abspath(repo_path))
    repo_tree = get_repo_structure(os.path.abspath(repo_path), max_depth=2)

    prompt = build_file_prompt(
        file_path=rel_path,
        file_content=content,
        imports=import_info["imports"] or None,
        imported_by=import_info["imported_by"] or None,
        repo_context=repo_tree,
    )

    click.echo(f"Running {model}...", err=True)
    try:
        result = asyncio.run(invoke(prompt, model, timeout=timeout))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    topic_name = _sanitize_path_for_filename(rel_path)
    _create_entry(topic_name, f"File: {rel_path}", result)
    _enqueue_topics(result, source=f"file:{rel_path}", project_dir=_get_project_dir(ctx))
    _report_beliefs(result)

    _emit(ctx, result)


@explain.command("function")
@click.argument("target")
@click.pass_context
def explain_function(ctx, target):
    """Explain a specific function or class. TARGET: file_path:symbol_name"""
    model = ctx.obj["model"]
    timeout = ctx.obj["timeout"]
    repo_path = _get_repo(ctx)

    if ":" not in target:
        click.echo("Error: TARGET must be FILE_PATH:SYMBOL_NAME", err=True)
        sys.exit(1)

    file_path, symbol_name = target.rsplit(":", 1)

    if not os.path.isfile(file_path):
        click.echo(f"Error: File not found: {file_path}", err=True)
        sys.exit(1)

    if not check_model_available(model):
        click.echo(f"Error: Model '{model}' CLI not available", err=True)
        sys.exit(1)

    abs_path = os.path.abspath(file_path)
    abs_repo = os.path.abspath(repo_path)

    symbol_source = extract_symbol(abs_path, symbol_name)
    if symbol_source is None:
        click.echo(f"Error: Symbol '{symbol_name}' not found in {file_path}", err=True)
        sys.exit(1)

    click.echo(f"Explaining {symbol_name} from {file_path}...", err=True)

    full_content = get_file_content(abs_path)
    related_tests = find_related_tests(abs_path, abs_repo, symbol_name)
    rel_path = os.path.relpath(abs_path, abs_repo)

    prompt = build_function_prompt(
        file_path=rel_path,
        symbol_name=symbol_name,
        symbol_source=symbol_source,
        full_file_content=full_content,
        related_tests=related_tests or None,
    )

    click.echo(f"Running {model}...", err=True)
    try:
        result = asyncio.run(invoke(prompt, model, timeout=timeout))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    topic_name = _sanitize_path_for_filename(rel_path) + f"-{symbol_name}"
    _create_entry(topic_name, f"Function: {symbol_name} in {rel_path}", result)
    _enqueue_topics(result, source=f"function:{rel_path}:{symbol_name}", project_dir=_get_project_dir(ctx))
    _report_beliefs(result)

    _emit(ctx, result)


@explain.command("repo")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False),
                default=".", required=False)
@click.pass_context
def explain_repo(ctx, repo_path):
    """Generate a high-level repository architecture overview."""
    model = ctx.obj["model"]
    timeout = ctx.obj["timeout"]

    if repo_path == ".":
        repo_path = _get_repo(ctx)

    if not check_model_available(model):
        click.echo(f"Error: Model '{model}' CLI not available", err=True)
        sys.exit(1)

    abs_repo = os.path.abspath(repo_path)
    repo_name = os.path.basename(abs_repo)
    click.echo(f"Analyzing repository at {abs_repo}...", err=True)

    tree = get_repo_structure(abs_repo)
    _, config_content = _find_project_config(abs_repo)
    readme_content = get_file_content(os.path.join(abs_repo, "README.md"))
    if readme_content is None:
        for alt in ["README.rst", "README.txt", "README"]:
            readme_content = get_file_content(os.path.join(abs_repo, alt))
            if readme_content is not None:
                break
    entry_points = _find_entry_points(abs_repo, config_content)

    prompt = build_repo_prompt(
        tree=tree,
        config_content=config_content,
        readme_content=readme_content,
        entry_points=entry_points or None,
    )

    click.echo(f"Running {model}...", err=True)
    try:
        result = asyncio.run(invoke(prompt, model, timeout=timeout))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    _create_entry(f"repo-{repo_name}", f"Repo Overview: {repo_name}", result)
    _enqueue_topics(result, source="repo-overview", project_dir=_get_project_dir(ctx))
    _report_beliefs(result)

    _emit(ctx, result)


@explain.command("diff")
@click.option("--branch", "-b", default=None, help="Branch to explain")
@click.option("--base", default="main", help="Base branch (default: main)")
@click.option("--since", default=None, help="Show changes since date (e.g., 2026-03-01, '1 week ago')")
@click.option("--since-last", is_flag=True, default=False,
              help="Show changes since last explain diff run")
@click.pass_context
def explain_diff(ctx, branch, base, since, since_last):
    """Explain what changed in a diff and why."""
    model = ctx.obj["model"]
    timeout = ctx.obj["timeout"]
    repo_path = _get_repo(ctx)
    project_dir = _get_project_dir(ctx)

    if not check_model_available(model):
        click.echo(f"Error: Model '{model}' CLI not available", err=True)
        sys.exit(1)

    abs_repo = os.path.abspath(repo_path)

    try:
        if since_last:
            checkpoint = load_diff_checkpoint(project_dir)
            if not checkpoint:
                click.echo("No previous diff checkpoint found. Use --since DATE first.", err=True)
                sys.exit(1)
            click.echo(f"Picking up from {checkpoint['timestamp']} ({checkpoint['head'][:8]})", err=True)
            diff_content, commit_log = get_diff_since_commit(
                checkpoint["head"], cwd=abs_repo,
            )
        elif since:
            diff_content, commit_log = get_diff_since(since, cwd=abs_repo)
        elif branch:
            diff_content = get_diff(branch, base, cwd=abs_repo)
            commit_log = get_commit_log(branch, base, cwd=abs_repo)
        else:
            diff_content = get_diff(cwd=abs_repo)
            commit_log = None
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if not diff_content.strip():
        click.echo("No changes since last run.", err=True)
        sys.exit(0)

    changed_files = []
    for line in diff_content.split("\n"):
        if line.startswith("+++ b/"):
            path = line[6:]
            if path != "/dev/null":
                changed_files.append(path)

    if since_last:
        diff_label = f"since-last ({checkpoint['head'][:8]})"
    elif since:
        diff_label = f"since {since}"
    else:
        diff_label = branch or "staged"
    click.echo(f"Explaining {diff_label} changes ({len(changed_files)} files)...", err=True)

    max_diff_chars = 100_000
    if len(diff_content) > max_diff_chars:
        click.echo(
            f"Diff too large ({len(diff_content):,} chars). Using summary mode — "
            f"run 'explore' afterward to examine individual files.",
            err=True,
        )
        prompt = build_diff_summary_prompt(
            commit_log=commit_log,
            changed_files=changed_files or None,
        )
    else:
        prompt = build_diff_prompt(
            diff_content=diff_content,
            commit_log=commit_log,
            changed_files_summary=changed_files or None,
        )

    click.echo(f"Running {model}...", err=True)
    try:
        result = asyncio.run(invoke(prompt, model, timeout=timeout))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    safe_label = diff_label.replace("/", "-").replace(" ", "-")
    _create_entry(f"diff-{safe_label}", f"Diff: {diff_label}", result)
    _enqueue_topics(result, source=f"diff:{diff_label}", project_dir=project_dir)
    _report_beliefs(result)

    # Save checkpoint so --since-last works next time
    if since or since_last:
        save_diff_checkpoint(project_dir, cwd=abs_repo)
        click.echo("Diff checkpoint saved.", err=True)

    _emit(ctx, result)


# --- topics ---


@cli.command()
@click.option("--all", "show_all", is_flag=True, default=False,
              help="Show all topics including done and skipped")
@click.pass_context
def topics(ctx, show_all):
    """Show the exploration queue."""
    queue = load_queue(_get_project_dir(ctx))

    if not queue:
        click.echo("No topics queued. Run `code-expert scan` to discover topics.")
        return

    pending = [t for t in queue if t.status == "pending"]
    done = [t for t in queue if t.status == "done"]
    skipped = [t for t in queue if t.status == "skipped"]

    if pending:
        click.echo(f"Pending ({len(pending)}):\n")
        for i, topic in enumerate(pending):
            click.echo(f"  {i}. [{topic.kind}] {topic.target}")
            click.echo(f"     {topic.title}")
            if topic.source:
                click.echo(f"     (from {topic.source})")
            click.echo()
    else:
        click.echo("No pending topics.")

    if show_all:
        if done:
            click.echo(f"Done ({len(done)}):\n")
            for topic in done:
                click.echo(f"  [{topic.kind}] {topic.target} - {topic.title}")
        if skipped:
            click.echo(f"\nSkipped ({len(skipped)}):\n")
            for topic in skipped:
                click.echo(f"  [{topic.kind}] {topic.target} - {topic.title}")

    click.echo(f"\n{len(pending)} pending, {len(done)} done, {len(skipped)} skipped")


# --- explore ---


@cli.command()
@click.option("--skip", "do_skip", is_flag=True, default=False,
              help="Skip the next topic")
@click.option("--pick", "pick_index", type=str, default=None,
              help="Pick topic(s) by index — single (3) or comma-separated (1,3,8)")
@click.option("--loop", "loop_max", type=int, default=None,
              help="Continuously explore up to N topics [default: 10]")
@click.pass_context
def explore(ctx, do_skip, pick_index, loop_max):
    """Explore the next topic in the queue (or --skip / --pick N[,N,...])."""
    from .caffeinate import hold as _caffeinate
    _caffeinate()
    project_dir = _get_project_dir(ctx)

    if loop_max is not None:
        if do_skip or pick_index:
            click.echo("Error: --loop cannot be combined with --skip or --pick", err=True)
            sys.exit(1)
        _explore_loop(ctx, project_dir, loop_max)
        return

    if do_skip:
        if skip_topic(0, project_dir):
            queue = load_queue(project_dir)
            pending = [t for t in queue if t.status == "pending"]
            if pending:
                click.echo(f"Skipped. Next: [{pending[0].kind}] {pending[0].target}")
            else:
                click.echo("Skipped. No more pending topics.")
        else:
            click.echo("Nothing to skip.")
        return

    if pick_index is not None:
        # Parse comma-separated indices
        try:
            indices = [int(x.strip()) for x in pick_index.split(",")]
        except ValueError:
            click.echo(f"Error: --pick must be integers, got: {pick_index}", err=True)
            sys.exit(1)

        if len(indices) > 1:
            topics = pop_multiple(indices, project_dir)
        else:
            topics = [pop_at(indices[0], project_dir)]
    else:
        topics = [pop_next(project_dir)]

    # Filter out None (invalid indices)
    valid_topics = [(i, t) for i, t in zip(
        indices if pick_index is not None else [0],
        topics,
    ) if t is not None]

    if not valid_topics:
        click.echo("No pending topics. Run `code-expert scan` to discover topics.")
        return

    invalid_count = len(topics) - len(valid_topics)
    if invalid_count:
        click.echo(f"Warning: {invalid_count} index(es) out of bounds, skipped.", err=True)

    repo_path = _get_repo(ctx)
    abs_repo = os.path.abspath(repo_path)
    model = ctx.obj["model"]
    timeout = ctx.obj["timeout"]

    for seq, (idx, topic) in enumerate(valid_topics):
        if len(valid_topics) > 1:
            click.echo(f"\n{'=' * 40}", err=True)
            click.echo(f"[{seq + 1}/{len(valid_topics)}] Topic #{idx}", err=True)
            click.echo(f"{'=' * 40}", err=True)

        click.echo(f"Topic: [{topic.kind}] {topic.target}", err=True)
        click.echo(f"  {topic.title}", err=True)
        if topic.source:
            click.echo(f"  (from {topic.source})", err=True)
        click.echo(err=True)

        if topic.kind == "file":
            _run_file_topic(ctx, topic, model, abs_repo)
        elif topic.kind == "function":
            _run_function_topic(ctx, topic, model, abs_repo)
        elif topic.kind == "repo":
            _run_repo_topic(ctx, topic, model, abs_repo)
        elif topic.kind == "diff":
            _run_diff_topic(ctx, topic, model, abs_repo)
        elif topic.kind == "general":
            _run_general_topic(ctx, topic, model, abs_repo)
        else:
            click.echo(f"Unknown topic kind: {topic.kind}", err=True)

    remaining = pending_count(project_dir)
    if remaining:
        click.echo(f"\n{remaining} topic(s) remaining. Run `code-expert explore` to continue.", err=True)
    else:
        click.echo("\nNo more topics. Exploration complete.", err=True)


def _explore_loop(ctx, project_dir, max_topics):
    """Continuously explore topics up to max_topics."""
    repo_path = _get_repo(ctx)
    abs_repo = os.path.abspath(repo_path)
    model = ctx.obj["model"]
    timeout = ctx.obj["timeout"]

    explored = 0
    while explored < max_topics:
        topic = pop_next(project_dir)
        if topic is None:
            if explored == 0:
                click.echo("No pending topics. Run `code-expert scan` to discover topics.")
            else:
                click.echo(f"\nNo more topics after {explored} exploration(s).", err=True)
            return

        explored += 1
        remaining = pending_count(project_dir)
        click.echo(f"\n{'=' * 40}", err=True)
        click.echo(f"[{explored}/{max_topics}] ({remaining} remaining in queue)", err=True)
        click.echo(f"{'=' * 40}", err=True)
        click.echo(f"Topic: [{topic.kind}] {topic.target}", err=True)
        click.echo(f"  {topic.title}", err=True)
        if topic.source:
            click.echo(f"  (from {topic.source})", err=True)
        click.echo(err=True)

        if topic.kind == "file":
            _run_file_topic(ctx, topic, model, abs_repo)
        elif topic.kind == "function":
            _run_function_topic(ctx, topic, model, abs_repo)
        elif topic.kind == "repo":
            _run_repo_topic(ctx, topic, model, abs_repo)
        elif topic.kind == "diff":
            _run_diff_topic(ctx, topic, model, abs_repo)
        elif topic.kind == "general":
            _run_general_topic(ctx, topic, model, abs_repo)
        else:
            click.echo(f"Unknown topic kind: {topic.kind}", err=True)

    remaining = pending_count(project_dir)
    click.echo(f"\nExplored {explored} topic(s). {remaining} remaining in queue.", err=True)


def _run_file_topic(ctx, topic, model, repo_path):
    """Handle a file exploration topic."""
    timeout = ctx.obj["timeout"]
    file_path = topic.target
    abs_path = os.path.join(repo_path, file_path) if not os.path.isabs(file_path) else file_path

    if os.path.isdir(abs_path):
        click.echo(f"{file_path} is a directory — exploring as repo topic.", err=True)
        topic.kind = "repo"
        _run_repo_topic(ctx, topic, model, repo_path)
        return

    if not os.path.isfile(abs_path):
        click.echo(f"File not found: {file_path} (skipping)", err=True)
        return

    if not check_model_available(model):
        click.echo(f"Error: Model '{model}' CLI not available", err=True)
        sys.exit(1)

    click.echo(f"Reading {file_path}...", err=True)
    content = get_file_content(abs_path)
    if content is None:
        click.echo(f"Cannot read file: {file_path}", err=True)
        return

    rel_path = os.path.relpath(abs_path, repo_path)
    import_info = get_imports(abs_path, repo_path)
    repo_tree = get_repo_structure(repo_path, max_depth=2)

    prompt = build_file_prompt(
        file_path=rel_path,
        file_content=content,
        imports=import_info["imports"] or None,
        imported_by=import_info["imported_by"] or None,
        repo_context=repo_tree,
    )

    click.echo(f"Explaining {rel_path} with {model}...", err=True)
    try:
        result = asyncio.run(invoke(prompt, model, timeout=timeout))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    topic_name = _sanitize_path_for_filename(rel_path)
    _create_entry(topic_name, f"File: {rel_path}", result)
    _enqueue_topics(result, source=f"file:{rel_path}", project_dir=_get_project_dir(ctx))
    _report_beliefs(result)

    _emit(ctx, result)


def _run_function_topic(ctx, topic, model, repo_path):
    """Handle a function exploration topic."""
    timeout = ctx.obj["timeout"]
    if ":" not in topic.target:
        click.echo(f"Function topic must be file:symbol, got: {topic.target}", err=True)
        return

    file_path, symbol_name = topic.target.rsplit(":", 1)
    abs_path = os.path.join(repo_path, file_path) if not os.path.isabs(file_path) else file_path

    if not os.path.isfile(abs_path):
        click.echo(f"File not found: {file_path} (skipping)", err=True)
        return

    if not check_model_available(model):
        click.echo(f"Error: Model '{model}' CLI not available", err=True)
        sys.exit(1)

    click.echo(f"Reading {file_path}:{symbol_name}...", err=True)
    symbol_source = extract_symbol(abs_path, symbol_name)
    if symbol_source is None:
        click.echo(f"Symbol '{symbol_name}' not found in {file_path} (skipping)", err=True)
        return

    full_content = get_file_content(abs_path)
    related_tests = find_related_tests(abs_path, repo_path, symbol_name)
    rel_path = os.path.relpath(abs_path, repo_path)

    prompt = build_function_prompt(
        file_path=rel_path,
        symbol_name=symbol_name,
        symbol_source=symbol_source,
        full_file_content=full_content,
        related_tests=related_tests or None,
    )

    click.echo(f"Explaining {rel_path}:{symbol_name} with {model}...", err=True)
    try:
        result = asyncio.run(invoke(prompt, model, timeout=timeout))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    topic_name = _sanitize_path_for_filename(rel_path) + f"-{symbol_name}"
    _create_entry(topic_name, f"Function: {symbol_name} in {rel_path}", result)
    _enqueue_topics(result, source=f"function:{rel_path}:{symbol_name}", project_dir=_get_project_dir(ctx))
    _report_beliefs(result)

    _emit(ctx, result)


def _run_repo_topic(ctx, topic, model, repo_path):
    """Handle a repo exploration topic."""
    timeout = ctx.obj["timeout"]
    target_path = os.path.join(repo_path, topic.target) if topic.target != "." else repo_path
    if not os.path.isdir(target_path):
        target_path = repo_path

    if not check_model_available(model):
        click.echo(f"Error: Model '{model}' CLI not available", err=True)
        sys.exit(1)

    click.echo(f"Scanning repo structure...", err=True)
    tree = get_repo_structure(target_path)
    _, config_content = _find_project_config(target_path)
    readme_content = get_file_content(os.path.join(target_path, "README.md"))
    entry_points = _find_entry_points(target_path, config_content)

    prompt = build_repo_prompt(
        tree=tree,
        config_content=config_content,
        readme_content=readme_content,
        entry_points=entry_points or None,
    )

    click.echo(f"Explaining repo with {model}...", err=True)
    try:
        result = asyncio.run(invoke(prompt, model, timeout=timeout))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    _create_entry("repo-overview", "Repo Overview", result)
    _enqueue_topics(result, source="repo-overview", project_dir=_get_project_dir(ctx))
    _report_beliefs(result)

    _emit(ctx, result)


def _run_diff_topic(ctx, topic, model, repo_path):
    """Handle a diff exploration topic."""
    timeout = ctx.obj["timeout"]
    if not check_model_available(model):
        click.echo(f"Error: Model '{model}' CLI not available", err=True)
        sys.exit(1)

    try:
        diff_content = get_diff(topic.target, cwd=repo_path)
    except RuntimeError as e:
        click.echo(f"Error getting diff: {e}", err=True)
        return

    if not diff_content.strip():
        click.echo("No changes to explain.", err=True)
        return

    commit_log = get_commit_log(topic.target, cwd=repo_path)

    changed_files = []
    for line in diff_content.split("\n"):
        if line.startswith("+++ b/"):
            path = line[6:]
            if path != "/dev/null":
                changed_files.append(path)

    prompt = build_diff_prompt(
        diff_content=diff_content,
        commit_log=commit_log,
        changed_files_summary=changed_files or None,
    )

    click.echo(f"Explaining diff {topic.target} ({len(changed_files)} files) with {model}...", err=True)
    try:
        result = asyncio.run(invoke(prompt, model, timeout=timeout))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    safe_label = topic.target.replace("/", "-")
    _create_entry(f"diff-{safe_label}", f"Diff: {topic.target}", result)
    _enqueue_topics(result, source=f"diff:{topic.target}", project_dir=_get_project_dir(ctx))
    _report_beliefs(result)

    _emit(ctx, result)


def _run_general_topic(ctx, topic, model, repo_path):
    """Handle a general exploration topic using observe-then-explain."""
    timeout = ctx.obj["timeout"]
    if not check_model_available(model):
        click.echo(f"Error: Model '{model}' CLI not available", err=True)
        sys.exit(1)

    from .prompts.common import BELIEFS_INSTRUCTIONS, TOPICS_INSTRUCTIONS

    # Phase 1: Observe
    tree = get_repo_structure(repo_path, max_depth=2)
    observe_prompt = build_observe_prompt(question=topic.title, tree=tree)

    click.echo(f"Gathering observations with {model}...", err=True)
    try:
        observe_response = asyncio.run(invoke(observe_prompt, model))
    except Exception as e:
        click.echo(f"Error during observe: {e}", err=True)
        sys.exit(1)

    requested_obs = parse_observation_requests(observe_response)

    # Phase 2: Run observations
    obs_results = {}
    if requested_obs:
        click.echo(f"Running {len(requested_obs)} observation(s):", err=True)
        for obs in requested_obs:
            click.echo(f"  - {obs.get('tool')}: {obs.get('name')}", err=True)
        obs_results = asyncio.run(run_observations(requested_obs, repo_path))

        failed = [n for n, r in obs_results.items() if isinstance(r, dict) and "error" in r]
        if failed:
            click.echo(f"  ({len(failed)} failed)", err=True)
    else:
        click.echo("No observations requested.", err=True)

    # Phase 3: Explain with targeted context
    explain_sections = [
        "You are a senior software engineer explaining a codebase to a new team member.",
        f"The reader wants to understand: **{topic.title}**",
        "",
    ]

    if obs_results:
        explain_sections.extend([
            "## Observations",
            "",
            "The following information was gathered from the codebase:",
            "",
            "```json",
            json.dumps(obs_results, indent=2, default=str),
            "```",
            "",
        ])

    explain_sections.extend([
        "## Instructions",
        "",
        f"Explain **{topic.title}** based on the observations above.",
        "Reference specific files, functions, and line numbers from the observations.",
        "If the observations are insufficient, say what's missing.",
        "",
        "Format your response as markdown.",
        TOPICS_INSTRUCTIONS,
        BELIEFS_INSTRUCTIONS,
    ])

    prompt = "\n".join(explain_sections)

    click.echo(f"Explaining with {model}...", err=True)
    try:
        result = asyncio.run(invoke(prompt, model, timeout=timeout))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    safe_label = _sanitize_path_for_filename(topic.target)
    _create_entry(f"topic-{safe_label}", f"Topic: {topic.title}", result)
    _enqueue_topics(result, source=f"general:{topic.target}", project_dir=_get_project_dir(ctx))
    _report_beliefs(result)

    _emit(ctx, result)


def _repo_path_to_entry_pattern(repo_path: str) -> str:
    """Convert a repo file path to the entry-name pattern used in belief sources.

    Example: src/redhat_agents/capabilities/dataverse/mart_proxy.py
          -> src-redhat_agents-capabilities-dataverse-mart_proxy
    """
    # Strip .py extension
    if repo_path.endswith(".py"):
        repo_path = repo_path[:-3]
    # Replace path separators with dashes
    return repo_path.replace("/", "-").replace("\\", "-")


def _retract_beliefs_for_deleted_files(deleted_files: set[str]) -> None:
    """Find beliefs sourced from deleted files and retract them via reasons."""
    beliefs_path = Path("beliefs.md")
    if not beliefs_path.exists():
        return

    # Build entry-name patterns for deleted files
    patterns = {_repo_path_to_entry_pattern(f) for f in deleted_files}

    # Parse beliefs and find those sourced from deleted files
    beliefs = _parse_beliefs_md(beliefs_path)
    to_retract = []
    for belief in beliefs:
        source = belief.get("source", "")
        for pattern in patterns:
            if pattern in source:
                to_retract.append(belief["id"])
                break

    if not to_retract:
        click.echo(f"  No beliefs found sourced from deleted files", err=True)
        return

    deleted_names = ", ".join(sorted(deleted_files))
    click.echo(
        f"  Retracting {len(to_retract)} belief(s) sourced from deleted file(s): {deleted_names}",
        err=True,
    )

    if _has_reasons() and Path("reasons.db").exists():
        retracted = 0
        for belief_id in to_retract:
            result = subprocess.run(
                ["reasons", "retract", belief_id,
                 "--reason", f"Source file deleted: {deleted_names}"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                retracted += 1
                click.echo(f"    Retracted: {belief_id}", err=True)
            else:
                click.echo(f"    Failed to retract {belief_id}: {result.stderr.strip()}", err=True)
        click.echo(f"  Retracted {retracted}/{len(to_retract)} belief(s)", err=True)
    else:
        click.echo(
            "  WARNING: reasons.db not found — cannot retract beliefs automatically.",
            err=True,
        )
        click.echo("  Beliefs to retract manually:", err=True)
        for belief_id in to_retract:
            click.echo(f"    - {belief_id}", err=True)


# --- walk-commits ---


@cli.command("walk-commits")
@click.option("--since", default=None, help="Walk commits since date (e.g., 2026-03-01, '1 week ago')")
@click.option("--since-commit", default=None, help="Walk commits since a specific commit SHA")
@click.option("--since-last", is_flag=True, default=False,
              help="Walk commits since last diff checkpoint")
@click.option("--dry-run", is_flag=True, default=False,
              help="List commits and files without exploring")
@click.pass_context
def walk_commits(ctx, since, since_commit, since_last, dry_run):
    """Walk commits since a date/commit and explore each changed file.

    For each commit, reads every changed file and runs file exploration,
    creating one entry per file with commit context.

    Example:
        code-expert walk-commits --since 2026-03-01
        code-expert walk-commits --since-commit abc1234
        code-expert walk-commits --since-last
        code-expert walk-commits --since "1 week ago" --dry-run
    """
    from .caffeinate import hold as _caffeinate
    _caffeinate()

    repo_path = _get_repo(ctx)
    abs_repo = os.path.abspath(repo_path)
    project_dir = _get_project_dir(ctx)
    model = ctx.obj["model"]

    # Resolve the starting point
    if since_last:
        checkpoint = load_diff_checkpoint(project_dir)
        if not checkpoint:
            click.echo("No previous diff checkpoint found. Use --since DATE first.", err=True)
            sys.exit(1)
        since_commit = checkpoint["head"]
        click.echo(f"Walking from checkpoint {since_commit[:8]}", err=True)
    elif not since and not since_commit:
        click.echo("Error: provide --since DATE, --since-commit SHA, or --since-last", err=True)
        sys.exit(1)

    # Get commits with their changed files
    try:
        commits = list_commits_with_files(
            since=since, since_commit=since_commit, cwd=abs_repo,
        )
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if not commits:
        click.echo("No commits found.", err=True)
        sys.exit(0)

    # Deduplicate files across commits — only explore each file once,
    # using the latest commit that touched it
    file_to_commit: dict[str, dict] = {}
    deleted_files: set[str] = set()
    for commit in commits:
        for f in commit["files"]:
            file_to_commit[f] = commit
        for f in commit.get("deleted_files", []):
            deleted_files.add(f)
    # If a file was deleted then re-added across commits, it's not deleted
    for f in list(deleted_files):
        abs_path = os.path.join(abs_repo, f)
        if os.path.isfile(abs_path):
            deleted_files.discard(f)

    total_files = len(file_to_commit)
    click.echo(
        f"Found {len(commits)} commit(s), {total_files} unique file(s) to explore",
        err=True,
    )
    if deleted_files:
        click.echo(f"  {len(deleted_files)} file(s) deleted", err=True)

    if dry_run:
        for commit in commits:
            click.echo(f"\n  {commit['sha'][:8]} {commit['subject']}")
            for f in commit["files"]:
                marker = " " if file_to_commit[f] is commit else " (earlier version, skip)"
                if f in deleted_files:
                    marker = " [DELETED]"
                click.echo(f"    {marker} {f}")
        click.echo(f"\nWould explore {total_files} file(s)")
        if deleted_files:
            click.echo(f"Would retract beliefs sourced from {len(deleted_files)} deleted file(s)")
        return

    # Retract beliefs sourced from deleted files
    if deleted_files:
        _retract_beliefs_for_deleted_files(deleted_files)

    if not check_model_available(model):
        click.echo(f"Error: Model '{model}' CLI not available", err=True)
        sys.exit(1)

    # Explore each unique file
    explored = 0
    skipped = 0
    from .topics import Topic
    for file_path, commit in file_to_commit.items():
        explored += 1
        click.echo(f"\n{'=' * 40}", err=True)
        click.echo(
            f"[{explored}/{total_files}] {file_path} (from {commit['sha'][:8]})",
            err=True,
        )
        click.echo(f"{'=' * 40}", err=True)

        topic = Topic(
            title=f"{commit['subject']} — {file_path}",
            kind="file",
            target=file_path,
            source=f"walk-commits:{commit['sha'][:8]}",
        )

        abs_path = os.path.join(abs_repo, file_path)
        if not os.path.isfile(abs_path):
            click.echo(f"  File not found (deleted?): {file_path}, skipping", err=True)
            skipped += 1
            continue

        _run_file_topic(ctx, topic, model, abs_repo)

    # Save checkpoint so --since-last works next time
    save_diff_checkpoint(project_dir, cwd=abs_repo)
    click.echo(f"\nWalked {len(commits)} commit(s), explored {explored - skipped} file(s) ({skipped} skipped)", err=True)
    click.echo("Diff checkpoint saved.", err=True)


# --- propose-beliefs ---


@cli.command("propose-beliefs")
@click.option("--batch-size", type=int, default=5, help="Entries per LLM batch (default: 5)")
@click.option("--output", default="proposed-beliefs.md", help="Output file")
@click.option("--model", "-m", default=None, help="Override model")
@click.option("--entry", "entry_paths", multiple=True, type=click.Path(exists=True),
              help="Process specific entry file(s) instead of all entries")
@click.option("--all", "process_all", is_flag=True,
              help="Re-process all entries (ignore processed tracking)")
@click.option("--auto", "auto_accept", is_flag=True, default=False,
              help="Automatically accept all proposed beliefs (no review step)")
@click.pass_context
def propose_beliefs(ctx, batch_size, output, model, entry_paths, process_all, auto_accept):
    """Extract candidate beliefs from entries for human review."""
    from .caffeinate import hold as _caffeinate
    _caffeinate()
    if model is None:
        model = ctx.obj["model"]
    timeout = ctx.obj["timeout"]

    if not check_model_available(model):
        click.echo(f"Error: Model '{model}' CLI not available", err=True)
        sys.exit(1)

    # Collect entries
    if entry_paths:
        entries = [Path(p) for p in entry_paths]
    else:
        input_dir = Path("entries")
        if not input_dir.exists():
            click.echo("No entries/ directory found. Run explorations first.")
            sys.exit(1)
        entries = sorted(input_dir.rglob("*.md"))

    if not entries:
        click.echo("No .md files found.")
        return

    # Filter out already-processed entries (unless --all or --entry)
    processed_path = Path(PROJECT_DIR) / "proposed-entries.json"
    processed = _load_processed(processed_path)
    if not process_all and not entry_paths:
        total = len(entries)
        entries = _filter_unprocessed(entries, processed)
        skipped = total - len(entries)
        if skipped:
            click.echo(f"Skipping {skipped} already-processed entries (use --all to reprocess)")
        if not entries:
            click.echo("No new entries to process.")
            return

    # Load existing beliefs (IDs + text + source) for dedup context
    existing_beliefs = _load_existing_beliefs(Path("beliefs.md"))
    existing_ids = {b["id"] for b in existing_beliefs}

    if existing_ids:
        click.echo(f"Found {len(existing_ids)} existing beliefs (will skip duplicates)")

    # Compute belief embeddings once for all batches (if fastembed available)
    belief_vectors = None
    if existing_beliefs and _has_embeddings():
        click.echo("Computing belief embeddings for semantic dedup...")
        cache_path = Path(PROJECT_DIR) / "belief-vectors.json"
        belief_vectors = _get_belief_embeddings(existing_beliefs, cache_path)
        click.echo(f"  {len(belief_vectors)} belief vectors ready")
    elif existing_beliefs:
        click.echo("(install fastembed for semantic dedup: uv pip install 'ftl-code-expert[embeddings]')")

    click.echo(f"Reading {len(entries)} entries...")

    # Batch entries — track paths per batch for relevance scoring
    batches = []
    batch_paths = []
    current_batch = []
    current_paths = []
    for entry_path in entries:
        content = entry_path.read_text()
        if len(content) > 10000:
            content = content[:10000] + "\n[Truncated]"
        current_batch.append(f"--- FILE: {entry_path} ---\n{content}")
        current_paths.append(str(entry_path))
        if len(current_batch) >= batch_size:
            batches.append("\n\n".join(current_batch))
            batch_paths.append(current_paths)
            current_batch = []
            current_paths = []
    if current_batch:
        batches.append("\n\n".join(current_batch))
        batch_paths.append(current_paths)

    click.echo(f"Processing {len(batches)} batches (batch size: {batch_size})...")

    all_proposals = []
    for i, batch_text in enumerate(batches):
        click.echo(f"  Batch {i + 1}/{len(batches)}...")
        existing_context = _build_dedup_context(
            existing_beliefs, batch_paths[i], batch_text,
            belief_vectors=belief_vectors,
        )
        prompt = PROPOSE_BELIEFS_CODE.format(entries=batch_text) + existing_context
        try:
            result = invoke_sync(prompt, model=model, timeout=timeout)
            all_proposals.append(result)
        except Exception as e:
            click.echo(f"  ERROR: {e}")
            continue

    # Filter out proposals whose IDs already exist
    filtered_proposals = []
    skipped = 0
    for proposal in all_proposals:
        lines = proposal.split("\n")
        filtered_lines = []
        skip_until_next = False
        for line in lines:
            m = re.match(r"^### \[?(?:ACCEPT|REJECT)\]? (\S+)", line)
            if m:
                belief_id = m.group(1)
                if belief_id in existing_ids:
                    skip_until_next = True
                    skipped += 1
                    continue
                else:
                    skip_until_next = False
            if skip_until_next:
                # Skip lines until the next ### header
                if line.startswith("### "):
                    skip_until_next = False
                    filtered_lines.append(line)
                continue
            filtered_lines.append(line)
        filtered_proposals.append("\n".join(filtered_lines))

    if skipped:
        click.echo(f"  Filtered {skipped} already-accepted beliefs")

    # Record processed entries
    _save_processed(processed_path, entries, processed)

    if auto_accept:
        # Parse proposals and accept all directly
        accept_pattern = re.compile(
            r"^### \[?(?:ACCEPT(?:/REJECT)?|REJECT)\]? (\S+)\n(.+?)\n- Source: (.+?)(?:\n|$)",
            re.MULTILINE,
        )
        matches = []
        for proposal in filtered_proposals:
            matches.extend(accept_pattern.findall(proposal))
        if not matches:
            click.echo("No beliefs extracted from proposals.")
            return
        click.echo(f"\nAuto-accepting {len(matches)} beliefs...")
        _accept_proposals(matches)
        return

    # Write proposals file (append if it already exists)
    source_desc = ", ".join(str(e) for e in entries) if entry_paths else f"{len(entries)} entries from entries/"
    output_path = Path(output)
    if output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open("a") as f:
            f.write(f"\n---\n\n")
            f.write(f"**Generated:** {date.today().isoformat()}\n")
            f.write(f"**Source:** {source_desc}\n")
            f.write(f"**Model:** {model}\n\n")
            for proposal in filtered_proposals:
                f.write(proposal)
                f.write("\n\n")
        click.echo(f"\nAppended to {output_path}")
    else:
        with output_path.open("w") as f:
            f.write("# Proposed Beliefs\n\n")
            f.write("Edit each entry: change `[ACCEPT/REJECT]` to `[ACCEPT]` or `[REJECT]`.\n")
            f.write("Then run: `code-expert accept-beliefs`\n\n")
            f.write("---\n\n")
            f.write(f"**Generated:** {date.today().isoformat()}\n")
            f.write(f"**Source:** {source_desc}\n")
            f.write(f"**Model:** {model}\n\n")
            for proposal in filtered_proposals:
                f.write(proposal)
                f.write("\n\n")
        click.echo(f"\nWrote {output_path}")

    click.echo("Review the file, mark entries as [ACCEPT] or [REJECT], then run:")
    click.echo("  code-expert accept-beliefs")


def _load_processed(path: Path) -> dict[str, str]:
    """Load processed entries tracking {path: content_hash}."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _save_processed(path: Path, entries: list[Path], existing: dict[str, str]):
    """Record entries as processed by content hash."""
    import hashlib
    updated = dict(existing)
    for entry_path in entries:
        content = entry_path.read_text()
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        updated[str(entry_path)] = content_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(updated, indent=2) + "\n")


def _filter_unprocessed(entries: list[Path], processed: dict[str, str]) -> list[Path]:
    """Return entries that are new or modified since last propose."""
    import hashlib
    unprocessed = []
    for entry_path in entries:
        key = str(entry_path)
        if key not in processed:
            unprocessed.append(entry_path)
            continue
        # Re-process if content changed
        content = entry_path.read_text()
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        if content_hash != processed[key]:
            unprocessed.append(entry_path)
    return unprocessed


# --- accept-beliefs ---


def _accept_proposals(matches: list[tuple[str, str, str]]) -> tuple[int, int, int]:
    """Import belief proposals into the primary store.

    Returns (added, skipped, failed) counts.
    """
    if _has_reasons():
        click.echo("Using reasons as primary store...")
        added = 0
        failed = 0
        skipped = 0
        for belief_id, claim_text, source in matches:
            result = subprocess.run(
                ["reasons", "add", belief_id, claim_text.strip(),
                 "--source", source.strip()],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                click.echo(f"  Added: {belief_id}")
                added += 1
            else:
                stderr = result.stderr.strip()
                stdout = result.stdout.strip()
                if "already exists" in stderr or "already exists" in stdout:
                    click.echo(f"  EXISTS: {belief_id}")
                    skipped += 1
                else:
                    click.echo(f"  FAIL: {belief_id}: {stderr or stdout}")
                    failed += 1

        click.echo(f"\nAccepted {added} beliefs ({skipped} existing, {failed} failed)")

        if added > 0:
            _reasons_export()
        return added, skipped, failed

    # Try beliefs batch mode
    if _accept_batch(matches):
        return len(matches), 0, 0

    # Fall back to per-belief add via beliefs CLI
    click.echo("Falling back to per-belief add...")
    added = 0
    failed = 0
    skipped = 0
    for belief_id, claim_text, source in matches:
        try:
            result = subprocess.run(
                ["beliefs", "add",
                 "--id", belief_id,
                 "--text", claim_text.strip(),
                 "--source", source.strip()],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                click.echo(f"  Added: {belief_id}")
                added += 1
            else:
                stderr = result.stderr.strip()
                if "already exists" in stderr or "already exists" in result.stdout:
                    click.echo(f"  EXISTS: {belief_id}")
                    skipped += 1
                else:
                    click.echo(f"  FAIL: {belief_id}: {stderr or result.stdout.strip()}")
                    failed += 1
        except FileNotFoundError:
            click.echo("ERROR: beliefs CLI not found. Install with: uv tool install beliefs")
            sys.exit(1)

    click.echo(f"\nAccepted {added} beliefs ({skipped} existing, {failed} failed)")
    return added, skipped, failed


@cli.command("accept-beliefs")
@click.option("--file", "proposals_file", default="proposed-beliefs.md",
              help="Proposals file (default: proposed-beliefs.md)")
def accept_beliefs(proposals_file):
    """Import accepted beliefs from proposals file."""
    proposals_path = Path(proposals_file)
    if not proposals_path.exists():
        click.echo(f"Proposals file not found: {proposals_file}")
        click.echo("Run: code-expert propose-beliefs")
        sys.exit(1)

    text = proposals_path.read_text()

    pattern = re.compile(
        r"### \[?ACCEPT\]? (\S+)\n"
        r"(.+?)\n"
        r"- Source: (.+?)(?:\n|$)"
    )
    matches = pattern.findall(text)

    if not matches:
        click.echo("No [ACCEPT] entries found in proposals file.")
        click.echo("Edit the file and change [ACCEPT/REJECT] to [ACCEPT] for beliefs to keep.")
        return

    click.echo(f"Found {len(matches)} accepted beliefs")
    _accept_proposals(matches)


def _load_existing_beliefs(beliefs_path: Path) -> list[dict]:
    """Load existing beliefs. Prefers reasons.db if available, falls back to beliefs.md."""
    # Try reasons first for authoritative IDs (beliefs.md may be stale)
    if _has_reasons() and Path("reasons.db").exists():
        reasons_beliefs = _load_existing_from_reasons()
        if reasons_beliefs:
            # Enrich with text/source from beliefs.md if available
            md_beliefs = {}
            if beliefs_path.exists():
                for b in _parse_beliefs_md(beliefs_path):
                    md_beliefs[b["id"]] = b
            for b in reasons_beliefs:
                if b["id"] in md_beliefs:
                    b["text"] = md_beliefs[b["id"]]["text"]
                    b["source"] = md_beliefs[b["id"]]["source"]
            return reasons_beliefs
    return _parse_beliefs_md(beliefs_path)


def _parse_beliefs_md(beliefs_path: Path) -> list[dict]:
    """Parse beliefs.md into list of {id, text, source} dicts."""
    if not beliefs_path.exists():
        return []
    text = beliefs_path.read_text()
    beliefs = []
    sections = re.split(r'^(?=### )', text, flags=re.MULTILINE)
    for section in sections:
        m = re.match(r'^### ([\w-]+) \[(IN|OUT|STALE)\]', section)
        if not m:
            continue
        lines = section.strip().splitlines()
        claim_text = lines[1].strip() if len(lines) > 1 else ""
        source = ""
        for line in lines:
            if line.startswith("- Source:"):
                source = line.replace("- Source:", "").strip()
        beliefs.append({"id": m.group(1), "text": claim_text, "source": source})
    return beliefs


def _has_embeddings() -> bool:
    """Check if fastembed is available."""
    try:
        import numpy  # noqa: F401
        from fastembed import TextEmbedding  # noqa: F401
        return True
    except ImportError:
        return False


# Cache the embedding model across calls within a session
_embed_model = None


def _get_embed_model():
    """Lazy-load the fastembed model."""
    global _embed_model
    if _embed_model is None:
        from fastembed import TextEmbedding
        _embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embed_model


def _load_belief_vectors(cache_path: Path) -> dict[str, list[float]]:
    """Load cached belief vectors from JSON."""
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _save_belief_vectors(cache_path: Path, vectors: dict[str, list[float]]):
    """Save belief vectors to JSON cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(vectors))


def _get_belief_embeddings(
    beliefs: list[dict], cache_path: Path,
) -> dict[str, list[float]]:
    """Get embeddings for all beliefs, using cache for known ones.

    Cache key is belief ID. Cache is invalidated per-belief when text changes
    (tracked by a text hash suffix in the cache key).
    """
    import hashlib

    model = _get_embed_model()
    cached = _load_belief_vectors(cache_path)

    # Build map of belief_id -> expected cache key (id + text hash)
    def _cache_key(belief):
        text_hash = hashlib.sha256(belief["text"].encode()).hexdigest()[:8]
        return f"{belief['id']}:{text_hash}"

    # Find beliefs that need embedding
    needed = []
    needed_keys = []
    result = {}
    for belief in beliefs:
        key = _cache_key(belief)
        if key in cached:
            result[belief["id"]] = cached[key]
        else:
            needed.append(belief)
            needed_keys.append(key)

    # Embed missing beliefs
    if needed:
        texts = [b["text"] for b in needed]
        vectors = list(model.embed(texts))
        for belief, key, vec in zip(needed, needed_keys, vectors):
            vec_list = vec.tolist()
            cached[key] = vec_list
            result[belief["id"]] = vec_list

        # Prune stale entries from cache (IDs no longer in beliefs)
        current_keys = {_cache_key(b) for b in beliefs}
        cached = {k: v for k, v in cached.items() if k in current_keys}
        _save_belief_vectors(cache_path, cached)

    return result


def _score_by_embedding(
    beliefs: list[dict],
    belief_vectors: dict[str, list[float]],
    batch_text: str,
    batch_entry_paths: list[str],
) -> list[tuple[float, dict]]:
    """Score beliefs by embedding similarity to batch content."""
    import numpy as np

    model = _get_embed_model()

    # Embed the batch text (truncate to avoid excessive embedding time)
    batch_summary = batch_text[:4000]
    query_vec = np.array(list(model.embed([batch_summary]))[0], dtype=np.float32)

    scored = []
    for belief in beliefs:
        vec = belief_vectors.get(belief["id"])
        if vec is None:
            scored.append((0.0, belief))
            continue
        belief_vec = np.array(vec, dtype=np.float32)
        # Cosine similarity
        dot = np.dot(query_vec, belief_vec)
        norm = np.linalg.norm(query_vec) * np.linalg.norm(belief_vec)
        similarity = float(dot / norm) if norm > 0 else 0.0
        # Source match bonus
        if belief["source"] and any(belief["source"] in p or p in belief["source"]
                                     for p in batch_entry_paths):
            similarity += 1.0
        scored.append((similarity, belief))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _score_by_keywords(
    beliefs: list[dict],
    batch_text: str,
    batch_entry_paths: list[str],
) -> list[tuple[float, dict]]:
    """Score beliefs by keyword overlap (fallback when embeddings unavailable)."""
    batch_words = set(re.findall(r'[a-z]{3,}', batch_text.lower()))

    scored = []
    for belief in beliefs:
        score = 0.0
        if belief["source"] and any(belief["source"] in p or p in belief["source"]
                                     for p in batch_entry_paths):
            score += 1000
        belief_words = set(re.findall(r'[a-z]{3,}', belief["text"].lower()))
        belief_words |= set(belief["id"].replace("-", " ").lower().split())
        overlap = len(batch_words & belief_words)
        score += overlap
        scored.append((score, belief))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _build_dedup_context(
    existing_beliefs: list[dict],
    batch_entry_paths: list[str],
    batch_text: str,
    max_detailed: int = 50,
    max_compact: int = 200,
    belief_vectors: dict[str, list[float]] | None = None,
) -> str:
    """Build per-batch dedup context: relevant beliefs with text, rest as compact IDs.

    Uses embedding similarity when belief_vectors is provided, falls back
    to keyword overlap otherwise. Detailed beliefs include claim text for
    semantic dedup; compact beliefs are comma-separated IDs.
    """
    if not existing_beliefs:
        return ""

    # Score beliefs by relevance to this batch
    if belief_vectors:
        scored = _score_by_embedding(
            existing_beliefs, belief_vectors, batch_text, batch_entry_paths,
        )
    else:
        scored = _score_by_keywords(
            existing_beliefs, batch_text, batch_entry_paths,
        )

    # Split into detailed (with text) and compact (ID only)
    detailed = scored[:max_detailed]
    compact = scored[max_detailed:max_detailed + max_compact]

    parts = [
        "\n\n## Already Accepted Beliefs\n\n"
        "The following beliefs already exist. Do NOT propose beliefs with these IDs "
        "or that duplicate their meaning under different names.\n"
    ]

    if detailed:
        parts.append("\nRelevant existing beliefs:")
        for _, belief in detailed:
            parts.append(f"- `{belief['id']}`: {belief['text']}")

    if compact:
        compact_ids = ", ".join(b["id"] for _, b in compact)
        parts.append(f"\nOther existing IDs: {compact_ids}")

    return "\n".join(parts) + "\n"


def _accept_batch(matches: list[tuple[str, str, str]]) -> bool:
    """Try to add all beliefs in one subprocess via 'beliefs add-batch'.

    Returns True if batch mode succeeded, False to fall back to per-belief.
    """
    # Build JSON lines
    lines = []
    for belief_id, claim_text, source in matches:
        lines.append(json.dumps({
            "id": belief_id,
            "text": claim_text.strip(),
            "source": source.strip(),
        }))
    json_input = "\n".join(lines)

    try:
        result = subprocess.run(
            ["beliefs", "add-batch"],
            input=json_input,
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        click.echo("ERROR: beliefs CLI not found. Install with: uv tool install beliefs")
        sys.exit(1)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        # add-batch not available in this version of beliefs
        if "invalid choice" in stderr or "unrecognized arguments" in stderr:
            return False
        click.echo(f"Batch failed: {stderr}")
        return False

    # Print batch output
    if result.stdout.strip():
        click.echo(result.stdout.strip())

    return True


def _reasons_export():
    """Re-export beliefs.md and network.json from reasons after adding beliefs."""
    beliefs_path = Path("beliefs.md")
    network_path = Path("network.json")

    result = subprocess.run(
        ["reasons", "export-markdown"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        beliefs_path.write_text(result.stdout)
        click.echo(f"Updated {beliefs_path}")

    result = subprocess.run(
        ["reasons", "export"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        network_path.write_text(result.stdout)
        click.echo(f"Updated {network_path}")


def _load_existing_from_reasons() -> list[dict]:
    """Load existing beliefs from reasons.db via CLI."""
    result = subprocess.run(
        ["reasons", "list"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []
    beliefs = []
    for line in result.stdout.splitlines():
        # Format: "  [+] node-id  (premise)" or "  [-] node-id  (premise)"
        m = re.match(r'\s*\[[+-]\]\s+([\w-]+)', line)
        if m:
            beliefs.append({"id": m.group(1), "text": "", "source": ""})
    return beliefs


# --- status ---


@cli.command()
@click.pass_context
def status(ctx):
    """Show code-expert dashboard."""
    config = _load_config()

    click.echo("=== Code Expert Status ===\n")

    if config:
        click.echo(f"Repo:     {config.get('repo_path', 'unknown')}")
        click.echo(f"Domain:   {config.get('domain', 'unknown')}")
        click.echo(f"Created:  {config.get('created', 'unknown')}")
    else:
        click.echo("Not initialized. Run: code-expert init <repo-path>")
        return

    click.echo()

    # Count entries
    entries_dir = Path("entries")
    entry_count = len(list(entries_dir.rglob("*.md"))) if entries_dir.exists() else 0
    click.echo(f"Entries:  {entry_count}")

    # Count beliefs — prefer reasons if available
    if _has_reasons() and Path("reasons.db").exists():
        result = subprocess.run(
            ["reasons", "list"], capture_output=True, text=True,
        )
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            r_in = sum(1 for l in lines if l.strip().startswith("[+]"))
            r_out = sum(1 for l in lines if l.strip().startswith("[-]"))
            click.echo(f"Beliefs:  {r_in} IN, {r_out} OUT (reasons.db)")
        else:
            click.echo("Beliefs:  reasons.db error")
    else:
        beliefs_path = Path("beliefs.md")
        beliefs_in = 0
        beliefs_stale = 0
        if beliefs_path.exists():
            text = beliefs_path.read_text()
            beliefs_in = len(re.findall(r"^### \S+ \[IN\]", text, re.MULTILINE))
            beliefs_stale = len(re.findall(r"^### \S+ \[STALE\]", text, re.MULTILINE))
        status_parts = [f"{beliefs_in} IN"]
        if beliefs_stale:
            status_parts.append(f"{beliefs_stale} STALE")
        click.echo(f"Beliefs:  {', '.join(status_parts)}")

    # Count nogoods
    nogoods_path = Path("nogoods.md")
    nogood_count = 0
    if nogoods_path.exists():
        text = nogoods_path.read_text()
        nogood_count = len(re.findall(r"^### nogood-\d+", text, re.MULTILINE))
    click.echo(f"Nogoods:  {nogood_count}")

    # Count topics
    queue = load_queue(_get_project_dir(ctx))
    pending = sum(1 for t in queue if t.status == "pending")
    done = sum(1 for t in queue if t.status == "done")
    skipped = sum(1 for t in queue if t.status == "skipped")
    click.echo(f"Topics:   {pending} pending, {done} done, {skipped} skipped")

    # Diff checkpoint
    repo_path = _get_repo(ctx)
    unexplored = commits_since_checkpoint(_get_project_dir(ctx), cwd=repo_path)
    if unexplored is not None:
        checkpoint = load_diff_checkpoint(_get_project_dir(ctx))
        ts = checkpoint["timestamp"] if checkpoint else "?"
        if unexplored == 0:
            click.echo(f"Diff:     up to date (last: {ts})")
        else:
            click.echo(f"Diff:     {unexplored} unexplored commit(s) (last: {ts})")
    else:
        click.echo("Diff:     no checkpoint (run: code-expert explain diff --since DATE)")

    # Count proposals
    proposals_path = Path("proposed-beliefs.md")
    if proposals_path.exists():
        text = proposals_path.read_text()
        total = len(re.findall(r"^### \[(?:ACCEPT|REJECT|ACCEPT/REJECT)\]", text, re.MULTILINE))
        accepted = len(re.findall(r"^### \[ACCEPT\]", text, re.MULTILINE))
        click.echo(f"Proposed: {total} candidates ({accepted} accepted)")


# --- generate-spec ---


def _gather_beliefs_for_spec(keywords: list[str]) -> list[dict]:
    """Gather IN beliefs matching any keyword from beliefs.md."""
    beliefs_path = Path("beliefs.md")
    if not beliefs_path.exists():
        return []

    text = beliefs_path.read_text()
    sections = re.split(r'^(?=### )', text, flags=re.MULTILINE)
    matched = []

    for section in sections:
        m = re.match(r'^### ([\w-]+) \[(IN|OUT)\]\s*(\w+)?', section)
        if not m:
            continue
        belief_id = m.group(1)
        status = m.group(2)
        belief_type = m.group(3) or "OBSERVATION"
        if status != "IN":
            continue

        lines = section.strip().splitlines()
        claim_text = lines[1].strip() if len(lines) > 1 else ""
        source = ""
        depends = ""
        for line in lines:
            if line.startswith("- Source:"):
                source = line.replace("- Source:", "").strip()
            if line.startswith("- Depends on:"):
                depends = line.replace("- Depends on:", "").strip()

        # Match against keywords (check ID, claim text, source)
        searchable = f"{belief_id} {claim_text} {source}".lower()
        if any(kw.lower() in searchable for kw in keywords):
            matched.append({
                "id": belief_id,
                "status": status,
                "type": belief_type,
                "text": claim_text,
                "source": source,
                "depends": depends,
            })

    return matched


def _gather_source_files(repo_path: str, beliefs: list[dict]) -> dict[str, str]:
    """Read source files referenced by beliefs."""
    # Collect unique file paths from belief sources and IDs
    file_paths = set()
    for belief in beliefs:
        # Extract paths from source entries
        source = belief.get("source", "")
        # Source format: entries/2026/03/11/src-redhat_agents-workflow-synthesizer.md
        # Extract the implied source file: src/redhat_agents/workflow/synthesizer.py
        m = re.search(r'src[-/](.+?)\.md', source)
        if m:
            # Convert dashes back to path separators
            path = "src/" + m.group(1).replace("-", "/") + ".py"
            file_paths.add(path)

    # Also look for common patterns in belief IDs
    source_files = {}
    for path in sorted(file_paths):
        abs_path = os.path.join(repo_path, path)
        content = get_file_content(abs_path)
        if content is not None:
            # Truncate very large files
            if len(content) > 20000:
                content = content[:20000] + "\n# [Truncated at 20000 chars]"
            source_files[path] = content

    return source_files


def _format_beliefs_for_prompt(beliefs: list[dict]) -> str:
    """Format beliefs into prompt-friendly text."""
    lines = []
    for b in beliefs:
        lines.append(f"### {b['id']} [{b['status']}] {b['type']}")
        lines.append(b['text'])
        if b.get('depends'):
            lines.append(f"- Depends on: {b['depends']}")
        if b.get('source'):
            lines.append(f"- Source: {b['source']}")
        lines.append("")
    return "\n".join(lines)


def _format_source_code(source_files: dict[str, str]) -> str:
    """Format source files into prompt-friendly text."""
    if not source_files:
        return "(No source files found)"
    parts = []
    for path, content in source_files.items():
        parts.append(f"### {path}\n\n```python\n{content}\n```")
    return "\n\n".join(parts)


@cli.command("generate-spec")
@click.argument("component")
@click.option("--keywords", "-k", required=True,
              help="Comma-separated keywords to match beliefs (e.g., 'synth,citation,tree-synth')")
@click.option("--output", "-o", default=None,
              help="Output file (default: docs/specs/<component>.spec.md)")
@click.option("--source-files", "-s", multiple=True,
              help="Additional source files to include (relative to repo)")
@click.option("--model", "-m", default=None, help="Override model")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show gathered beliefs and files without generating")
@click.pass_context
def generate_spec(ctx, component, keywords, output, source_files, model, dry_run):
    """Generate a spec from beliefs matching keywords.

    Example:
        code-expert generate-spec Synthesizer -k "synth,citation,tree-synth,pre-merge"
    """
    from .caffeinate import hold as _caffeinate
    from .prompts.spec import GENERATE_SPEC_PROMPT
    _caffeinate()

    if model is None:
        model = ctx.obj["model"]
    timeout = ctx.obj["timeout"]
    repo_path = _get_repo(ctx)

    # Parse keywords
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
    click.echo(f"Gathering beliefs matching: {', '.join(kw_list)}", err=True)

    # Gather beliefs
    beliefs = _gather_beliefs_for_spec(kw_list)
    if not beliefs:
        click.echo("No matching beliefs found.", err=True)
        sys.exit(1)
    click.echo(f"Found {len(beliefs)} matching beliefs", err=True)

    # Gather source files
    src_files = _gather_source_files(repo_path, beliefs)

    # Add explicit source files (expand directories to .py files)
    for sf in source_files:
        abs_path = os.path.join(repo_path, sf) if not os.path.isabs(sf) else sf
        if os.path.isdir(abs_path):
            for root, _, files in os.walk(abs_path):
                for fname in sorted(files):
                    if not fname.endswith(".py"):
                        continue
                    fpath = os.path.join(root, fname)
                    content = get_file_content(fpath)
                    if content is not None:
                        if len(content) > 20000:
                            content = content[:20000] + "\n# [Truncated at 20000 chars]"
                        rel = os.path.relpath(fpath, repo_path)
                        src_files[rel] = content
        else:
            content = get_file_content(abs_path)
            if content is not None:
                if len(content) > 20000:
                    content = content[:20000] + "\n# [Truncated at 20000 chars]"
                rel = os.path.relpath(abs_path, repo_path)
                src_files[rel] = content
            else:
                click.echo(f"WARN: Cannot read {sf}", err=True)

    click.echo(f"Found {len(src_files)} source files", err=True)

    if dry_run:
        click.echo(f"\n=== Beliefs ({len(beliefs)}) ===\n")
        for b in beliefs:
            click.echo(f"  {b['id']}: {b['text'][:80]}")
        click.echo(f"\n=== Source Files ({len(src_files)}) ===\n")
        for path in sorted(src_files):
            click.echo(f"  {path} ({len(src_files[path])} chars)")
        return

    if not check_model_available(model):
        click.echo(f"Error: Model '{model}' CLI not available", err=True)
        sys.exit(1)

    # Build prompt
    beliefs_text = _format_beliefs_for_prompt(beliefs)
    source_code = _format_source_code(src_files)
    file_list = ", ".join(sorted(src_files.keys())) if src_files else "(none)"

    config = _load_config()
    domain = config.get("domain", "code-expert") if config else "code-expert"

    prompt = GENERATE_SPEC_PROMPT.format(
        component=component,
        source_files=file_list,
        belief_count=len(beliefs),
        beliefs_text=beliefs_text,
        source_code=source_code,
        domain=domain,
    )

    click.echo(f"Generating spec with {model}...", err=True)
    try:
        result = asyncio.run(invoke(prompt, model, timeout=timeout))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Write output
    if output is None:
        output = f"docs/specs/{component.lower()}.spec.md"
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result + "\n")

    click.echo(f"\nWrote {output_path} ({len(beliefs)} beliefs, {len(src_files)} source files)")
    click.echo(f"To update: re-run this command after adding new beliefs.")


# --- generate-prd ---


def _gather_derived_beliefs() -> list[dict]:
    """Gather all DERIVED IN beliefs from beliefs.md."""
    beliefs_path = Path("beliefs.md")
    if not beliefs_path.exists():
        return []

    text = beliefs_path.read_text()
    sections = re.split(r'^(?=### )', text, flags=re.MULTILINE)
    derived = []

    for section in sections:
        m = re.match(r'^### ([\w-]+) \[(IN|OUT)\]\s*(\w+)?', section)
        if not m:
            continue
        if m.group(2) != "IN" or m.group(3) != "DERIVED":
            continue

        lines = section.strip().splitlines()
        claim_text = lines[1].strip() if len(lines) > 1 else ""
        depends = ""
        for line in lines:
            if line.startswith("- Depends on:"):
                depends = line.replace("- Depends on:", "").strip()

        derived.append({
            "id": m.group(1),
            "text": claim_text,
            "depends": depends,
        })

    return derived


def _gather_specs() -> dict[str, str]:
    """Read all spec files from docs/specs/."""
    specs_dir = Path("docs/specs")
    if not specs_dir.exists():
        return {}

    specs = {}
    for spec_file in sorted(specs_dir.glob("*.spec.md")):
        content = spec_file.read_text()
        # Truncate very large specs to keep prompt manageable
        if len(content) > 30000:
            content = content[:30000] + "\n\n[Truncated at 30000 chars]"
        specs[spec_file.stem.replace(".spec", "")] = content

    return specs


@cli.command("generate-prd")
@click.argument("product_name")
@click.option("--output", "-o", default=None,
              help="Output file (default: docs/prd/<product>.prd.md)")
@click.option("--specs", "-s", multiple=True,
              help="Specific spec names to include (default: all)")
@click.option("--model", "-m", default=None, help="Override model")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show gathered data without generating")
@click.pass_context
def generate_prd(ctx, product_name, output, specs, model, dry_run):
    """Generate a PRD from beliefs and specs.

    Example:
        code-expert generate-prd FTL2
        code-expert generate-prd "My Product" -s policy -s gate
    """
    from .caffeinate import hold as _caffeinate
    from .prompts.prd import GENERATE_PRD_PROMPT
    _caffeinate()

    if model is None:
        model = ctx.obj["model"]
    timeout = ctx.obj["timeout"]

    # Gather derived beliefs
    derived = _gather_derived_beliefs()
    click.echo(f"Found {len(derived)} derived beliefs", err=True)

    # Gather all IN beliefs for count
    all_beliefs = _gather_beliefs_for_spec([""])  # empty string matches nothing
    # Actually count from beliefs.md directly
    beliefs_path = Path("beliefs.md")
    belief_count = 0
    if beliefs_path.exists():
        belief_count = len(re.findall(r'^### [\w-]+ \[IN\]', beliefs_path.read_text(), re.MULTILINE))
    click.echo(f"Total IN beliefs: {belief_count}", err=True)

    # Gather specs
    all_specs = _gather_specs()
    if specs:
        all_specs = {k: v for k, v in all_specs.items() if k in specs}
    click.echo(f"Found {len(all_specs)} spec(s): {', '.join(sorted(all_specs.keys()))}", err=True)

    if not derived and not all_specs:
        click.echo("No derived beliefs or specs found. Run generate-spec first.", err=True)
        sys.exit(1)

    if dry_run:
        click.echo(f"\n=== Derived Beliefs ({len(derived)}) ===\n")
        for b in derived:
            click.echo(f"  {b['id']}: {b['text'][:80]}")
        click.echo(f"\n=== Specs ({len(all_specs)}) ===\n")
        for name in sorted(all_specs):
            click.echo(f"  {name} ({len(all_specs[name])} chars)")
        return

    if not check_model_available(model):
        click.echo(f"Error: Model '{model}' CLI not available", err=True)
        sys.exit(1)

    # Format derived beliefs
    derived_text = []
    for b in derived:
        derived_text.append(f"### {b['id']}")
        derived_text.append(b['text'])
        if b.get('depends'):
            derived_text.append(f"- Depends on: {b['depends']}")
        derived_text.append("")
    derived_beliefs = "\n".join(derived_text)

    # Format specs (include full content, truncated)
    specs_parts = []
    for name, content in sorted(all_specs.items()):
        specs_parts.append(f"## Spec: {name}\n\n{content}")
    specs_text = "\n\n---\n\n".join(specs_parts)

    config = _load_config()
    domain = config.get("domain", "code-expert") if config else "code-expert"

    prompt = GENERATE_PRD_PROMPT.format(
        product_name=product_name,
        domain=domain,
        derived_beliefs=derived_beliefs,
        specs_text=specs_text,
        belief_count=belief_count,
        spec_count=len(all_specs),
    )

    click.echo(f"Generating PRD with {model} ({len(prompt)} chars)...", err=True)
    try:
        result = asyncio.run(invoke(prompt, model, timeout=timeout))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Write output
    if output is None:
        output = f"docs/prd/{product_name.lower().replace(' ', '-')}.prd.md"
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result + "\n")

    click.echo(f"\nWrote {output_path} ({len(derived)} derived beliefs, {len(all_specs)} specs)")


# --- derive ---


def _load_network() -> dict:
    """Load network.json (exported from reasons)."""
    network_path = Path("network.json")
    if not network_path.exists():
        # Try exporting from reasons
        if _has_reasons():
            import subprocess
            result = subprocess.run(
                ["reasons", "export"], capture_output=True, text=True,
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        return {"nodes": {}}
    return json.loads(network_path.read_text())


@cli.command("derive")
@click.option("--output", "-o", default="proposed-derivations.md",
              help="Output file (default: proposed-derivations.md)")
@click.option("--model", "-m", default=None, help="Override model")
@click.option("--auto", "auto_add", is_flag=True, default=False,
              help="Automatically add proposals to reasons (no review step)")
@click.option("--exhaust", "exhaust", is_flag=True, default=False,
              help="Loop until no new derivations (implies --auto)")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what would be sent to the LLM without invoking it")
@click.option("--budget", type=int, default=300,
              help="Maximum number of beliefs in prompt (default: 300)")
@click.option("--sample/--no-sample", default=True,
              help="Randomly sample beliefs instead of alphabetical truncation (default: on)")
@click.option("--topic", default=None,
              help="Keyword filter — only include beliefs matching these keywords")
@click.option("--max-rounds", type=int, default=10,
              help="Maximum rounds for --exhaust (default: 10)")
@click.pass_context
def derive(ctx, output, model, auto_add, exhaust, dry_run, budget, sample, topic, max_rounds):
    """Derive deeper reasoning chains from existing beliefs.

    Delegates to `reasons derive` which handles prompt building, LLM
    invocation, proposal validation (including Jaccard dedup against
    retracted beliefs), and network updates.

    Example:
        code-expert derive              # propose derivations
        code-expert derive --auto       # propose and add automatically
        code-expert derive --exhaust    # loop until no new derivations
    """
    from .caffeinate import hold as _caffeinate
    _caffeinate()

    if not _has_reasons():
        click.echo("Error: reasons CLI required. Install with: uv tool install ftl-reasons", err=True)
        sys.exit(1)

    if model is None:
        model = ctx.obj["model"]
    timeout = ctx.obj["timeout"] if ctx.obj["timeout"] != 300 else 600

    cmd = ["reasons", "derive", "-m", model, "--timeout", str(timeout),
           "--budget", str(budget), "-o", output]
    if sample:
        cmd.append("--sample")
    if auto_add or exhaust:
        cmd.append("--auto")
    if exhaust:
        cmd.extend(["--exhaust", "--max-rounds", str(max_rounds)])
    if dry_run:
        cmd.append("--dry-run")
    if topic:
        cmd.extend(["--topic", topic])

    click.echo(f"Running: {' '.join(cmd)}", err=True)
    result = subprocess.run(cmd)

    if result.returncode != 0:
        sys.exit(result.returncode)

    # Re-export after derive modifies the database
    if (auto_add or exhaust) and not dry_run:
        _reasons_export()


# --- file-issues ---


def _detect_platform(repo_path: str) -> tuple[str | None, str | None]:
    """Detect GitHub/GitLab from git remote and return (platform, owner/repo)."""
    result = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        capture_output=True, text=True, cwd=repo_path,
    )
    if result.returncode != 0:
        return None, None
    url = result.stdout.strip()

    # Parse owner/repo from URL
    # Handles: git@github.com:owner/repo.git, https://github.com/owner/repo.git
    m = re.match(r"(?:https?://|git@)([^/:]+)[:/](.+?)(?:\.git)?$", url)
    if not m:
        return None, None
    host, path = m.group(1), m.group(2)

    if "github" in host:
        return "github", path
    elif "gitlab" in host:
        return "gitlab", path
    return None, None


def _find_existing_issues(platform: str, repo_slug: str,
                          blocker_ids: list[str],
                          blocker_texts: dict[str, str]) -> set[str]:
    """Search for existing issues matching blockers. Returns matched blocker IDs.

    Searches by belief ID and by key words from the blocker text to catch
    issues filed manually with different title formats.
    """
    found = set()

    if platform == "github":
        # Fetch all open+closed issues once (more efficient than per-blocker queries)
        result = subprocess.run(
            ["gh", "issue", "list", "--repo", repo_slug,
             "--state", "all", "--json", "title,number,state",
             "--limit", "200"],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return found
        issues = json.loads(result.stdout)
        titles_lower = [issue["title"].lower() for issue in issues]

        for bid in blocker_ids:
            # Check if belief ID appears in any title
            bid_words = bid.replace("-", " ").lower()
            for title in titles_lower:
                if bid.lower() in title or _titles_match(bid_words, title):
                    found.add(bid)
                    break

    elif platform == "gitlab":
        for bid in blocker_ids:
            # Search by first few words of the blocker text
            search = bid.replace("-", " ")
            result = subprocess.run(
                ["glab", "issue", "list", "--repo", repo_slug,
                 "--search", search, "--in", "title",
                 "--output", "json"],
                capture_output=True, text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                issues = json.loads(result.stdout)
                if issues:
                    found.add(bid)

    return found


def _titles_match(bid_words: str, title: str) -> bool:
    """Check if a belief ID's words substantially overlap with an issue title."""
    # Normalize hyphens to spaces for matching
    bid_tokens = set(bid_words.replace("-", " ").split())
    title_tokens = set(title.replace("-", " ").split())
    # Ignore very short common words
    bid_tokens -= {"is", "a", "an", "the", "and", "or", "not", "no", "in", "on", "to", "for", "of"}
    if not bid_tokens:
        return False
    overlap = bid_tokens & title_tokens
    return len(overlap) >= len(bid_tokens) * 0.6


def _build_issue_body(blocker_node: dict, gated_nodes: list[dict]) -> str:
    """Build issue body from a blocker node and the gated nodes it blocks."""
    lines = [
        f"## Problem",
        f"",
        f"{blocker_node['text']}",
        f"",
        f"## Impact",
        f"",
        f"This blocks {len(gated_nodes)} belief(s) in the knowledge base:",
        f"",
    ]
    for gated in gated_nodes:
        lines.append(f"- **{gated['id']}**: {gated['text'][:120]}")
    lines.extend([
        f"",
        f"## Resolution",
        f"",
        f"When this issue is resolved, retract the blocker belief to restore gated conclusions:",
        f"```bash",
        f"reasons retract {blocker_node['id']} --reason \"Fixed in <PR/commit>\"",
        f"```",
        f"",
        f"---",
        f"*Filed automatically from reasons network by `code-expert file-issues`*",
    ])
    return "\n".join(lines)


def _create_issue(platform: str, repo_slug: str, title: str, body: str,
                  labels: list[str]) -> str | None:
    """Create an issue and return its URL, or None on failure."""
    if platform == "github":
        cmd = ["gh", "issue", "create", "--repo", repo_slug,
               "--title", title, "--body", body]
        for label in labels:
            cmd.extend(["--label", label])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        click.echo(f"  Error creating issue: {result.stderr.strip()}", err=True)
        return None
    elif platform == "gitlab":
        cmd = ["glab", "issue", "create", "--repo", repo_slug,
               "--title", title, "--description", body, "--yes"]
        for label in labels:
            cmd.extend(["--label", label])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # glab prints URL to stdout
            url = result.stdout.strip()
            if not url:
                url = result.stderr.strip()
            return url
        click.echo(f"  Error creating issue: {result.stderr.strip()}", err=True)
        return None
    return None


@cli.command("file-issues")
@click.option("--repo", "-r", "repo_slug", default=None,
              help="Target repo (owner/repo). Auto-detected from git remote if omitted.")
@click.option("--platform", "-p", "platform_override", default=None,
              type=click.Choice(["github", "gitlab"]),
              help="Force platform (auto-detected if omitted)")
@click.option("--label", "-l", "labels", multiple=True,
              help="Extra labels to add (repeatable)")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what would be filed without creating issues")
@click.pass_context
def file_issues(ctx, repo_slug, platform_override, labels, dry_run):
    """File issues from gated beliefs that have active blockers.

    Finds GATE beliefs where outlist nodes are IN (blocking the conclusion),
    and creates issues for each blocker in the target repository.

    Checks for existing issues to avoid duplicates.

    Example:
        code-expert file-issues              # auto-detect repo, file issues
        code-expert file-issues --dry-run    # preview without filing
        code-expert file-issues --repo owner/repo --label bug
    """
    if not _has_reasons():
        click.echo("Error: reasons CLI required. Install with: uv tool install ftl-reasons", err=True)
        sys.exit(1)

    # Load network
    network = _load_network()
    nodes = network.get("nodes", {})
    if not nodes:
        click.echo("No beliefs found. Run explorations first.", err=True)
        sys.exit(1)

    # Find gated nodes with active blockers
    # blocker_id -> list of gated node dicts
    blockers: dict[str, list[dict]] = {}
    for nid, node in nodes.items():
        if node.get("truth_value") != "OUT":
            continue  # Only OUT nodes are actually gated
        if node.get("metadata", {}).get("superseded_by"):
            continue  # Superseded beliefs are OUT by design, not bugs
        for j in node.get("justifications", []):
            if not j.get("outlist"):
                continue
            for outlist_id in j["outlist"]:
                if outlist_id not in nodes:
                    continue
                if nodes[outlist_id].get("truth_value") != "IN":
                    continue
                # This outlist node is IN, blocking an OUT derived belief
                blockers.setdefault(outlist_id, []).append({
                    "id": nid,
                    "text": node.get("text", ""),
                })

    if not blockers:
        click.echo("No active blockers found. All gated beliefs are satisfied.")
        return

    click.echo(f"Found {len(blockers)} active blocker(s) gating {sum(len(v) for v in blockers.values())} belief(s)", err=True)

    # Detect platform
    config = _load_config()
    target_repo_path = config.get("repo_path", os.getcwd()) if config else os.getcwd()

    platform = platform_override
    if not repo_slug or not platform:
        detected_platform, detected_slug = _detect_platform(target_repo_path)
        if not platform:
            platform = detected_platform
        if not repo_slug:
            repo_slug = detected_slug

    if not platform or not repo_slug:
        click.echo("Error: Could not detect platform/repo. Use --repo and --platform flags.", err=True)
        sys.exit(1)

    cli_tool = "gh" if platform == "github" else "glab"
    if not shutil.which(cli_tool):
        click.echo(f"Error: {cli_tool} CLI not found. Install it first.", err=True)
        sys.exit(1)

    click.echo(f"Platform: {platform}, Repo: {repo_slug}", err=True)

    # Check for existing issues
    blocker_ids = list(blockers.keys())
    blocker_texts = {bid: nodes[bid].get("text", "") for bid in blocker_ids}
    if not dry_run:
        click.echo("Checking for existing issues...", err=True)
        existing = _find_existing_issues(platform, repo_slug, blocker_ids, blocker_texts)
        if existing:
            click.echo(f"  {len(existing)} already have issues: {', '.join(sorted(existing))}", err=True)
    else:
        existing = set()

    # File issues
    all_labels = ["reasons-gate"] + list(labels)
    filed = []
    skipped = []

    for blocker_id in sorted(blocker_ids):
        blocker_node = nodes[blocker_id]
        gated = blockers[blocker_id]
        title = f"[{blocker_id}] {blocker_node.get('text', '')[:80]}"
        body = _build_issue_body(
            {"id": blocker_id, "text": blocker_node.get("text", "")},
            gated,
        )

        if blocker_id in existing:
            skipped.append(blocker_id)
            click.echo(f"  SKIP {blocker_id} (issue already exists)")
            continue

        if dry_run:
            click.echo(f"\n  WOULD FILE: {title}")
            click.echo(f"  Blocks: {', '.join(g['id'] for g in gated)}")
            click.echo(f"  Labels: {', '.join(all_labels)}")
            continue

        click.echo(f"  Filing: {blocker_id}...", err=True)
        url = _create_issue(platform, repo_slug, title, body, all_labels)
        if url:
            filed.append((blocker_id, url))
            click.echo(f"  OK {blocker_id}: {url}")
        else:
            click.echo(f"  FAIL {blocker_id}")

    # Summary
    if dry_run:
        click.echo(f"\nDry run: {len(blocker_ids) - len(existing)} would be filed, {len(existing)} already exist")
    else:
        click.echo(f"\nFiled {len(filed)} issue(s), skipped {len(skipped)}")
        for bid, url in filed:
            click.echo(f"  {bid}: {url}")


# --- generate-summary ---

_NEGATIVE_KEYWORDS = re.compile(
    r"\b(bug|gap|vulnerability|missing|disabled|no tests|insecure|injection|fragile|risk|"
    r"broken|unsafe|unvalidated|unprotected|dormant|dead code|untested|hardcoded|leak)\b",
    re.IGNORECASE,
)

_POSITIVE_CONTEXT = re.compile(
    r"\b(is safe|is crash.safe|is secure|is protected|is validated|is tested|"
    r"has no gaps|no.gaps|are safe|are secure|are validated|are protected|"
    r"safely|gapless|sustainable|preserving|prevents|ensuring|"
    r"outlist injection|dependency injection|fault injection)\b",
    re.IGNORECASE,
)

_CRITICAL_KEYWORDS = re.compile(
    r"\b(security|injection|authentication|data loss|production|credential|vulnerability|"
    r"authorization|privilege|encryption|secret|password|token|remote code|command injection|"
    r"sql injection|xss|csrf)\b",
    re.IGNORECASE,
)


def _find_gated_out_beliefs(nodes: dict) -> list[dict]:
    """Find gated OUT beliefs and their active blockers."""
    results = []
    for nid, node in nodes.items():
        if node.get("truth_value") != "OUT":
            continue
        if node.get("metadata", {}).get("superseded_by"):
            continue
        for j in node.get("justifications", []):
            if not j.get("outlist"):
                continue
            active_blockers = [
                oid for oid in j["outlist"]
                if oid in nodes and nodes[oid].get("truth_value") == "IN"
            ]
            if active_blockers:
                results.append({
                    "id": nid,
                    "text": node.get("text", ""),
                    "blockers": [
                        {"id": bid, "text": nodes[bid].get("text", "")}
                        for bid in active_blockers
                    ],
                })
                break
    return results


def _find_negative_in_beliefs(nodes: dict) -> list[dict]:
    """Find IN beliefs with negative-signal keywords, excluding positive assertions."""
    results = []
    for nid, node in nodes.items():
        if node.get("truth_value") != "IN":
            continue
        text = node.get("text", "")
        if _NEGATIVE_KEYWORDS.search(text) and not _POSITIVE_CONTEXT.search(text):
            results.append({"id": nid, "text": text})
    return results


def _format_gated_section(beliefs: list[dict]) -> str:
    if not beliefs:
        return "_None_\n"
    lines = []
    for b in beliefs:
        lines.append(f"- **{b['id']}**: {b['text']}")
        for blocker in b["blockers"]:
            lines.append(f"  - Blocked by: `{blocker['id']}` — {blocker['text']}")
    return "\n".join(lines) + "\n"


def _format_belief_list(beliefs: list[dict]) -> str:
    if not beliefs:
        return "_None_\n"
    lines = []
    for b in beliefs:
        lines.append(f"- **{b['id']}**: {b['text']}")
    return "\n".join(lines) + "\n"


@cli.command("generate-summary")
@click.option("--snapshot-ids", multiple=True, hidden=True,
              help="Pre-run node IDs (passed by update command)")
@click.pass_context
def generate_summary(ctx, snapshot_ids):
    """Generate a morning summary entry of belief state.

    Highlights new gated OUT beliefs, new negative IN beliefs,
    and critical issues regardless of age.
    """
    network = _load_network()
    nodes = network.get("nodes", {})
    if not nodes:
        click.echo("No beliefs found. Run explorations first.", err=True)
        sys.exit(1)

    pre_run_ids = set(snapshot_ids) if snapshot_ids else set()

    # All gated OUT beliefs
    all_gated = _find_gated_out_beliefs(nodes)

    # All negative IN beliefs
    all_negative = _find_negative_in_beliefs(nodes)

    # Split into new vs existing
    if pre_run_ids:
        new_gated = [b for b in all_gated if b["id"] not in pre_run_ids]
        new_negative = [b for b in all_negative if b["id"] not in pre_run_ids]
    else:
        new_gated = all_gated
        new_negative = all_negative

    # Critical watch list — only problems, not positive assertions about safety
    critical_gated = [b for b in all_gated if _CRITICAL_KEYWORDS.search(b["text"])
                      or any(_CRITICAL_KEYWORDS.search(bl["text"]) for bl in b["blockers"])]
    critical_negative = [b for b in all_negative if _CRITICAL_KEYWORDS.search(b["text"])]

    # Statistics
    total_in = sum(1 for n in nodes.values() if n.get("truth_value") == "IN")
    total_out = sum(1 for n in nodes.values() if n.get("truth_value") == "OUT")
    total_derived = sum(1 for n in nodes.values()
                        if n.get("justifications") and len(n["justifications"]) > 0)

    # Build summary
    content = f"## New Gated OUT Beliefs\n\n{_format_gated_section(new_gated)}"
    content += f"\n## New Negative IN Beliefs\n\n{_format_belief_list(new_negative)}"
    content += f"\n## Critical Watch List\n\n"

    if critical_gated or critical_negative:
        if critical_gated:
            content += f"### Gated (blocked)\n\n{_format_gated_section(critical_gated)}\n"
        if critical_negative:
            content += f"### Active Issues\n\n{_format_belief_list(critical_negative)}\n"
    else:
        content += "_No critical issues detected._\n"

    content += f"\n## Statistics\n\n"
    content += f"- **Total beliefs:** {len(nodes)}\n"
    content += f"- **IN:** {total_in}\n"
    content += f"- **OUT:** {total_out}\n"
    content += f"- **Derived:** {total_derived}\n"
    content += f"- **Gated OUT (all):** {len(all_gated)}\n"
    content += f"- **Negative IN (all):** {len(all_negative)}\n"
    if pre_run_ids:
        content += f"- **New beliefs this run:** {len(nodes) - len(pre_run_ids)}\n"
        content += f"- **New gated OUT:** {len(new_gated)}\n"
        content += f"- **New negative IN:** {len(new_negative)}\n"

    _create_entry("update", "Update Summary", content)
    click.echo(f"\nSummary: {len(new_gated)} new gated OUT, {len(new_negative)} new negative IN, "
               f"{len(critical_gated) + len(critical_negative)} critical", err=True)


# --- update ---


@cli.command("update")
@click.option("--since", default=None,
              help="Walk commits since date (e.g., 2026-03-01, '1 week ago')")
@click.option("--since-commit", default=None,
              help="Walk commits since a specific commit SHA")
@click.option("--since-last", is_flag=True, default=False,
              help="Walk commits since last diff checkpoint")
@click.option("--file-issues", "do_file_issues", is_flag=True, default=False,
              help="Also file GitHub/GitLab issues for active blockers")
@click.pass_context
def update(ctx, since, since_commit, since_last, do_file_issues):
    """Automated update pipeline: walk commits, extract beliefs, derive, summarize.

    Runs the full pipeline in one command:
      1. walk-commits (explore changed files)
      2. propose-beliefs --auto (extract and accept beliefs)
      3. derive --exhaust (compute all logical consequences)
      4. generate-summary (morning report entry)

    Example:
        code-expert update --since-last
        code-expert update --since "1 week ago"
        code-expert update --since-last --file-issues
    """
    from .caffeinate import hold as _caffeinate
    _caffeinate()

    project_dir = _get_project_dir(ctx)
    errors = []
    started = datetime.now().isoformat(timespec="seconds")

    # Snapshot current node IDs before any changes
    try:
        network = _load_network()
        pre_run_ids = set(network.get("nodes", {}).keys())
    except Exception:
        pre_run_ids = set()

    # Step 1: walk-commits
    click.echo("\n=== Step 1: Walk commits ===\n", err=True)
    try:
        ctx.invoke(walk_commits, since=since, since_commit=since_commit,
                   since_last=since_last)
    except SystemExit as e:
        if e.code and e.code != 0:
            errors.append(f"walk-commits exited with code {e.code}")
            click.echo(f"WARN: walk-commits failed (exit {e.code}), continuing...", err=True)
    except Exception as e:
        errors.append(f"walk-commits: {e}")
        click.echo(f"WARN: walk-commits failed: {e}, continuing...", err=True)

    # Step 2: propose-beliefs --auto
    click.echo("\n=== Step 2: Propose and accept beliefs ===\n", err=True)
    try:
        ctx.invoke(propose_beliefs, auto_accept=True)
    except SystemExit as e:
        if e.code and e.code != 0:
            errors.append(f"propose-beliefs exited with code {e.code}")
            click.echo(f"WARN: propose-beliefs failed (exit {e.code}), continuing...", err=True)
    except Exception as e:
        errors.append(f"propose-beliefs: {e}")
        click.echo(f"WARN: propose-beliefs failed: {e}, continuing...", err=True)

    # Step 3: derive --exhaust
    click.echo("\n=== Step 3: Derive (exhaust) ===\n", err=True)
    try:
        ctx.invoke(derive, exhaust=True)
    except SystemExit as e:
        if e.code and e.code != 0:
            errors.append(f"derive exited with code {e.code}")
            click.echo(f"WARN: derive failed (exit {e.code}), continuing...", err=True)
    except Exception as e:
        errors.append(f"derive: {e}")
        click.echo(f"WARN: derive failed: {e}, continuing...", err=True)

    # Step 4: generate-summary
    click.echo("\n=== Step 4: Generate summary ===\n", err=True)
    try:
        ctx.invoke(generate_summary, snapshot_ids=tuple(pre_run_ids))
    except SystemExit as e:
        if e.code and e.code != 0:
            errors.append(f"generate-summary exited with code {e.code}")
    except Exception as e:
        errors.append(f"generate-summary: {e}")
        click.echo(f"WARN: generate-summary failed: {e}", err=True)

    # Step 5 (opt-in): file-issues
    if do_file_issues:
        click.echo("\n=== Step 5: File issues ===\n", err=True)
        try:
            ctx.invoke(file_issues)
        except SystemExit as e:
            if e.code and e.code != 0:
                errors.append(f"file-issues exited with code {e.code}")
        except Exception as e:
            errors.append(f"file-issues: {e}")
            click.echo(f"WARN: file-issues failed: {e}", err=True)

    # Save update checkpoint
    try:
        post_network = _load_network()
        post_run_ids = set(post_network.get("nodes", {}).keys())
    except Exception:
        post_run_ids = pre_run_ids

    checkpoint = {
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "beliefs_before": len(pre_run_ids),
        "beliefs_after": len(post_run_ids),
        "beliefs_added": len(post_run_ids - pre_run_ids),
        "errors": errors,
    }
    if project_dir:
        os.makedirs(project_dir, exist_ok=True)
        checkpoint_path = os.path.join(project_dir, "last-update.json")
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        click.echo(f"Update checkpoint saved to {checkpoint_path}", err=True)

    # Final report
    click.echo("\n=== Update complete ===\n", err=True)
    if errors:
        click.echo(f"Completed with {len(errors)} warning(s):", err=True)
        for err in errors:
            click.echo(f"  - {err}", err=True)
    else:
        click.echo("All steps completed successfully.", err=True)


# --- install-skill ---


@cli.command("install-skill")
@click.option("--skill-dir", type=click.Path(), default=None,
              help="Target directory (default: .claude/skills/code-expert)")
def install_skill(skill_dir):
    """Install the code-expert skill file for Claude Code."""
    if skill_dir:
        target_dir = Path(skill_dir)
    else:
        target_dir = Path.cwd() / ".claude" / "skills" / "code-expert"

    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "SKILL.md"

    skill_path = Path(__file__).parent / "data" / "SKILL.md"
    if skill_path.exists():
        target_file.write_text(skill_path.read_text())
    else:
        click.echo("WARN: Bundled SKILL.md not found", err=True)
        return

    click.echo(f"Installed skill to {target_file}")


if __name__ == "__main__":
    cli()
