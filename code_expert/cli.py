"""Command-line interface for code expert."""

import asyncio
import json
import os
import re
import subprocess
import sys
from datetime import date
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
    load_diff_checkpoint,
    save_diff_checkpoint,
)
from .llm import check_model_available, invoke, invoke_sync
from .observations import parse_observation_requests, run_observations
from .prompts import (
    PROPOSE_BELIEFS_CODE,
    build_diff_prompt,
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
@click.version_option(package_name="code-expert")
@click.option("--quiet", "-q", is_flag=True, default=False,
              help="Suppress explanation output to stdout")
@click.option("--repo", "-r", type=click.Path(exists=True, file_okay=False),
              default=None, help="Repository root (default: from config or cwd)")
@click.option("--model", "-m", default="claude", help="Model to use (default: claude)")
@click.pass_context
def cli(ctx, quiet, repo, model):
    """Build expert knowledge bases from codebases."""
    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet
    ctx.obj["model"] = model
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

    # Check prerequisites
    for tool in ["git", "beliefs", "entry"]:
        if not __import__("shutil").which(tool):
            click.echo(f"Error: {tool} not found on PATH", err=True)
            click.echo(f"Install with: uv tool install {tool}", err=True)
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

    # Init beliefs if needed
    if not Path("beliefs.md").exists():
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
        result = asyncio.run(invoke(prompt, model))
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
        result = asyncio.run(invoke(prompt, model))
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
        result = asyncio.run(invoke(prompt, model))
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
        result = asyncio.run(invoke(prompt, model))
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

    prompt = build_diff_prompt(
        diff_content=diff_content,
        commit_log=commit_log,
        changed_files_summary=changed_files or None,
    )

    click.echo(f"Running {model}...", err=True)
    try:
        result = asyncio.run(invoke(prompt, model))
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
@click.option("--pick", "pick_index", type=int, default=None,
              help="Pick a topic by index")
@click.pass_context
def explore(ctx, do_skip, pick_index):
    """Explore the next topic in the queue (or --skip / --pick N)."""
    from .caffeinate import hold as _caffeinate
    _caffeinate()
    project_dir = _get_project_dir(ctx)
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
        topic = pop_at(pick_index, project_dir)
    else:
        topic = pop_next(project_dir)

    if topic is None:
        click.echo("No pending topics. Run `code-expert scan` to discover topics.")
        return

    click.echo(f"Topic: [{topic.kind}] {topic.target}", err=True)
    click.echo(f"  {topic.title}", err=True)
    if topic.source:
        click.echo(f"  (from {topic.source})", err=True)
    click.echo(err=True)

    repo_path = _get_repo(ctx)
    abs_repo = os.path.abspath(repo_path)
    model = ctx.obj["model"]

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
        sys.exit(1)

    remaining = pending_count(project_dir)
    if remaining:
        click.echo(f"\n{remaining} topic(s) remaining. Run `code-expert explore` to continue.", err=True)
    else:
        click.echo("\nNo more topics. Exploration complete.", err=True)


def _run_file_topic(ctx, topic, model, repo_path):
    """Handle a file exploration topic."""
    file_path = topic.target
    abs_path = os.path.join(repo_path, file_path) if not os.path.isabs(file_path) else file_path

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
        result = asyncio.run(invoke(prompt, model))
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
        result = asyncio.run(invoke(prompt, model))
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
        result = asyncio.run(invoke(prompt, model))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    _create_entry("repo-overview", "Repo Overview", result)
    _enqueue_topics(result, source="repo-overview", project_dir=_get_project_dir(ctx))
    _report_beliefs(result)

    _emit(ctx, result)


def _run_diff_topic(ctx, topic, model, repo_path):
    """Handle a diff exploration topic."""
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
        result = asyncio.run(invoke(prompt, model))
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
        result = asyncio.run(invoke(prompt, model))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    safe_label = _sanitize_path_for_filename(topic.target)
    _create_entry(f"topic-{safe_label}", f"Topic: {topic.title}", result)
    _enqueue_topics(result, source=f"general:{topic.target}", project_dir=_get_project_dir(ctx))
    _report_beliefs(result)

    _emit(ctx, result)


# --- propose-beliefs ---


@cli.command("propose-beliefs")
@click.option("--batch-size", type=int, default=5, help="Entries per LLM batch (default: 5)")
@click.option("--output", default="proposed-beliefs.md", help="Output file")
@click.option("--model", "-m", default=None, help="Override model")
@click.option("--entry", "entry_paths", multiple=True, type=click.Path(exists=True),
              help="Process specific entry file(s) instead of all entries")
@click.option("--all", "process_all", is_flag=True,
              help="Re-process all entries (ignore processed tracking)")
@click.pass_context
def propose_beliefs(ctx, batch_size, output, model, entry_paths, process_all):
    """Extract candidate beliefs from entries for human review."""
    from .caffeinate import hold as _caffeinate
    _caffeinate()
    if model is None:
        model = ctx.obj["model"]

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
        click.echo("(install fastembed for semantic dedup: uv pip install 'code-expert[embeddings]')")

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
            result = invoke_sync(prompt, model=model, timeout=600)
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

    # Record processed entries
    _save_processed(processed_path, entries, processed)

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

    # Parse accepted beliefs — tolerate both ### [ACCEPT] and ### ACCEPT
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

    # Try batch mode first (single subprocess, single parse of beliefs.md)
    if _accept_batch(matches):
        return

    # Fall back to per-belief add
    click.echo("Falling back to per-belief add...")
    added = 0
    failed = 0
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
                else:
                    click.echo(f"  FAIL: {belief_id}: {stderr or result.stdout.strip()}")
                    failed += 1
        except FileNotFoundError:
            click.echo("ERROR: beliefs CLI not found. Install with: uv tool install beliefs")
            sys.exit(1)

    click.echo(f"\nAccepted {added} beliefs ({failed} failed)")


def _load_existing_beliefs(beliefs_path: Path) -> list[dict]:
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

    # Count beliefs
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
