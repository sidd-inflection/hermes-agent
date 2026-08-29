import os, subprocess, sys, tempfile
from pathlib import Path

SNIPPET = """
import os, sys
os.environ["GRID_CANARY"] = "set_by_host"
import run_agent  # noqa
print("CANARY=" + os.environ["GRID_CANARY"])
"""

def _run(embedded: bool):
    home = tempfile.mkdtemp()
    Path(home, ".env").write_text("GRID_CANARY=clobbered\n")
    env = {**os.environ, "HERMES_HOME": home}
    if embedded:
        env["HERMES_EMBEDDED"] = "1"
    else:
        env.pop("HERMES_EMBEDDED", None)
    out = subprocess.run([sys.executable, "-c", SNIPPET], env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout, home

def test_embedded_import_does_not_clobber_env():
    stdout, _ = _run(embedded=True)
    assert "CANARY=set_by_host" in stdout

def test_default_behavior_unchanged():
    stdout, _ = _run(embedded=False)
    assert "CANARY=clobbered" in stdout  # documents today's behavior

_INJECTION_SNIPPET = """
import os
import run_agent  # noqa
import hermes_cli.config as cfg
print("INJECTED=" + str(cfg._profile_env_vars_injected))
"""

def test_embedded_import_skips_profile_env_injection():
    """The actual contract this task guards: under HERMES_EMBEDDED,
    importing run_agent must not eagerly run _inject_profile_env_vars()
    (hermes_cli/config.py:6132), which populates OPTIONAL_ENV_VARS from
    provider profiles as an import-time side effect.
    """
    home = tempfile.mkdtemp()
    env = {**os.environ, "HERMES_HOME": home, "HERMES_EMBEDDED": "1"}
    out = subprocess.run([sys.executable, "-c", _INJECTION_SNIPPET], env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "INJECTED=False" in out.stdout

def test_embedded_import_still_creates_hermes_home_skeleton():
    """KNOWN RESIDUAL, not a bug: HERMES_EMBEDDED does not stop HERMES_HOME's
    directory skeleton from being created at import time.

    ensure_hermes_home() (hermes_cli/config.py:943) is still reached, via
    load_config(), from module-scope constants scattered across tools/ and
    agent/ that are unrelated to the two guarded sites — e.g.:
      - tools/vision_tools.py:92  (_VISION_DOWNLOAD_TIMEOUT)
      - tools/vision_tools.py:181 (_VISION_CPU_WORKERS)
      - agent/model_metadata.py:769 (_URL_TO_PROVIDER, via provider discovery
        -> hermes_cli/plugins.py's _get_disabled_plugins/_get_enabled_plugins)

    "Precompute a config-derived module constant at import time" is a
    repo-wide idiom, not a finite set of call sites, so closing this fully
    would mean guarding an unbounded and growing list of files — out of
    scope for the HERMES_EMBEDDED guard. Consequence for embedders:
    HERMES_HOME must already exist and be writable before `import run_agent`
    (Grid's start.sh does `mkdir -p "$HERMES_HOME"` before the server
    starts — see task B2).

    This test pins the residual so a change that "fixes" it by accident
    (e.g. gating ensure_hermes_home() itself, which would also suppress
    directory creation and SOUL.md seeding for real, later, non-embedded
    use) gets caught instead of silently changing the contract.
    """
    _, home = _run(embedded=True)
    created = {p.name for p in Path(home).iterdir()} - {".env"}
    assert "memories" in created and "pairing" in created

_AGENT_CONSTRUCTION_SNIPPET = """
import run_agent
agent = run_agent.AIAgent(model="dummy", base_url="http://localhost:1", api_key="x",
                           max_tokens=16, skip_memory=True, skip_context_files=True,
                           quiet_mode=True)
print("AGENT_OK")
"""

def test_embedded_agent_construction_succeeds():
    """Regression for A12: agent_init.py's agent construction path calls
    setup_logging(hermes_home=run_agent._hermes_home) unconditionally
    (agent/agent_init.py ~line 1097), with no HERMES_EMBEDDED check of its
    own. A2's guard moved the `_hermes_home = get_hermes_home()` assignment
    inside `if not _is_embedded():`, so under HERMES_EMBEDDED=1 the
    attribute was never bound and every AIAgent(...) construction raised
    AttributeError: module 'run_agent' has no attribute '_hermes_home'.

    Before this task's fix, this test fails RED with exactly that
    AttributeError surfaced through the subprocess's stderr/returncode.
    """
    home = tempfile.mkdtemp()
    env = {**os.environ, "HERMES_HOME": home, "HERMES_EMBEDDED": "1"}
    out = subprocess.run([sys.executable, "-c", _AGENT_CONSTRUCTION_SNIPPET], env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "AGENT_OK" in out.stdout
