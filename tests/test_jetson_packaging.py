import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
KIOSK_LAUNCHER = ROOT / "deploy/jetson/launch-kiosk.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}")
    path.chmod(0o755)


def _write_kiosk_system_fakes(
    fake_bin: Path,
    *,
    locked_hint: str,
    idle_hint: str,
    monitor_state: str,
    nongraphical_session_id: str | None = None,
) -> None:
    nongraphical_case = ""
    if nongraphical_session_id:
        nongraphical_case = f"""
    if [ "$2" = "{nongraphical_session_id}" ]; then
      case "$property" in
        User) printf '{os.getuid()}\\n' ;;
        Type) printf 'tty\\n' ;;
        Remote) printf 'yes\\n' ;;
        Active) printf 'yes\\n' ;;
        State) printf 'active\\n' ;;
        LockedHint|IdleHint) printf 'no\\n' ;;
        *) exit 1 ;;
      esac
      exit 0
    fi
"""
    loginctl = f"""
property=
for argument in "$@"; do
  case "$argument" in
    --property=*) property="${{argument#--property=}}" ;;
  esac
done
case "$1" in
  list-sessions) printf '2 {os.getuid()} tester seat0 tty2 active yes 1h\\n' ;;
  show-session)
{nongraphical_case}
    case "$property" in
      User) printf '{os.getuid()}\\n' ;;
      Type) printf 'x11\\n' ;;
      Remote) printf 'no\\n' ;;
      Active) printf 'yes\\n' ;;
      State) printf 'active\\n' ;;
      LockedHint) printf '{locked_hint}\\n' ;;
      IdleHint) printf '{idle_hint}\\n' ;;
      *) exit 1 ;;
    esac
    ;;
  *) exit 1 ;;
esac
"""
    _write_executable(fake_bin / "loginctl", loginctl)
    _write_executable(
        fake_bin / "xset",
        f"printf 'DPMS is Enabled\\n  Monitor is {monitor_state}\\n'\n",
    )
    _write_executable(
        fake_bin / "systemd-inhibit",
        """printf 'inhibitor=sleep\n'
while [ "$#" -gt 0 ]; do
  case "$1" in
    --what=*|--who=*|--why=*|--mode=*) shift ;;
    *) exec "$@" ;;
  esac
done
exit 1
""",
    )


def _run_fake_kiosk(
    tmp_path: Path,
    *browser_names: str,
    overrides: dict[str, str] | None = None,
    locked_hint: str = "no",
    idle_hint: str = "no",
    monitor_state: str = "On",
    xdg_session_id: str = "2",
    nongraphical_session_id: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    _write_executable(fake_bin / "curl", "exit 0\n")
    _write_kiosk_system_fakes(
        fake_bin,
        locked_hint=locked_hint,
        idle_hint=idle_hint,
        monitor_state=monitor_state,
        nongraphical_session_id=nongraphical_session_id,
    )
    for browser_name in browser_names:
        _write_executable(
            fake_bin / browser_name,
            f"printf 'browser={browser_name}\\n'\nprintf 'arg=%s\\n' \"$@\"\n",
        )

    environment = os.environ.copy()
    environment.pop("BOOKFORGE_BROWSER_BIN", None)
    environment.pop("BOOKFORGE_CHROMIUM_BIN", None)
    environment.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "HOME": str(tmp_path / "home"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
            "XDG_SESSION_ID": xdg_session_id,
            "DISPLAY": ":1",
            "XAUTHORITY": str(tmp_path / "Xauthority"),
            "BOOKFORGE_KIOSK_URL": "http://127.0.0.1:18081/projector?live=1",
            "BOOKFORGE_READY_URL": "http://127.0.0.1:18081/readyz",
            "BOOKFORGE_KIOSK_STARTUP_TIMEOUT": "1",
        }
    )
    (tmp_path / "Xauthority").write_text("test authority")
    browser_overrides = dict(overrides or {})
    if (
        "BOOKFORGE_BROWSER_BIN" not in browser_overrides
        and "BOOKFORGE_CHROMIUM_BIN" not in browser_overrides
        and "chromium" not in browser_names
        and "chromium-browser" not in browser_names
        and "firefox" in browser_names
    ):
        # Hosted Linux runners can have a real Chromium in /usr/bin. Keep
        # Firefox-only tests hermetic instead of launching a host browser.
        browser_overrides["BOOKFORGE_BROWSER_BIN"] = str(fake_bin / "firefox")
    environment.update(browser_overrides)
    return subprocess.run(
        [str(KIOSK_LAUNCHER)],
        check=check,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_kiosk_preflight_refuses_locked_idle_and_dark_sessions(tmp_path: Path) -> None:
    cases = (
        ("locked", {"locked_hint": "yes"}, "is locked"),
        ("idle", {"idle_hint": "yes"}, "is idle"),
        ("display-off", {"monitor_state": "Off"}, "display is off"),
    )
    for name, state, expected_error in cases:
        result = _run_fake_kiosk(
            tmp_path / name,
            "firefox",
            check=False,
            **state,
        )

        assert result.returncode == 78
        assert expected_error in result.stderr
        assert "browser=firefox" not in result.stdout


def test_kiosk_ignores_inherited_ssh_session_and_finds_local_x11(tmp_path: Path) -> None:
    result = _run_fake_kiosk(
        tmp_path,
        "firefox",
        xdg_session_id="45",
        nongraphical_session_id="45",
    )

    assert result.returncode == 0
    assert "browser=firefox" in result.stdout


def test_unattended_kiosk_can_explicitly_allow_idle_but_unlocked_x11(
    tmp_path: Path,
) -> None:
    result = _run_fake_kiosk(
        tmp_path,
        "firefox",
        idle_hint="yes",
        overrides={"BOOKFORGE_KIOSK_ALLOW_IDLE": "true"},
    )

    assert result.returncode == 0
    assert "browser=firefox" in result.stdout
    assert "allow_idle=true" in result.stdout


def test_preferred_firefox_override_uses_only_supported_kiosk_flags(tmp_path: Path) -> None:
    firefox = tmp_path / "bin" / "firefox"
    legacy = tmp_path / "bin" / "legacy-chromium-wrapper"
    result = _run_fake_kiosk(
        tmp_path,
        "firefox",
        "legacy-chromium-wrapper",
        overrides={
            "BOOKFORGE_BROWSER_BIN": str(firefox),
            "BOOKFORGE_CHROMIUM_BIN": str(legacy),
        },
    )

    assert "browser=firefox" in result.stdout
    assert "inhibitor=sleep" in result.stdout
    assert "arg=--kiosk" in result.stdout
    assert "arg=--profile" in result.stdout
    assert f"arg={tmp_path / 'state' / 'bookforge' / 'firefox'}" in result.stdout
    assert "arg=--new-instance" in result.stdout
    assert "arg=--private-window" in result.stdout
    assert "arg=http://127.0.0.1:18081/projector?live=1" in result.stdout
    assert "--app=" not in result.stdout
    assert "--disable-background-networking" not in result.stdout
    assert "--user-data-dir" not in result.stdout


def test_legacy_chromium_override_retains_hardened_arguments(tmp_path: Path) -> None:
    legacy = tmp_path / "bin" / "legacy-chromium-wrapper"
    result = _run_fake_kiosk(
        tmp_path,
        "legacy-chromium-wrapper",
        overrides={"BOOKFORGE_CHROMIUM_BIN": str(legacy)},
    )

    assert "browser=legacy-chromium-wrapper" in result.stdout
    assert "inhibitor=sleep" in result.stdout
    assert "arg=--kiosk" in result.stdout
    assert "arg=--app=http://127.0.0.1:18081/projector?live=1" in result.stdout
    assert "arg=--disable-background-networking" in result.stdout
    assert "arg=--disable-component-update" in result.stdout
    assert "arg=--disable-sync" in result.stdout
    assert "arg=--user-data-dir=" in result.stdout


def test_browser_autodetection_prefers_chromium_and_accepts_firefox(tmp_path: Path) -> None:
    preferred = _run_fake_kiosk(tmp_path / "preferred", "chromium", "firefox")
    fallback = _run_fake_kiosk(tmp_path / "fallback", "firefox")

    assert "browser=chromium" in preferred.stdout
    assert "browser=firefox" not in preferred.stdout
    assert "browser=firefox" in fallback.stdout
    assert "arg=--private-window" in fallback.stdout


def test_firefox_kiosk_profile_disables_first_run_and_telemetry(tmp_path: Path) -> None:
    _run_fake_kiosk(tmp_path, "firefox")
    profile = tmp_path / "state" / "bookforge" / "firefox"
    preferences = (profile / "user.js").read_text()

    assert 'browser.aboutwelcome.enabled", false' in preferences
    assert 'browser.startup.homepage_override.mstone", "ignore"' in preferences
    assert 'browser.shell.checkDefaultBrowser", false' in preferences
    assert 'datareporting.policy.dataSubmissionEnabled", false' in preferences
    assert 'toolkit.telemetry.enabled", false' in preferences
    assert (profile / "user.js").stat().st_mode & 0o777 == 0o600
