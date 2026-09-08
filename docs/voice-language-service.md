# Optional local language service

The voice API can use a CPU-only spaCy service over an owner-only Unix socket.
It predicts syntax and proposes typed scene facts; original-source grounding and
privacy checks still apply. Learned facts and omissions require explicit review
and a matching confirmation digest before any image request. The bounded compiler
remains available when the socket setting is absent.

The Jetson environment is `/opt/bookforge/.venv-voice-language`, separate from the
API's `/opt/bookforge/.venv`. The [environment receipt](../benchmarks/voice-language-2026-09-07/jetson-environment.json)
pins installed wheels: spaCy 3.8.11, en_core_web_sm 3.8.0, Pydantic 2.13.4 and
Click 8.1.8. No GPU framework is required. One isolated CPU smoke check loaded the
model in 2.10 seconds and parsed synthetic text in 73 ms, peaking at 140 MB RSS;
this is not a service latency benchmark.

## Installation

Run on the Jetson as the Bookforge owner. The environment already exists; these
commands reproduce its direct dependencies on a fresh installation:

```sh
/usr/bin/python3.12 -m venv /opt/bookforge/.venv-voice-language
/opt/bookforge/.venv-voice-language/bin/python -m pip --isolated install \
  --only-binary=:all: --index-url https://pypi.org/simple \
  'spacy==3.8.11' 'pydantic==2.13.4' 'click==8.1.8'
/opt/bookforge/.venv-voice-language/bin/python -m pip --isolated install --no-deps \
  'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85'
/opt/bookforge/.venv-voice-language/bin/python -m pip check
```

Use a reviewed immutable source release under
`/opt/bookforge/voice-language-releases/<release>`, containing a `bookforge`
package with `__init__.py`, `voice_language.py`, `voice_dependencies.py`,
`domain.py`, `scene_facts.py`, `privacy_policy.py`, and `semantic_text.py`.
Record its source hashes and previous symlink target before switching
`/opt/bookforge/voice-language-current`. Do not copy the whole repository into the
older API environment. API compatibility patches are deployed separately with
before/after hashes and backups.

From the reviewed repository checkout on the Jetson, install the user unit:

```sh
install -d -m 700 ~/.config/systemd/user
install -m 644 deploy/jetson/bookforge-voice-language.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now bookforge-voice-language.service
systemctl --user is-active bookforge-voice-language.service
stat -c '%a %n' /run/user/1000/bookforge-voice-language/parser.sock
```

The unit loads one CPU pipeline from `voice-language-current`. Its runtime
directory is mode 0700 and socket mode 0600. SIGINT removes the owned socket;
systemd also manages runtime-directory cleanup. It does not open a TCP port or
log source descriptions.

After the reviewed API compatibility patch is installed, enable its socket
setting through the prepared systemd drop-in. The appliance readiness guard
rejects a checkout `/opt/bookforge/.env` file; do not create one or alter the
root-owned service environment. From the reviewed checkout on the Jetson:

```sh
sudo install -d -m 755 /etc/systemd/system/bookforge@operator.service.d
sudo install -m 644 deploy/jetson/bookforge-voice-language.conf \
  /etc/systemd/system/bookforge@operator.service.d/50-voice-language.conf
sudo systemctl daemon-reload
sudo -n /usr/local/sbin/bookforge-admin restart-api
```

The drop-in sets only `BOOKFORGE_REVIEWED_SCENE_PARSER_SOCKET` to
`/run/user/1000/bookforge-voice-language/parser.sock`. Installation and daemon
reload require an administrator password in an interactive terminal; the
restricted restart helper does not grant permission to install system files.
Record any existing drop-in before replacing it.

The administrator installed the drop-in on September 7. The isolated service,
API and kiosk are active. The [live checks](../benchmarks/voice-language-2026-09-07/live-smoke.json)
verified preparation, confirmation rejection, and one synthetic image through
the gateway. Merely preparing a draft does not authorize generation.

## Rollback

To disable this integration, remove only its dedicated drop-in (or restore its
recorded predecessor), reload systemd and restart the API. Then stop the optional
service:

```sh
sudo rm /etc/systemd/system/bookforge@operator.service.d/50-voice-language.conf
sudo systemctl daemon-reload
sudo -n /usr/local/sbin/bookforge-admin restart-api
systemctl --user disable --now bookforge-voice-language.service
```

Leave the appliance's other environment settings and readiness guard intact.

To roll back only its code, set `previous_voice_release` to the verified prior
release recorded during deployment, then switch the symlink and restart:

```sh
test -d "$previous_voice_release/bookforge"
ln -s "$previous_voice_release" /opt/bookforge/voice-language-current.next
mv -Tf /opt/bookforge/voice-language-current.next /opt/bookforge/voice-language-current
systemctl --user restart bookforge-voice-language.service
```

Keep releases and API backup receipts until rollback verification completes.
Disabling this service leaves existing artwork and the renderer configuration
unchanged; unsupported bounded descriptions continue to request editing.
