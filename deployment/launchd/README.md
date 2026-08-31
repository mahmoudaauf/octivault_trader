# launchd agents — INSTALLED and verified

Keeps `hybrid_supervisor.sh`, `delisting_exit_supervisor.sh` and
`spread_mm_supervisor.sh` alive across sleep, logout and reboot.

    launchctl list | grep octivault      # status 0 = healthy

## History: why the project moved

These could not work from the old location. macOS TCC protects `~/Desktop`, and
launchd (like cron) has no Full Disk Access there, so it could not even read the
scripts:

    /bin/bash: .../hybrid_supervisor.sh: Operation not permitted

Agents exited 126 immediately. The same wall had been silently killing the
project's **cron jobs since mid-July 2026** — `retrain_weekly.py` last wrote
2026-07-14, `carry_paper_trader.py report` last wrote 2026-07-15, ~59 missed runs
with no error surfaced anywhere.

Resolved 2026-08-31 by moving the repo to `~/Projects/octivault_trader`, which is
not TCC-protected. All three agents now report status 0.

## Design

`KeepAlive = {SuccessfulExit: false}` — restart only on a NON-zero exit. This
depends on the supervisors' exit codes:

| situation | exit | launchd |
|---|---|---|
| operator set the stop flag | 0 | leaves it down |
| killed by signal, no stop flag | 75 | restarts it |

The supervisors also no longer clear the stop flag at startup — otherwise an
auto-restarter would wipe the operator's off switch on every relaunch.

## Verified end-to-end (2026-08-31)

- **Resurrection**: killed the supervisor with no stop flag → exit 75, alert
  written to `logs/hybrid_alerts.log` + macOS notification, launchd relaunched it
  within ~5s and the daemon came back with it.
- **Off switch**: with the stop flag present → exit 0, stayed down past the
  throttle interval, flag not wiped. The operator still wins.

## Managing them

    launchctl unload ~/Library/LaunchAgents/com.octivault.<name>.plist   # disable
    launchctl load   ~/Library/LaunchAgents/com.octivault.<name>.plist   # enable
    touch logs/<name>_supervisor.stop                                    # stop, stays stopped

Plists here are the source of truth; copy to `~/Library/LaunchAgents/` after
editing. They embed absolute paths — regenerate them if the repo moves again.
