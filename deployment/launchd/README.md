# launchd agents — BLOCKED while the project lives under ~/Desktop

These agents keep `hybrid_supervisor.sh` / `delisting_exit_supervisor.sh` alive
across sleep, logout and reboot. They are **not installed**, because they cannot
work from the project's current location.

## Why

macOS TCC treats `~/Desktop` as a privacy-protected location. launchd agents (and
cron) run without Full Disk Access, so they cannot even read a script there:

    /bin/bash: .../hybrid_supervisor.sh: Operation not permitted
    shell-init: error retrieving current directory: getcwd: cannot access parent
                directories: Operation not permitted

Verified 2026-08-31 — both agents exited 126 immediately. The same wall has been
silently killing this project's **cron jobs since mid-July**: `retrain_weekly.py`
(weekly) last wrote 2026-07-14 and `carry_paper_trader.py report` (daily) last
wrote 2026-07-15. Neither has run since, with no error surfaced anywhere.

## To enable

Move the project out of a protected location (`~/Desktop`, `~/Documents`,
`~/Downloads`) — e.g. to `~/Projects/octivault_trader` — then:

    # fix the paths inside the plists to the new location first
    cp deployment/launchd/*.plist ~/Library/LaunchAgents/
    launchctl load ~/Library/LaunchAgents/com.octivault.hybrid.plist
    launchctl load ~/Library/LaunchAgents/com.octivault.delisting.plist
    launchctl list | grep octivault      # status 0 = healthy

Granting Full Disk Access to `/bin/bash` would also work but is a far broader
security change; moving the project is the better trade.

## Design note

`KeepAlive` is `{SuccessfulExit: false}` — restart only on a NON-zero exit. This
relies on the supervisor's exit codes: **0** when the operator's stop flag is
present (stay down), **75** when killed with no stop flag (come back). The
supervisor also no longer clears the stop flag at startup, so an auto-restarter
cannot defeat the off switch.

## Until then

Supervision is manual:

    nohup ./hybrid_supervisor.sh > logs/hybrid_sup_nohup.out 2>&1 &
    nohup ./delisting_exit_supervisor.sh > logs/delisting_exit_sup_nohup.out 2>&1 &

which does NOT survive logout or a signal to the supervisor — the gap that left
the account unmanaged for ~8.7h on 2026-08-31.
