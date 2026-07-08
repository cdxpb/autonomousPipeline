#!/bin/bash
# ============================================================
# bot_containers.sh – Audit and optionally stop Docker containers
#                      running on the Duckiebot to free up RAM.
#
# The Jetson Nano only has 4GB RAM total; leftover/duplicate
# containers from earlier test runs (old `dts devel run` sessions,
# stopped-but-not-removed segmentation servers, etc.) eat into
# that budget even when idle.
#
# Usage:
#   ./bot_containers.sh                      # list only (default, safe)
#   ./bot_containers.sh --list               # same as above
#   ./bot_containers.sh --stop <name-or-id>  # stop ONE container (asks to confirm)
#   ./bot_containers.sh --stop-all-except core1,core2   # stop everything NOT in this list
#   ./bot_containers.sh --prune              # remove stopped containers + dangling images/layers (safe, frees disk not RAM)
# ============================================================

set -e

ROBOT=duckie@duckiexp.local
MODE=list
TARGET=""
KEEP_LIST=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --list) MODE=list ;;
        --stop) MODE=stop; TARGET="$2"; shift ;;
        --stop-all-except) MODE=stop_all_except; KEEP_LIST="$2"; shift ;;
        --prune) MODE=prune ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

echo "======================================================="
echo " Duckiebot Docker Container Audit  ($ROBOT)"
echo "======================================================="

if [ "$MODE" == "list" ]; then
    echo
    echo "--- Running containers (name / image / status / ports) ---"
    ssh "$ROBOT" "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'"

    echo
    echo "--- Live memory usage per container (no-stream snapshot) ---"
    ssh "$ROBOT" "docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}'"

    echo
    echo "--- Host memory ---"
    ssh "$ROBOT" "free -h"

    echo
    echo "Typical Duckietown CORE containers you probably want to KEEP:"
    echo "  names containing: duckiebot-interface, ros, roscore, portainer, dt-files-api, watchtower"
    echo
    echo "Likely SAFE-TO-STOP candidates: leftover 'dts-run-*' dev sessions from earlier"
    echo "'dts devel run' testing, duplicate/old autonomousPipeline containers, unused VNC/desktop sessions."
    echo
    echo "Review the list above yourself before stopping anything -- container names vary by setup."
    echo "To stop one:            ./bot_containers.sh --stop <name>"
    echo "To stop all except X,Y: ./bot_containers.sh --stop-all-except X,Y"

elif [ "$MODE" == "stop" ]; then
    if [ -z "$TARGET" ]; then
        echo "ERROR: --stop requires a container name or ID"; exit 1
    fi
    echo "About to stop container: $TARGET"
    read -p "Confirm? [y/N] " CONFIRM
    if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
        ssh "$ROBOT" "docker stop '$TARGET'"
        echo "Stopped $TARGET."
    else
        echo "Aborted."
    fi

elif [ "$MODE" == "stop_all_except" ]; then
    if [ -z "$KEEP_LIST" ]; then
        echo "ERROR: --stop-all-except requires a comma-separated keep list"; exit 1
    fi
    echo "Containers currently running:"
    ssh "$ROBOT" "docker ps --format '{{.Names}}'"
    echo
    echo "Will stop everything EXCEPT: $KEEP_LIST"
    read -p "Confirm? [y/N] " CONFIRM
    if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
        ssh "$ROBOT" "KEEP='$KEEP_LIST'; for n in \$(docker ps --format '{{.Names}}'); do
            skip=0
            IFS=',' read -ra KEEPARR <<< \"\$KEEP\"
            for k in \"\${KEEPARR[@]}\"; do
                if [[ \"\$n\" == \"\$k\" ]]; then skip=1; fi
            done
            if [[ \$skip -eq 0 ]]; then echo \"Stopping \$n...\"; docker stop \"\$n\"; fi
        done"
    else
        echo "Aborted."
    fi

elif [ "$MODE" == "prune" ]; then
    echo "This removes STOPPED containers and dangling images/build cache (frees disk, not RAM)."
    read -p "Confirm? [y/N] " CONFIRM
    if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
        ssh "$ROBOT" "docker container prune -f && docker image prune -f && docker builder prune -f"
    else
        echo "Aborted."
    fi
fi
