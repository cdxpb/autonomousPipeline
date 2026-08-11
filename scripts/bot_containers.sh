#!/bin/bash
# Audit/stop Docker containers on the Duckiebot to free up RAM.
#
# Usage:
#   ./bot_containers.sh                                  # list only (default)
#   ./bot_containers.sh --stop <name-or-id>
#   ./bot_containers.sh --stop-all-except core1,core2
#   ./bot_containers.sh --prune                          # remove stopped containers + dangling images

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

echo "=== Duckiebot container audit ($ROBOT) ==="

if [ "$MODE" == "list" ]; then
    echo
    echo "--- Running containers ---"
    ssh "$ROBOT" "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'"

    echo
    echo "--- Memory usage ---"
    ssh "$ROBOT" "docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}'"

    echo
    echo "--- Host memory ---"
    ssh "$ROBOT" "free -h"

    echo
    echo "Keep: duckiebot-interface, ros, roscore, portainer, dt-files-api, watchtower"
    echo "Usually safe to stop: leftover dts-run-* sessions, old autonomousPipeline containers."
    echo "  ./bot_containers.sh --stop <name>"
    echo "  ./bot_containers.sh --stop-all-except X,Y"

elif [ "$MODE" == "stop" ]; then
    if [ -z "$TARGET" ]; then
        echo "ERROR: --stop requires a container name or ID"; exit 1
    fi
    read -p "Stop $TARGET? [y/N] " CONFIRM
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
    read -p "Stop everything EXCEPT $KEEP_LIST? [y/N] " CONFIRM
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
    echo "Removes STOPPED containers and dangling images/build cache (frees disk, not RAM)."
    read -p "Confirm? [y/N] " CONFIRM
    if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
        ssh "$ROBOT" "docker container prune -f && docker image prune -f && docker builder prune -f"
    else
        echo "Aborted."
    fi
fi
