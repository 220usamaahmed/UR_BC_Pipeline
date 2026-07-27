#!/bin/bash

set -e

# Default to base compose file(s)
COMPOSE_FILES=("docker-compose.yml")
MODE="base (mock)"
COMMAND=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --real)
            COMPOSE_FILES=("docker-compose.yml" "docker-compose.real.yml")
            MODE="real"
            shift
            ;;
        --base|--mock)
            COMPOSE_FILES=("docker-compose.yml")
            MODE="base (mock)"
            shift
            ;;
        exec)
            COMMAND="exec"
            shift
            # Remaining args are for docker compose exec
            break
            ;;
        --help)
            echo "Usage: ./start-docker.sh [OPTIONS] [COMMAND]"
            echo ""
            echo "OPTIONS:"
            echo "  --base, --mock    Use base/mock docker-compose.yml (default)"
            echo "  --real            Use real hardware (layers docker-compose.real.yml)"
            echo "  --help            Show this message"
            echo ""
            echo "COMMANDS:"
            echo "  (none)            Start the container with 'docker compose up --build'"
            echo "  exec [ARGS]       Execute a command in the running container"
            echo "                    Example: ./start-docker.sh exec bash"
            echo "                    Example: ./start-docker.sh --real exec /bin/bash"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Check if external drive exists
EXTERNAL_DRIVE="/media/siddiquieu1/AHMED/new-ur3e-trajectories"
if [ -d "$EXTERNAL_DRIVE" ]; then
    export EXTERNAL_DRIVE_PATH="$EXTERNAL_DRIVE"
    echo "✓ External drive found at $EXTERNAL_DRIVE"
else
    echo "⚠ External drive not found at $EXTERNAL_DRIVE"
    echo "  Starting without the external drive mount"

    # An empty EXTERNAL_DRIVE_PATH produces an invalid empty volume entry in
    # Docker Compose. Use temporary copies with that optional mount removed.
    TEMP_COMPOSE_DIR="$(mktemp -d)"
    trap 'rm -rf "$TEMP_COMPOSE_DIR"' EXIT

    FILTERED_COMPOSE_FILES=()
    for file in "${COMPOSE_FILES[@]}"; do
        filtered_file="$TEMP_COMPOSE_DIR/$(basename "$file")"
        sed '/EXTERNAL_DRIVE_PATH/d' "$file" > "$filtered_file"
        FILTERED_COMPOSE_FILES+=("$filtered_file")
    done
    COMPOSE_FILES=("${FILTERED_COMPOSE_FILES[@]}")
fi

# Build docker compose file arguments
COMPOSE_ARGS=()
for file in "${COMPOSE_FILES[@]}"; do
    COMPOSE_ARGS+=("-f" "$file")
done

echo "📦 Docker mode: $MODE"
echo ""

if [ "$COMMAND" = "exec" ]; then
    # Execute a command in running container
    docker compose --project-directory "$PWD" "${COMPOSE_ARGS[@]}" exec -it ur_sim "$@"
else
    # Start the container
    docker compose --project-directory "$PWD" "${COMPOSE_ARGS[@]}" up --build
fi
