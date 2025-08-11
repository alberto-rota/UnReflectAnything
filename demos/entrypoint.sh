#!/bin/bash

# Get demo file from environment variable, default to green
DEMO_FILE=${DEMO_FILE:-green}

# Use fixed internal ports (container will map them dynamically)
WS_PORT=8000
WEB_PORT=8001

echo "=================================="
echo "Starting Rerun viewer for: ${DEMO_FILE}.rrd"
echo "Checking port mappings..."

# Get the actual host ports from the container's perspective
HOST_WS_PORT=$(python3 -c "
import socket
import os
# The host ports will be available as environment variables or we can detect them
# For now, we'll use the mapped ports that docker-compose assigns
print('${WS_PORT_HOST:-8000}')
")

HOST_WEB_PORT=$(python3 -c "
import socket
import os
print('${WEB_PORT_HOST:-8001}')
")

# Construct the URL using host ports
URL="http://localhost:${HOST_WEB_PORT}/?url=rerun%2Bws://localhost:${HOST_WS_PORT}"

echo "Internal WebSocket Port: ${WS_PORT}"
echo "Internal Web Port: ${WEB_PORT}"
echo "Host WebSocket Port: ${HOST_WS_PORT}"
echo "Host Web Port: ${HOST_WEB_PORT}"
echo ""
echo "Click here to view: ${URL}"
echo "=================================="

# Start rerun with fixed internal ports
exec python -m rerun --web-viewer "/demos/${DEMO_FILE}.rrd" --port "${WS_PORT}" --web-viewer-port "${WEB_PORT}" --bind 0.0.0.0