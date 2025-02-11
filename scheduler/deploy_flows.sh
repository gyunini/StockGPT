#!/bin/bash

echo "================ 🚀 Deploying Prefect Flows ================="

# Ensure script is executed from project root
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

# Set Prefect API URL for local server
export PREFECT_API_URL="http://127.0.0.1:4200/api"

# Ensure Prefect connection is valid
uv run prefect config view

# Define the directory containing flow definitions
FLOW_DIR="scheduler/flows"

# Loop through Python files in the flows directory
for flow_file in "$FLOW_DIR"/*.py; do
    if [[ -f "$flow_file" ]]; then
        # Extract flow name (filename without extension)
        flow_name=$(basename "$flow_file" .py)
        flow_path="scheduler.flows.${flow_name}"

        echo "📌 Deploying flow: $flow_name..."

        # Use `prefect deploy` for Prefect 3.x
        uv run prefect deploy "$flow_path:$flow_name" --name "$flow_name-deployment"

        echo "✅ Successfully deployed: $flow_name"
    fi
done

echo "================ ✅ All Flows Deployed Successfully ================="
