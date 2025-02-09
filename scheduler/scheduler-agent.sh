#!/bin/bash

echo "================ Starting Prefect Scheduler ================="

# Ensure script is run from project root
PROJECT_ROOT=$(pwd)
echo "Project root set to: $PROJECT_ROOT"

# Define Prefect profile name
PREFECT_PROFILE="stockgpt"

# Set Prefect API URL for local server
# These should also be set in the profile
export PREFECT_API_URL="http://127.0.0.1:4201/api"

# Ensure logs directory exists
mkdir -p logs

# Function to check if Prefect server is running
is_prefect_server_running() {
    curl -s -o /dev/null -w "%{http_code}" "$PREFECT_API_URL/health"
}

### **🔹 Step 1: Set Up Prefect Profile if Missing**
if ! uv run prefect profile ls | grep -q "$PREFECT_PROFILE"; then
    echo "⚠️  Prefect profile '$PREFECT_PROFILE' not found. Creating it..."
    uv run prefect profile create "$PREFECT_PROFILE"
fi

# Ensure the correct profile is active
echo "🔄 Switching to Prefect profile '$PREFECT_PROFILE'..."
uv run prefect profile use "$PREFECT_PROFILE"

# Verify Prefect connection
uv run prefect config view

# Define work pool name
WORK_POOL_NAME="default-agent-pool"

### **🔹 Step 2: Start Prefect Server if Not Running**
if [[ $(is_prefect_server_running) -ne 200 ]]; then
    echo "🚀 Starting Prefect server..."
    uv run prefect server start --host 127.0.0.1 --port 4201 > logs/prefect_server.log 2>&1 &
    
    # Wait for Prefect server to start
    echo "⏳ Waiting for Prefect server to be ready..."
    until [[ $(is_prefect_server_running) -eq 200 ]]; do
        sleep 3
    done
    echo "✅ Prefect server is ready."
else
    echo "✅ Prefect server is already running."
fi

### **🔹 Step 3: Check/Create Work Pool**
if ! uv run prefect work-pool ls | grep -q "$WORK_POOL_NAME"; then
    echo "⚠️  Work pool '$WORK_POOL_NAME' not found. Creating it..."
    uv run prefect work-pool create --type process "$WORK_POOL_NAME"
else
    echo "✅ Work pool '$WORK_POOL_NAME' already exists."
fi

### **🔹 Step 4: Start Prefect Worker**
echo "🚀 Starting Prefect worker..."
uv run prefect worker start --pool "$WORK_POOL_NAME" > logs/prefect_worker.log 2>&1 &

echo "================ ✅ Prefect Agent and UI Started Successfully ================="
