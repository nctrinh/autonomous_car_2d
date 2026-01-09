#!/bin/bash
set -e

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[1]}" )" && pwd )"
echo "Project root: $PROJECT_ROOT"

# Activate virtual environment if existed
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

DEFAULT_CONFIG="$PROJECT_ROOT/config/RL_config.yaml"

PYTHON_SCRIPT="$PROJECT_ROOT/scripts/run_train_RL.py"

echo "======================================================================"
echo "STARTING CURRICULUM TRAINING"
echo "Script: $PYTHON_SCRIPT"
echo "Config: $DEFAULT_CONFIG (Default)"
echo "======================================================================"

python3 "$PYTHON_SCRIPT" \
    --config "$DEFAULT_CONFIG" \
    "$@"