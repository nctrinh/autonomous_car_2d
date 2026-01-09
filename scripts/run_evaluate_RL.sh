#!/bin/bash
set -e

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[1]}" )" && pwd )"
echo "Project root: $PROJECT_ROOT"

# Activate virtual environment if exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

CONFIG_PATH="$PROJECT_ROOT/config/RL_config.yaml"

MODEL_PATH="$PROJECT_ROOT/trained_models/ppo/ppo_final.zip"

ALGO="ppo"

EPISODES=10

VISUALIZE_FLAG="--visualize"

PYTHON_SCRIPT="$PROJECT_ROOT/scripts/run_evaluate_RL.py"

echo "Starting evaluation with model: $MODEL_PATH"

python3 "$PYTHON_SCRIPT" \
    --config "$CONFIG_PATH" \
    --model "$MODEL_PATH" \
    --algorithm "$ALGO" \
    --episodes "$EPISODES" \
    $VISUALIZE_FLAG \
    "$@"