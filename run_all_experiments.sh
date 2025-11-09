#!/bin/bash
# Run all 100 custom tasks across all three policies
# This script will run all experiments sequentially

echo "=================================="
echo "Running 100 Custom Navigation Tasks"
echo "=================================="
echo ""

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Please run: source .venv/bin/activate"
    exit 1
fi

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    echo "⚠️  .env file not found!"
    echo "Please create .env file with your GEMINI_API_KEY"
    exit 1
fi

# Check if tasks file exists
if [[ ! -f "tasks_custom_100.json" ]]; then
    echo "📝 Generating custom 100 tasks..."
    python3 tasks_custom_100.py
    echo ""
fi

# Default values
MAX_STEPS=50
RESUME_FLAG=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_FLAG="--resume"
            echo "📋 Resume mode: Will skip already completed tasks"
            shift
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--resume] [--max-steps N]"
            exit 1
            ;;
    esac
done

echo "Settings:"
echo "  Max steps per task: $MAX_STEPS"
echo "  Resume mode: ${RESUME_FLAG:-disabled}"
echo ""

# Function to run a policy
run_policy() {
    local policy=$1
    echo ""
    echo "=================================="
    echo "Running: $policy policy"
    echo "=================================="
    echo "Started at: $(date)"
    echo ""
    
    python loop.py \
        --policy "$policy" \
        --taskset custom100 \
        --max_steps $MAX_STEPS \
        $RESUME_FLAG
    
    local status=$?
    echo ""
    echo "Completed at: $(date)"
    
    if [[ $status -eq 0 ]]; then
        echo "✅ $policy policy completed successfully"
    else
        echo "❌ $policy policy failed with exit code $status"
        return $status
    fi
}

# Run all three policies
echo "Starting experiments..."
echo ""

# 1. Baseline
run_policy "baseline"
if [[ $? -ne 0 ]]; then
    echo "❌ Baseline policy failed. Stopping."
    exit 1
fi

# 2. Handicapped
run_policy "handicapped"
if [[ $? -ne 0 ]]; then
    echo "❌ Handicapped policy failed. Stopping."
    exit 1
fi

# 3. Two-agent
run_policy "two_agent"
if [[ $? -ne 0 ]]; then
    echo "❌ Two-agent policy failed. Stopping."
    exit 1
fi

echo ""
echo "=================================="
echo "All Experiments Complete! 🎉"
echo "=================================="
echo ""
echo "Results are saved in:"
echo "  - logs/baseline/"
echo "  - logs/handicapped/"
echo "  - logs/two_agent/"
echo ""
echo "To analyze results, run:"
echo "  python policy_analysis.py"
