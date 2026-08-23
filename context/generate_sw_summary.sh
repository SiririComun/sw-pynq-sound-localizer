#!/usr/bin/env bash
# ==============================================================================
# Script: context/generate_sw_summary.sh
# Target: sw-pynq-sound-localizer
# Purpose: Dump Git history, structure, SW files, and notebook JSON into context/sw_summary.txt
# ==============================================================================

# Dynamically resolve paths
CONTEXT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CONTEXT_DIR}/.." && pwd)"
OUTPUT_FILE="${CONTEXT_DIR}/sw_summary.txt"

echo "Generating concise SW summary into: ${OUTPUT_FILE}..."
> "${OUTPUT_FILE}"

cd "${REPO_ROOT}" || exit 1

# 1. Header, Branch Status & Git History
cat << 'LOG_EOF' >> "${OUTPUT_FILE}"
Git Context & Branch Info:

```log
LOG_EOF

if command -v git &> /dev/null && git rev-parse --is-inside-work-tree &> /dev/null; then
    echo "Current Branch: $(git branch --show-current)" >> "${OUTPUT_FILE}"
    echo "Branch Tracking Status:" >> "${OUTPUT_FILE}"
    git branch -vv >> "${OUTPUT_FILE}"
    echo "" >> "${OUTPUT_FILE}"
    echo "Git Commit Graph & History:" >> "${OUTPUT_FILE}"
    git log --graph --pretty=format:"%h %d - %cd : %s (%an)" --date=short -n 15 >> "${OUTPUT_FILE}"
fi

cat << 'LOG_EOF' >> "${OUTPUT_FILE}"
```

Files structure: 

```log
LOG_EOF

# 2. Directory Tree
if command -v tree &> /dev/null; then
    tree -a -I '.git|__pycache__|*.egg-info|.pytest_cache|venv|env|dist|build|.ipynb_checkpoints' >> "${OUTPUT_FILE}"
fi

cat << 'LOG_EOF' >> "${OUTPUT_FILE}"
```

LOG_EOF

# 3. Exact list of targeted files and their markdown syntax
TARGET_FILES=(
    ".github/workflows/pypi.yml:yaml"
    "hardware.json:json"
    "setup.py:python"
    "requirements.txt:text"
    "pynq_localizer/__init__.py:python"
    "pynq_localizer/loader.py:python"
    "pynq_localizer/hw_trigger.py:python"
    "pynq_localizer/array.py:python"
    "pynq_localizer/kinematics.py:python"
    "pynq_localizer/kinematics_dashboard.py:python"
    "pynq_localizer/doa.py:python"
    "pynq_localizer/radar_dashboard.py:python"
    "pynq_localizer/notebooks.py:python"
    "notebooks/01_zero_skew_verification.ipynb:json"
    "notebooks/02_realtime_kinematics_telemetry.ipynb:json"
    "notebooks/03_doppler_velocity_tracking.ipynb:json"
    "notebooks/04_acoustic_free_fall_gravity.ipynb:json"
    "notebooks/05_doa_polar_radar.ipynb:json"
    "context/generate_sw_summary.sh:bash"
)

# 4. Append each file
for item in "${TARGET_FILES[@]}"; do
    filepath="${item%%:*}"
    syntax="${item##*:}"
    filename=$(basename "$filepath")

    if [ -f "$filepath" ]; then
        echo "Adding: ${filepath}"
        {
            echo "${filename}:"
            echo ""
            echo "\`\`\`${syntax}"
            cat "${filepath}"
            echo "\`\`\`"
            echo ""
        } >> "${OUTPUT_FILE}"
    fi
done

echo "Done! SW context generated in: ${OUTPUT_FILE}"