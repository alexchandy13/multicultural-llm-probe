#!/bin/bash
# Re-run decide_culture_neurons.py for every condition/dataset combination
# that has the required score files locally. Fast — no GPU needed.
# Run from repo root: bash scripts/regen_culture_neurons.sh

set -euo pipefail
NEURONS="outputs/neurons"

# Both model sizes: the countryrc extension (8 -> 81 countries) invalidated every
# existing selection, not just gemma4's. Pass a glob as $1 to narrow, e.g.
#   bash scripts/regen_culture_neurons.sh '*_8b'
PATTERN="${1:-*}"

for cond_dir in "$NEURONS"/$PATTERN/; do
    dir_name=$(basename "$cond_dir")
    case "$dir_name" in
        *_gemma4) model_size="gemma4"; cond="${dir_name%_gemma4}" ;;
        *_8b)     model_size="8b";     cond="${dir_name%_8b}" ;;
        *) echo "  [skip] $dir_name: unrecognized model-size suffix"; continue ;;
    esac

    echo "=== $cond ($model_size) ==="

    # normad (yn)
    if [[ -f "$cond_dir/normad_yn_max_scores.json" && -f "$cond_dir/normadcontrol_max_scores.json" ]]; then
        echo "  normad (yn)"
        python culnig/decide_culture_neurons.py \
            --condition "$cond" --model-size "$model_size" \
            --dataset-names normad --yn-only
    fi

    # culturalbench
    if [[ -f "$cond_dir/culturalbench_max_scores.json" && -f "$cond_dir/culturalbenchcontrol_max_scores.json" ]]; then
        echo "  culturalbench"
        python culnig/decide_culture_neurons.py \
            --condition "$cond" --model-size "$model_size" \
            --dataset-names culturalbench
    fi

    # blend
    if [[ -f "$cond_dir/blend_max_scores.json" && -f "$cond_dir/blendcontrol_max_scores.json" ]]; then
        echo "  blend"
        python culnig/decide_culture_neurons.py \
            --condition "$cond" --model-size "$model_size" \
            --dataset-names blend
    fi
done

echo "Done."
