#!/bin/bash
# Re-run decide_culture_neurons.py for every condition/dataset combination
# that has the required score files locally. Fast — no GPU needed.
# Run from repo root: bash scripts/regen_culture_neurons.sh

set -euo pipefail
NEURONS="outputs/neurons"

for cond_dir in "$NEURONS"/*_gemma4/; do
    dir_name=$(basename "$cond_dir")
    model_size="gemma4"
    cond="${dir_name%_gemma4}"

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
