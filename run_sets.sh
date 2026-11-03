#!/bin/bash
# Track and analyse each set of clips into _output/<set>/ and keep its results in results/<set>/ (main.py needs a GPU).
# veo3 and minimax_h3 come from videos/<set>/; real_minimax from data/real_minimax/ (paired_real_clips.py cuts it from
# the CROWD videos), real_minimax_480 is the same scaled to 864x480. Speeds are m/s with crowd-city's Waymo model.
#     SETS="minimax_h3 real_minimax" ./run_sets.sh
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-.venv/bin/python}
for set in ${SETS:-veo3 minimax_h3 real_minimax}; do
  case $set in
    veo3|minimax_h3) data=videos/$set ;;
    *) data=data/$set ;;
  esac
  $PY - "$set" "$data" <<'PYTHON'
import json, sys
name, data = sys.argv[1:]
config = json.load(open("default.config"))
config.update(data=data, output=f"_output/{name}", seg_data=f"_output/{name}/segmentation", open_figures=False,
              speed_model="models/waymo_calibration_v32/crowd_waymo_pipeline_model.json")
json.dump(config, open("config", "w"), indent=2)
PYTHON
  echo "=== $set $(date)"
  nice -n 10 $PY main.py && nice -n 10 $PY analysis.py && $PY export_results.py "$set" || echo "=== $set FAILED"
done
echo "=== done $(date)"
