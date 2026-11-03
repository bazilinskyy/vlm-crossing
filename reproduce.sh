#!/bin/bash
# Rebuild every number and figure of the paper from this repository alone: results/ (the crossings tracked in each set
# of clips), crowd/ (CROWD's per-city values), models/ and videos/generated_videos.csv. No GPU, crowd-city checkout or
# video files are needed; only the grid of city frames (Fig. 1) needs the videos in videos/ (see README).
# Figures go to paper_figures/ (PDF, the paper's file names); tables and the per-test statistics to _output/.
#     ./reproduce.sh                  # with uv: uv run is used
#     PYTHON=path/to/python ./reproduce.sh
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-}
[ -z "$PY" ] && PY="uv run python"
OUT=${FIGURES:-paper_figures}
mkdir -p "$OUT"
[ -f config ] || cp default.config config
scaled=()
[ -f results/real_minimax_480/crossings.csv ] && scaled=(--scaled results/real_minimax_480)
step() { echo "=== $*"; $PY "$@" || { echo "failed: $*"; exit 1; }; }
step generated_speeds.py --pdf-dir "$OUT"               # Table 2, city medians, correlations, paired clips
step speed_statistics.py --pdf-dir "$OUT" "${scaled[@]}"  # tests and intervals in the text; Fig. 2
step speed_generated_vs_crowd.py --pdf "$OUT/speed_generated_vs_crowd.pdf"   # Fig. 3
step appearance_analysis.py --from-results --pdf-dir "$OUT"                  # appearance results; map
if [ -f "$(sed -n '2p' videos/generated_videos.csv | cut -d, -f5)" ]; then
  step city_frames.py --pdf-dir "$OUT"                                       # Fig. 1
else
  echo "=== city_frames.py skipped: the videos are not in videos/ (Fig. 1 needs them)"
fi
echo "Figures in $OUT/; numbers in _output/ (generated_usability.csv, generated_statistics.csv,"
echo "generated_reliability.csv, generated_correlations.csv, generated_paired_clips.csv, appearance_summary.csv)."
