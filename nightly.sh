#!/bin/bash
# Nightly on mini (LaunchAgent nl.tue.vlm-nightly, 05:30): copy crowd-city's newest per-city values of CROWD into
# crowd/, rerun the comparison of the generated clips with CROWD, and push what changed. The values are the newest
# crowd_crossing_speed_by_city.csv on CROWD_HOST among CROWD_CSV_GLOBS: Shadab's incremental crowd-city run on lin5080
# leaves one in each finished step's folder (~/crowd_city_runs/<step>/_output/crossing_speed/), the older full run
# in ~/Github/crowd-city. Tables of fewer than MIN_CITIES cities (such as the one-city test step) are skipped.
# PYTHON is the environment with this project's packages and open_clip.
# Log: ~/Library/Logs/vlm-nightly.log.
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-$HOME/.venvs/vlm-crossing/bin/python}
HOST=${CROWD_HOST:-lin5080}
GLOBS=${CROWD_CSV_GLOBS:-"crowd_city_runs/*/_output/crossing_speed/crowd_crossing_speed_by_city.csv Github/crowd-city/_output/crossing_speed/crowd_crossing_speed_by_city.csv"}
MIN_CITIES=${MIN_CITIES:-50}
TARGET=crowd/crowd_crossing_speed_by_city.csv
export PATH="$HOME/micromamba/envs/crowd/bin:/usr/local/bin:/usr/bin:/bin:$PATH"  # ffprobe, gh, git
echo "=== $(date)"
git pull -q --ff-only || { echo "git pull failed"; exit 1; }
copy=$(mktemp)
newest=$(ssh -o BatchMode=yes "$HOST" "for f in \$(ls -t $GLOBS 2>/dev/null); do \
  [ \$(wc -l < \$f) -gt $MIN_CITIES ] && echo \$f && break; done")
if [ -n "$newest" ] && scp -q "$HOST:$newest" "$copy" && [ -s "$copy" ]; then
  if cmp -s "$copy" "$TARGET"; then echo "CROWD values unchanged ($HOST:$newest)"
  else mv "$copy" "$TARGET"; echo "CROWD values updated from $HOST:$newest"; fi
else
  echo "no per-city table found on $HOST; using the values in the repository"
fi
rm -f "$copy"
scaled=()
[ -f results/real_minimax_480/crossings.csv ] && scaled=(--scaled results/real_minimax_480)
run() { nice -n 10 "$PY" "$@" > /dev/null 2>&1 || { echo "failed: $*"; exit 1; }; }
run generated_speeds.py --readme
run speed_statistics.py --readme "${scaled[@]}"
run speed_generated_vs_crowd.py --readme
run city_frames.py --readme
run appearance_analysis.py --readme
git add crowd results readme README.md
if git diff --cached --quiet; then echo "nothing changed"; exit 0; fi
git commit -q -m "Nightly: comparison with CROWD's per-city values of $(date +%F)" && git push -q && echo "pushed"
