#!/usr/bin/env bash
# Launch 6 parallel rsync streams to push the project to HPC account 2.
# Each stream writes to its own log file in outputs/rsync_par_<stamp>/.
# Resume-safe: --partial keeps half-done files, re-running picks up.
set -eo pipefail   # NOTE: NO -u; empty array expansion under set -u crashes the function

ROOT="/Users/liufan/projects/share/AI4S-PDE-CNS"
cd "$ROOT"

set -a
source /Users/liufan/projects/share/easy_connect_vpn/login/hpc_auth.conf
set +a

RSH="sshpass -p '$HPC_PASS_2' ssh -o StrictHostKeyChecking=no -o ProxyCommand='nc -X connect -x 127.0.0.1:8888 %h %p' -p $HPC_PORT_2"
DEST="$HPC_USER_2@$HPC_HOST_2:projects/AI4S-PDE-CNS"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$ROOT/outputs/rsync_par_$STAMP"
mkdir -p "$LOGDIR"

echo "log dir: $LOGDIR"
echo "remote:  $DEST/"
echo

launch() {
  local name=$1 src=$2 dst=$3
  shift 3
  local extra=("$@")
  echo "[launch] $name  ($src -> $dst)"
  nohup rsync -a --partial -e "$RSH" "${extra[@]}" \
    "$src" "$dst" > "$LOGDIR/$name.log" 2>&1 &
  echo $! >> "$LOGDIR/pids"
}

# Stream 1: heaviest — task1 (8 GB training data + checkpoint + src + description)
launch task1   "$ROOT/tasks/ai4s-pde-task1-burgers-fixed/"   "$DEST/tasks/ai4s-pde-task1-burgers-fixed/"

# Stream 2: task2 (~1 GB)
launch task2   "$ROOT/tasks/ai4s-pde-task2-burgers-multinu/" "$DEST/tasks/ai4s-pde-task2-burgers-multinu/"

# Stream 3: task3 (~850 MB)
launch task3   "$ROOT/tasks/ai4s-pde-task3-ks-multiparam/"   "$DEST/tasks/ai4s-pde-task3-ks-multiparam/"

# Stream 4: outputs/ (~2.8 GB of AIDE runs)
launch outs    "$ROOT/outputs/"                              "$DEST/outputs/"                                  \
  --exclude='rsync_par_*/' --exclude='rsync_to_hpc_*.log'

# Stream 5: data_and_sample_submission/ + task3_data_sample_submission/ (~2.3 GB combined)
launch dss1    "$ROOT/data_and_sample_submission/"           "$DEST/data_and_sample_submission/"
launch dss2    "$ROOT/task3_data_sample_submission/"         "$DEST/task3_data_sample_submission/"

# Stream 7: everything else — root files + small dirs (submission/, .git, dslighting, scripts, src, etc.)
launch rest    "$ROOT/"                                       "$DEST/"                                          \
  --exclude='tasks/ai4s-pde-task1-burgers-fixed/' \
  --exclude='tasks/ai4s-pde-task2-burgers-multinu/' \
  --exclude='tasks/ai4s-pde-task3-ks-multiparam/' \
  --exclude='outputs/' \
  --exclude='data_and_sample_submission/' \
  --exclude='task3_data_sample_submission/'

echo
echo "all streams launched. pids:"
cat "$LOGDIR/pids"
echo
echo "monitor with:"
echo "  ls $LOGDIR/"
echo "  tail -f $LOGDIR/task1.log"
echo "  watch 'pgrep -af rsync.*AI4S-PDE-CNS | wc -l'"
