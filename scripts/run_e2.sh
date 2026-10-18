#!/usr/bin/env bash
# Launch 4 concurrent trace_gen shards over the held-out split on the L40.
# Usage: run_e2.sh <seed> <tag>
set -e
export LD_LIBRARY_PATH=$HOME/llama-build/bin:/usr/local/cuda/lib64
SEED=${1:-1234}
TAG=${2:-s1234}
cd $HOME/data
[ -f ho_00 ] || split -n l/4 -d gsm8k_heldout.tsv ho_
for i in 0 1 2 3; do
  setsid bash -c "$HOME/trace_gen --model $HOME/models/1p5b.gguf --data $HOME/data/ho_0$i --out $HOME/e2_${TAG}_$i --limit 2000 --streams 16 --max-trace 1024 --n-ctx 32768 --temp 0.6 --seed $SEED > $HOME/e2_${TAG}_$i.log 2>&1" < /dev/null > /dev/null 2>&1 &
done
echo "launched 4 shards seed=$SEED tag=$TAG"
