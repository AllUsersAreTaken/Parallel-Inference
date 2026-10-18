#!/usr/bin/env bash
# Wait for the current seed-1234 shards, then run seeds 5678 and 9012.
while pgrep -f "e2_s1234_" > /dev/null; do sleep 30; done
bash $HOME/run_e2.sh 5678 s5678
while pgrep -f "e2_s5678_" > /dev/null; do sleep 30; done
bash $HOME/run_e2.sh 9012 s9012
while pgrep -f "e2_s9012_" > /dev/null; do sleep 30; done
echo ALL_DONE > $HOME/e2_all_done
