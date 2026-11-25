#!/usr/bin/env bash
# Build, test, benchmark and analyse in one command.
set -euo pipefail

mkdir -p results figures

make -s all
./test_correctness            # abort (set -e) if any test fails

# Record the environment for reproducibility.
{
  echo "date: $(date -Is)"
  echo "kernel: $(uname -a)"
  echo "compiler: $(g++ --version | head -n1)"
  lscpu || true
} > results/meta.txt

# Thread list: 1,2,4,... up to the number of logical cores.
MAXT=$(nproc)
THREADS="1"
for ((t=2; t<=MAXT; t*=2)); do THREADS="$THREADS,$t"; done
case ",$THREADS," in *",$MAXT,"*) ;; *) THREADS="$THREADS,$MAXT";; esac

./bench --out results/raw.csv --reps 30 --warmup 5 \
        --sizes 65536,262144,1048576,4194304,16777216,33554432 \
        --threads "$THREADS"

python3 scripts/analyze.py --raw results/raw.csv \
        --summary results/summary.csv --figdir figures
echo "Done. See results/summary.csv and figures/*.png"