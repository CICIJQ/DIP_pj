#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:-checkpoints/errnet_ours_realistic_cascade/errnet_cascade_latest.pt}"
RESULT_DIR="${2:-results/eval_ours}"

python test_errnet.py --model errnet_cascade_model --dataset ceilnet_table2 --result_dir "${RESULT_DIR}" --save_subdir CEILNet_table2 -r --icnn_path "${CKPT}" --hyper
python test_errnet.py --model errnet_cascade_model --dataset real20 --result_dir "${RESULT_DIR}" --save_subdir real20 -r --icnn_path "${CKPT}" --hyper
python test_errnet.py --model errnet_cascade_model --dataset sir2_withgt --result_dir "${RESULT_DIR}" --save_subdir sir2_withgt -r --icnn_path "${CKPT}" --hyper
python test_errnet.py --model errnet_cascade_model --dataset objects --result_dir "${RESULT_DIR}" --save_subdir SIR2_objects -r --icnn_path "${CKPT}" --hyper
python test_errnet.py --model errnet_cascade_model --dataset postcard --result_dir "${RESULT_DIR}" --save_subdir SIR2_postcard -r --icnn_path "${CKPT}" --hyper
python test_errnet.py --model errnet_cascade_model --dataset wild --result_dir "${RESULT_DIR}" --save_subdir SIR2_wild -r --icnn_path "${CKPT}" --hyper
