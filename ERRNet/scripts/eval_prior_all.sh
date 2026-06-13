#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:-checkpoints/errnet_prior_errnet/errnet_prior_latest.pt}"
RESULT_DIR="${2:-results/eval_prior}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON:-python}"

${PYTHON_BIN} test_errnet_prior.py --dataset ceilnet_table2 --result_dir "${RESULT_DIR}" --save_subdir CEILNet_table2 -r --icnn_path "${CKPT}" --hyper ${EXTRA_ARGS}
${PYTHON_BIN} test_errnet_prior.py --dataset real20 --result_dir "${RESULT_DIR}" --save_subdir real20 -r --icnn_path "${CKPT}" --hyper ${EXTRA_ARGS}
${PYTHON_BIN} test_errnet_prior.py --dataset sir2_withgt --result_dir "${RESULT_DIR}" --save_subdir sir2_withgt -r --icnn_path "${CKPT}" --hyper ${EXTRA_ARGS}
${PYTHON_BIN} test_errnet_prior.py --dataset objects --result_dir "${RESULT_DIR}" --save_subdir SIR2_objects -r --icnn_path "${CKPT}" --hyper ${EXTRA_ARGS}
${PYTHON_BIN} test_errnet_prior.py --dataset postcard --result_dir "${RESULT_DIR}" --save_subdir SIR2_postcard -r --icnn_path "${CKPT}" --hyper ${EXTRA_ARGS}
${PYTHON_BIN} test_errnet_prior.py --dataset wild --result_dir "${RESULT_DIR}" --save_subdir SIR2_wild -r --icnn_path "${CKPT}" --hyper ${EXTRA_ARGS}
