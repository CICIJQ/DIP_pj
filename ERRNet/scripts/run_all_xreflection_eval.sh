#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
XREFLECTION_ROOT="${XREFLECTION_ROOT:-external_models/xreflection/XReflection}"
RDNET_CONFIG="${RDNET_CONFIG:-${XREFLECTION_ROOT}/options/train_rdnet.yml}"
RDNET_CHECKPOINT="${RDNET_CHECKPOINT:-checkpoints/xreflection/rdnet-26.4849.ckpt}"
DEVICE="${DEVICE:-cuda:0}"
DATA_ROOT="${DATA_ROOT:-datasets/processed_data}"
OUT_ROOT="${OUT_ROOT:-results/xreflection_rdnet}"
SELF_INPUT_DIR="${SELF_INPUT_DIR:-datasets/processed_data/reflection_pairs_tight_eval/testdata_CEILNET_table2}"
SELF_GT_DIR="${SELF_GT_DIR:-}"
COMMAND_TEMPLATE="${COMMAND_TEMPLATE:-}"

COMMON_ARGS=(
  --xreflection_root "${XREFLECTION_ROOT}"
  --config "${RDNET_CONFIG}"
  --checkpoint "${RDNET_CHECKPOINT}"
  --device "${DEVICE}"
  --python "${PYTHON}"
)

if [[ -n "${COMMAND_TEMPLATE}" ]]; then
  COMMON_ARGS+=(--command_template "${COMMAND_TEMPLATE}")
fi

run_dataset() {
  local name="$1"
  local input_dir="$2"
  local gt_dir="$3"
  local output_dir="${OUT_ROOT}/${name}"

  echo "[i] Running XReflection-RDNet on ${name}"
  "${PYTHON}" scripts/run_xreflection_rdnet.py \
    "${COMMON_ARGS[@]}" \
    --input_dir "${input_dir}" \
    --output_dir "${output_dir}" \
    --dataset_name "${name}"

  if [[ -n "${gt_dir}" && -d "${gt_dir}" ]]; then
    echo "[i] Evaluating ${name}"
    "${PYTHON}" scripts/eval_reflection_methods.py \
      --pred_dir "${output_dir}" \
      --gt_dir "${gt_dir}" \
      --pred_filename xreflection_rdnet.png \
      --output_csv "${output_dir}/per_image_metrics.csv" \
      --output_json "${output_dir}/average_metrics.json" \
      --summary_txt "${output_dir}/summary.txt"
  else
    echo "[i] No GT for ${name}; saved outputs only."
  fi
}

run_dataset "CEILNet_table2" "${DATA_ROOT}/testdata_CEILNET_table2" "${DATA_ROOT}/testdata_CEILNET_table2"
run_dataset "real20" "${DATA_ROOT}/real20" "${DATA_ROOT}/real20"
run_dataset "sir2_withgt" "${DATA_ROOT}/sir2_withgt" "${DATA_ROOT}/sir2_withgt"
run_dataset "SIR2_objects" "${DATA_ROOT}/objects" "${DATA_ROOT}/objects"
run_dataset "SIR2_postcard" "${DATA_ROOT}/postcard" "${DATA_ROOT}/postcard"
run_dataset "SIR2_wild" "${DATA_ROOT}/wild" "${DATA_ROOT}/wild"
run_dataset "self_collected" "${SELF_INPUT_DIR}" "${SELF_GT_DIR}"

echo "[i] XReflection-RDNet outputs and metrics are under ${OUT_ROOT}"
