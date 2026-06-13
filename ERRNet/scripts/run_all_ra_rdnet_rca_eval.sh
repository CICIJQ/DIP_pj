#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-datasets/processed_data}"
RDNET_ROOT="${RDNET_ROOT:-results/xreflection_rdnet}"
OUT_ROOT="${OUT_ROOT:-results/ra_rdnet_rca}"
DEVICE="${DEVICE:-cpu}"
STRENGTH="${STRENGTH:-0.75}"
MASK_GAMMA="${MASK_GAMMA:-0.65}"
MASK_SENSITIVITY="${MASK_SENSITIVITY:-1.25}"
MASK_BLUR_RADIUS="${MASK_BLUR_RADIUS:-5}"
MAX_EXTRA_DELTA="${MAX_EXTRA_DELTA:-0.35}"
MASK_FLOOR="${MASK_FLOOR:-0.0}"
SELF_INPUT_DIR="${SELF_INPUT_DIR:-datasets/processed_data/reflection_pairs_tight_eval/testdata_CEILNET_table2}"
SELF_GT_DIR="${SELF_GT_DIR:-}"
SELF_RDNET_SUBDIR="${SELF_RDNET_SUBDIR:-self_collected}"

run_dataset() {
  local name="$1"
  local input_dir="$2"
  local gt_dir="$3"
  local rdnet_subdir="$4"
  local output_dir="${OUT_ROOT}/${name}"

  echo "[i] Running RA-RDNet RCA on ${name}"
  "${PYTHON}" scripts/run_ra_rdnet_rca.py \
    --input_dir "${input_dir}" \
    --rdnet_dir "${RDNET_ROOT}/${rdnet_subdir}" \
    --output_dir "${output_dir}" \
    --device "${DEVICE}" \
    --strength "${STRENGTH}" \
    --mask_gamma "${MASK_GAMMA}" \
    --mask_sensitivity "${MASK_SENSITIVITY}" \
    --mask_blur_radius "${MASK_BLUR_RADIUS}" \
    --max_extra_delta "${MAX_EXTRA_DELTA}" \
    --mask_floor "${MASK_FLOOR}"

  if [[ -n "${gt_dir}" && -d "${gt_dir}" ]]; then
    echo "[i] Evaluating ${name}"
    "${PYTHON}" scripts/eval_reflection_methods.py \
      --pred_dir "${output_dir}" \
      --gt_dir "${gt_dir}" \
      --pred_filename ra_rdnet_rca.png \
      --output_csv "${output_dir}/per_image_metrics.csv" \
      --output_json "${output_dir}/average_metrics.json" \
      --summary_txt "${output_dir}/summary.txt"
  else
    echo "[i] No GT for ${name}; saved outputs only."
  fi
}

run_dataset "CEILNet_table2" "${DATA_ROOT}/testdata_CEILNET_table2" "${DATA_ROOT}/testdata_CEILNET_table2" "CEILNet_table2"
run_dataset "real20" "${DATA_ROOT}/real20" "${DATA_ROOT}/real20" "real20"
run_dataset "sir2_withgt" "${DATA_ROOT}/sir2_withgt" "${DATA_ROOT}/sir2_withgt" "sir2_withgt"
run_dataset "SIR2_objects" "${DATA_ROOT}/objects" "${DATA_ROOT}/objects" "SIR2_objects"
run_dataset "SIR2_postcard" "${DATA_ROOT}/postcard" "${DATA_ROOT}/postcard" "SIR2_postcard"
run_dataset "SIR2_wild" "${DATA_ROOT}/wild" "${DATA_ROOT}/wild" "SIR2_wild"
run_dataset "self_collected" "${SELF_INPUT_DIR}" "${SELF_GT_DIR}" "${SELF_RDNET_SUBDIR}"

echo "[i] RA-RDNet RCA outputs and metrics are under ${OUT_ROOT}"
