#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-datasets/processed_data}"
RDNET_ROOT="${RDNET_ROOT:-results/xreflection_rdnet}"
OUT_ROOT="${OUT_ROOT:-results/ra_rdnet_mg}"
DEVICE="${DEVICE:-cuda:0}"
MG_CHECKPOINT="${MG_CHECKPOINT:-}"
INFER_SCRIPT="${INFER_SCRIPT:-scripts/infer_mg_rdnet_refiner.py}"
SELF_INPUT_DIR="${SELF_INPUT_DIR:-datasets/processed_data/reflection_pairs_tight_eval/testdata_CEILNET_table2}"
SELF_GT_DIR="${SELF_GT_DIR:-}"
SELF_RDNET_SUBDIR="${SELF_RDNET_SUBDIR:-self_collected}"

if [[ -z "${MG_CHECKPOINT}" ]]; then
  echo "ERROR: set MG_CHECKPOINT to a trained MG-RDNet checkpoint, e.g. best_psnr.pth." >&2
  exit 1
fi
if [[ ! -f "${MG_CHECKPOINT}" ]]; then
  echo "ERROR: MG checkpoint does not exist: ${MG_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${INFER_SCRIPT}" ]]; then
  echo "ERROR: inference script does not exist: ${INFER_SCRIPT}" >&2
  exit 1
fi

run_dataset() {
  local name="$1"
  local input_dir="$2"
  local gt_dir="$3"
  local rdnet_subdir="$4"
  local output_dir="${OUT_ROOT}/${name}"

  echo "[i] Running RA-RDNet MG Refiner on ${name}"
  "${PYTHON}" "${INFER_SCRIPT}" \
    --input_dir "${input_dir}" \
    --rdnet_dir "${RDNET_ROOT}/${rdnet_subdir}" \
    --checkpoint "${MG_CHECKPOINT}" \
    --output_dir "${output_dir}" \
    --device "${DEVICE}" \
    --output_filename ra_rdnet_mg.png

  if [[ -n "${gt_dir}" && -d "${gt_dir}" ]]; then
    echo "[i] Evaluating ${name}"
    "${PYTHON}" scripts/eval_reflection_methods.py \
      --pred_dir "${output_dir}" \
      --gt_dir "${gt_dir}" \
      --pred_filename ra_rdnet_mg.png \
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

echo "[i] RA-RDNet MG Refiner outputs and metrics are under ${OUT_ROOT}"
