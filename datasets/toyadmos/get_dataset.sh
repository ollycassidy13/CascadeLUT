#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"

TRAIN_URL="https://github.com/ollycassidy13/CascadeLUT/releases/download/v1.0.0/toyadmos-train.zip"
TEST_URL="https://github.com/ollycassidy13/CascadeLUT/releases/download/v1.0.0/toyadmos-test.zip"

TMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

mkdir -p "${DATA_DIR}"

download_and_extract() {
    local url="$1"
    local archive_name="$2"
    local output_filename="$3"

    local archive_path="${TMP_DIR}/${archive_name}"
    local extract_dir="${TMP_DIR}/${archive_name%.zip}"

    echo "Downloading ${archive_name}..."

    curl \
        --fail \
        --location \
        --retry 3 \
        --output "${archive_path}" \
        "${url}"

    mkdir -p "${extract_dir}"

    echo "Extracting ${archive_name}..."
    unzip -q "${archive_path}" -d "${extract_dir}"

    mapfile -t pkl_files < <(
        find "${extract_dir}" -type f -name '*.pkl'
    )

    if [[ "${#pkl_files[@]}" -ne 1 ]]; then
        echo "Error: expected exactly one .pkl file in ${archive_name}, found ${#pkl_files[@]}." >&2
        exit 1
    fi

    mv -f "${pkl_files[0]}" "${DATA_DIR}/${output_filename}"

    echo "Installed ${DATA_DIR}/${output_filename}"
}

download_and_extract \
    "${TRAIN_URL}" \
    "toyadmos-train.zip" \
    "train.pkl"

download_and_extract \
    "${TEST_URL}" \
    "toyadmos-test.zip" \
    "test.pkl"

echo
echo "ToyADMOS dataset installed successfully:"
echo "  ${DATA_DIR}/train.pkl"
echo "  ${DATA_DIR}/test.pkl"