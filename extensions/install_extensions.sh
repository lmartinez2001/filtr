#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

extensions_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for setup_file in "${extensions_dir}"/*/setup.py; do
    extension_dir="$(dirname "${setup_file}")"
    echo "Installing extension: $(basename "${extension_dir}")"
    (
        cd "${extension_dir}"
        python3 setup.py install --user
    )
done
