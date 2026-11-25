#!/usr/bin/env bash
# Builds the submission copies (Word + PDF) of docs/manuscript.md.
# Needs pandoc, python-docx and LibreOffice. Run from the repository root: bash submission/build.sh
set -euo pipefail
pandoc docs/manuscript.md -f markdown-yaml_metadata_block -o submission/manuscript.docx --resource-path=docs
python3 submission/format_docx.py submission/manuscript.docx
(cd submission && soffice --headless --convert-to pdf manuscript.docx >/dev/null)
