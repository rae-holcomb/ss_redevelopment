#!/bin/zsh
# Render the SMARTS noise-source note to HTML (pandoc) and PDF (headless Chrome; no LaTeX needed).
set -e
cd "${0:A:h}"
pandoc smarts_noise_source_note.md -s --css note.css --self-contained --metadata pagetitle="SMARTS noise-source note" -o smarts_noise_source_note.html
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$PWD/smarts_noise_source_note.pdf" "file://$PWD/smarts_noise_source_note.html" 2>/dev/null
ls -la smarts_noise_source_note.pdf
