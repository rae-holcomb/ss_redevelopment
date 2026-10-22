#!/bin/zsh
# Render the SMARTS noise-source note to PDF.
# Uses pandoc + xelatex (MacTeX) when available; otherwise falls back to
# pandoc -> self-contained HTML -> headless Chrome (no LaTeX needed).
set -e
cd "${0:A:h}"
path=(/Library/TeX/texbin $path)
if command -v xelatex >/dev/null; then
  pandoc smarts_noise_source_note.md --pdf-engine=xelatex \
    -V geometry:margin=0.9in -V fontsize=10pt -V mainfont="Helvetica Neue" -V monofont="Menlo" \
    -V colorlinks=true -o smarts_noise_source_note.pdf
else
  echo "xelatex not found; rendering via headless Chrome" >&2
  pandoc smarts_noise_source_note.md -s --css note.css --self-contained \
    --metadata pagetitle="SMARTS noise-source note" -o smarts_noise_source_note.html
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$PWD/smarts_noise_source_note.pdf" "file://$PWD/smarts_noise_source_note.html" 2>/dev/null
  rm -f smarts_noise_source_note.html
fi
ls -la smarts_noise_source_note.pdf
