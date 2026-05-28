.PHONY: paper roc app test lint

# ── Paper ──────────────────────────────────────────────────────────────────────
# Compile paper/paper.tex → PDF, then render each page to a PNG for the app.
# Run this whenever paper.tex changes, then commit the updated files:
#   make paper && git add paper/ && git commit -m "docs: update paper"
paper:
	cd paper && \
	pdflatex -interaction=nonstopmode paper.tex > /dev/null && \
	pdflatex -interaction=nonstopmode paper.tex > /dev/null && \
	pdftoppm -r 180 -png paper.pdf page
	@echo "Paper compiled. Updated files:"
	@ls paper/paper.pdf paper/page-*.png

# ── ROC figure ─────────────────────────────────────────────────────────────────
# Regenerate artifacts/paper_roc.png (all three detectors at SNR 0/3/6 dB).
# Run after changing the models or scoring logic.
roc:
	.venv/bin/python scripts/gen_paper_roc.py

# ── App ────────────────────────────────────────────────────────────────────────
app:
	.venv/bin/streamlit run app.py

# ── CI checks ──────────────────────────────────────────────────────────────────
lint:
	.venv/bin/python -m ruff check .

test:
	.venv/bin/python -m pytest tests/ -v
