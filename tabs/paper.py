"""
tabs/paper.py — render() for tab_paper.
"""

from pathlib import Path
import streamlit as st


def render():
    _pdf_path = Path("paper/paper.pdf")

    # ── PDF pages as images ───────────────────────────────────────────────────────
    _page_imgs = sorted(Path("paper").glob("page-*.png"))
    if _page_imgs:
        _dl_col, _ = st.columns([1, 4])
        with _dl_col:
            if _pdf_path.exists():
                with open(_pdf_path, "rb") as _f:
                    st.download_button(
                        "⬇ Download PDF",
                        data=_f,
                        file_name="Bach_RadarDetection_2026.pdf",
                        mime="application/pdf",
                    )
        for _pg in _page_imgs:
            st.image(str(_pg), use_container_width=True)
            st.divider()
    else:
        st.error(
            "Paper pages not found — run: "
            "`cd paper && pdflatex paper.tex && "
            "pdftoppm -r 180 -png paper.pdf page`"
        )
