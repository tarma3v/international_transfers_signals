"""Render the plain-language final model report requested for the defense."""
from research.build_publication_applicability_report import ROOT, build


if __name__ == "__main__":
    build(
        ROOT / "research/final_model_plain_explained.md",
        ROOT / "output/pdf/ivan_final_model_plain_explained.pdf",
        "Итоговая модель международного перевода простыми словами",
    )
