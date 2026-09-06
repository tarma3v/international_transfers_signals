"""Render the frozen final temperature-model explainer."""
from research.build_publication_applicability_report import ROOT, build


if __name__ == "__main__":
    build(
        ROOT / "research/final_temperature_model_explained.md",
        ROOT / "output/pdf/ivan_final_temperature_model_explained.pdf",
        "Итоговая модель температуры международного перевода",
    )
