"""Render the short narrative report for the defense."""
from research.build_publication_applicability_report import ROOT, build


if __name__ == "__main__":
    build(
        ROOT / "research/defense_model_short.md",
        ROOT / "output/pdf/ivan_defense_model_short.pdf",
        "Итоговое решение: коротко и понятно для защиты",
        author="Ivan Kalinin",
        footer_label="короткое объяснение итоговой модели",
    )
