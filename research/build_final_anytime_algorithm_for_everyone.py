"""Render the final any-time algorithm in deliberately simple language."""
from research.build_publication_applicability_report import ROOT, build


if __name__ == "__main__":
    build(
        ROOT / "research/final_anytime_algorithm_for_everyone.md",
        ROOT / "output/pdf/ivan_final_anytime_algorithm_for_everyone.pdf",
        "Финальное решение простыми словами",
    )
