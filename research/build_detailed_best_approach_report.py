"""Render the school-level detailed explanation of the strongest strict approach."""

from research.build_publication_applicability_report import ROOT, build


if __name__ == "__main__":
    build(
        ROOT / "research/detailed_best_approach_for_school.md",
        ROOT / "output/pdf/описание_подробное.pdf",
        "Самый эффективный подход - подробное объяснение",
    )
