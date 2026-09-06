"""Render the plain-language production candidate and business brief."""

from research.build_publication_applicability_report import ROOT, build


if __name__ == "__main__":
    build(
        ROOT / "research/production_model_business_brief.md",
        ROOT / "output/pdf/ivan_production_model_business_brief.pdf",
        "AP37, температура и бизнес-логика",
    )
