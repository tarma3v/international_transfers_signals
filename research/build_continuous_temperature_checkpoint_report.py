"""Render the audited continuous-temperature checkpoint report."""
from research.build_publication_applicability_report import ROOT, build


if __name__ == "__main__":
    build(
        ROOT / "research/continuous_temperature_checkpoint_report.md",
        ROOT / "output/pdf/ivan_continuous_temperature_checkpoint.pdf",
        "Прогноз выгодности перевода в любой момент",
    )
