"""Render the audited AP40-AP45 technical checkpoint."""
from research.build_publication_applicability_report import ROOT, build


if __name__ == '__main__':
    build(
        ROOT / 'research/after_publication_ap45_effective_report.md',
        ROOT / 'output/pdf/ivan_after_publication_ap45_effective.pdf',
        'После курса ЦБ на завтра: AP40-AP45',
    )
