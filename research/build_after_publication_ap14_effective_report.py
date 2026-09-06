"""Render the audited AP14 effective-reference checkpoint."""
from research.build_publication_applicability_report import ROOT, build


if __name__ == '__main__':
    build(
        ROOT / 'research/after_publication_ap14_effective_report.md',
        ROOT / 'output/pdf/ivan_after_publication_ap14_effective.pdf',
        'После курса ЦБ на завтра: AP14',
    )
