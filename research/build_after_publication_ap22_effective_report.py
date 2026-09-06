"""Render the audited AP18-AP22 effective-reference checkpoint."""
from research.build_publication_applicability_report import ROOT, build


if __name__ == '__main__':
    build(
        ROOT / 'research/after_publication_ap22_effective_report.md',
        ROOT / 'output/pdf/ivan_after_publication_ap22_effective.pdf',
        'После курса ЦБ на завтра: AP18-AP22',
    )
