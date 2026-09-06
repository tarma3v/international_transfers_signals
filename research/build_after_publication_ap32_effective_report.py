"""Render the audited AP28-AP32 effective-reference checkpoint."""
from research.build_publication_applicability_report import ROOT, build


if __name__ == '__main__':
    build(
        ROOT / 'research/after_publication_ap32_effective_report.md',
        ROOT / 'output/pdf/ivan_after_publication_ap32_effective.pdf',
        'После курса ЦБ на завтра: AP28-AP32',
    )
