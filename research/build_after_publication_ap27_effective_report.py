"""Render the audited AP23-AP27 effective-reference checkpoint."""
from research.build_publication_applicability_report import ROOT, build


if __name__ == '__main__':
    build(
        ROOT / 'research/after_publication_ap27_effective_report.md',
        ROOT / 'output/pdf/ivan_after_publication_ap27_effective.pdf',
        'После курса ЦБ на завтра: AP23-AP27',
    )
