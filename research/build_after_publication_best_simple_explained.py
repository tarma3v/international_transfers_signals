"""Render a plain-language explanation of the best audited approach."""
from research.build_publication_applicability_report import ROOT, build


if __name__ == '__main__':
    build(
        ROOT / 'research/after_publication_best_simple_explained.md',
        ROOT / 'output/pdf/ivan_after_publication_best_simple_explained.pdf',
        'Лучший подход после публикации курса ЦБ - простое объяснение',
    )
