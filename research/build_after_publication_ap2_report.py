from pathlib import Path
from research.build_publication_applicability_report import build

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    build(ROOT/'research/after_publication_ap2_report.md',
          ROOT/'output/pdf/ivan_after_publication_ap2.pdf',
          'После публикации ЦБ: вечерний биржевой сигнал')
