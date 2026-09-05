"""Render the AP1 results without overwriting the preceding applicability report."""
from research.build_publication_applicability_report import ROOT, build


if __name__ == "__main__":
    build(ROOT/"research/after_publication_ap1_report.md",
          ROOT/"output/pdf/ivan_after_publication_ap1.pdf",
          "После публикации ЦБ: первый пакет экспериментов")
