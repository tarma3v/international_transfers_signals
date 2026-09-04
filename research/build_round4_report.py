"""Build the compact, verified PDF for round-four research."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle

from research.build_round3_report import (
    BLUE,
    GREEN,
    NAVY,
    ORANGE,
    PALE_BLUE,
    PALE_GREEN,
    PALE_ORANGE,
    bullets,
    callout,
    footer,
    get_styles,
    heading,
    para,
    register_fonts,
    table,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "research" / "round4"
PDF = ROOT / "output" / "pdf" / "ivan_deep_research_round4.pdf"


def _metric_cards(styles):
    cells = [
        [para("2.459", styles, "MetricR3"), para("1.228", styles, "MetricR3"),
         para("1.132", styles, "MetricR3")],
        [para("after-publication selected", styles, "MetricSmallR3"),
         para("ordinary selected", styles, "MetricSmallR3"),
         para("window-closing selected", styles, "MetricSmallR3")],
    ]
    result = Table(cells, colWidths=[58 * mm] * 3, rowHeights=[15 * mm, 10 * mm])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 1, BLUE),
        ("INNERGRID", (0, 0), (-1, -1), .4, colors.HexColor("#bfdbfe")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return result


def build() -> Path:
    register_fonts()
    s = get_styles()
    PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=18 * mm,
        title="Round 4: conditional publication-time signals",
        author="international_transfers_signals",
    )
    story = []

    story += [Spacer(1, 18 * mm), para("Round 4: условные сигналы", s, "TitleR3"),
              para("После публикации ЦБ, обычный past-only режим и закрытие окна", s, "SubtitleR3"),
              _metric_cards(s), Spacer(1, 8 * mm), callout(
                  "Главный результат: после публикации следующего эффективного курса выбранный "
                  "на 2017-2023 conditional ExtraTrees даёт lift 2.459 при частоте 1.069. "
                  "До публикации новые state-модели не проходят lift 1.30 стабильно. "
                  "Для window-closing лучший честно выбранный результат - 1.132.",
                  s, PALE_GREEN, GREEN), Spacer(1, 6 * mm),
              para("Все оценки 2024-2026 являются ретроспективными: блок уже просматривался.", s, "SmallR3"),
              para("Ветка: ivan-experiments | 04.09.2026", s, "SmallR3"), PageBreak()]

    story += heading("1. Что выбрано честно, а что найдено ретроспективно", s)
    story += [table([
        ["Контур", "Выбор до final", "2017-2020", "2022-2023", "2024-2026", "Частота final"],
        ["После публикации", "ExtraTrees 7y", "2.632", "2.395", "2.459", "1.069"],
        ["Обычный сигнал", "EB + LCB", "0.994", "1.098", "1.228", "1.235"],
        ["Окно закрывается", "Trend anchor", "0.989", "1.181", "1.132", "1.629"],
    ], s, [31*mm, 35*mm, 26*mm, 26*mm, 30*mm, 27*mm])]
    story += [Spacer(1, 5 * mm), callout(
        "Не смешивать две цифры. 2.459 - результат кандидата, выбранного по ранним "
        "блокам. 2.553 - максимум среди четырёх уже отобранных моделей на просмотренном "
        "2024-2026; это challenger, а не новая unbiased оценка.", s, PALE_ORANGE, ORANGE)]
    story += [para("Единый протокол", s, "H2R3")]
    story += bullets([
        "General validation: 2017-2020; проверка переноса режима: 2022-2023.",
        "Каждый тестовый год имеет отдельный предыдущий калибровочный год.",
        "h=5 label попадает в train только после фактической пятой публикации.",
        "Порог считается по прошлому году отдельно для каждой валюты.",
        "Требование частоты: 1-2 сигнала на валюту в неделю.",
        "Block bootstrap ресемплирует четырёхнедельные блоки целиком.",
    ], s)
    story += [para("Условная модель после публикации", s, "H2R3"), table([
        ["Метрика 2024-2026", "Selected ExtraTrees", "Best finalist ensemble"],
        ["Lift", "2.459", "2.553"],
        ["95% CI lift", "[2.160; 2.797]", "[2.237; 2.922]"],
        ["Future-only выгода", "+138.3 б.п.", "+142.0 б.п."],
        ["Минимум по году", "2.268", "2.258"],
        ["Минимум по валюте", "2.371", "2.376"],
    ], s, [58*mm, 58*mm, 58*mm]), Spacer(1, 3*mm), para(
        "Multiplicity audit: circular-shift max-adjusted p=0.00025 по 17 политикам; "
        "95-й процентиль null max lift равен 1.239.", s, "SmallR3")]
    story += [PageBreak()]

    story += heading("2. Почему известный следующий курс меняет задачу", s)
    story += [para(
        "Target fav_h5 требует, чтобы текущий курс был не выше всех следующих пяти. "
        "После публикации v[t+1] первый элемент target уже известен. Если он ниже v[t], "
        "target невозможен. Если выше, известный запас защищает текущий минимум от "
        "умеренного отката следующих четырёх публикаций.", s), callout(
            "Новая модель не предсказывает уже известный первый шаг. Она оценивает, "
            "выдержит ли известный запас четыре оставшиеся неизвестные публикации.",
            s, PALE_BLUE, BLUE), Spacer(1, 5*mm)]
    story += [table([
        ["Абляция; политика выбрана раньше final", "Lift", "Частота", "Выгода, б.п."],
        ["Только gate v[t+1] >= v[t]", "1.959", "1.369", "+77.8"],
        ["Gate + past-only features, logit", "2.363", "1.080", "+112.5"],
        ["Только величина известного запаса", "2.342", "1.139", "+127.4"],
        ["Признаки в опубликованной точке t+1", "2.493", "1.066", "+135.5"],
        ["t+1 + запас, logit", "2.515", "1.080", "+137.9"],
        ["Logit + ExtraTrees, retro finalist", "2.553", "1.076", "+142.0"],
    ], s, [83*mm, 29*mm, 31*mm, 31*mm])]
    story += [para("Что несёт сигнал", s, "H2R3")]
    story += bullets([
        "Главный feature: известный рост v[t+1] относительно v[t], делённый на недавнюю волатильность.",
        "Далее: общий размер объявленного движения и новая позиция в коротком диапазоне.",
        "Полный logit добавляет past-only контекст; деревья ловят нелинейную достаточность запаса.",
        "Ансамбль усредняет percentile ranks logit и ExtraTrees по каждой валюте.",
    ], s)
    story += [para("Операционная граница", s, "H2R3"), para(
        "Банк России указывает: точное время публикации не регламентировано, обычно курс "
        "появляется до 18:00 МСК; установленный курс вступает в силу на следующий календарный "
        "день. Production обязан проверять timestamp обновления. До него весь этот контур запрещён.", s),
        PageBreak()]

    story += heading("3. До публикации: новые режимные модели", s)
    story += [para(
        "Иерархический empirical Bayes частично объединяет sparse-состояния по положению "
        "в 90-публикационном диапазоне, ret20/ret60, волатильности, валюте и месяцу. "
        "Directional Markov использует последние up/flat/down переходы, streak и режим "
        "волатильности. Оба семейства имеют expanding, decay, shrinkage, LCB и anchor-blend варианты.", s)]
    story += [table([
        ["Результат", "General", "Shock", "Final", "Частота final", "Вердикт"],
        ["Честно выбранный EB+LCB", "0.994", "1.098", "1.228", "1.235", "не проходит"],
        ["Markov+anchor, retro max", "1.112", "1.175", "1.316", "0.961", "ниже частоты"],
        ["Лучший min через 3 эпохи", "1.131", "1.129", "1.195", "в полосе", "min=1.129"],
    ], s, [48*mm, 23*mm, 23*mm, 23*mm, 29*mm, 28*mm])]
    story += [Spacer(1, 5*mm), callout(
        "Честного стабильного пробития 1.30 до публикации нет. Значение 1.316 нельзя "
        "выносить как победу: оно найдено на уже просмотренном final, частота 0.961, "
        "а ранние эпохи заметно слабее.", s, PALE_ORANGE, ORANGE)]
    story += [para(
        "Max-adjusted p после перебора ordinary-семейства: 0.0502 у EB и 0.0575 у "
        "Markov; обе строки нарушают полосу частоты.", s, "SmallR3")]
    story += [para("Post-2022 reset", s, "H2R3"), table([
        ["Кандидат", "2024 screen", "2025 confirm", "2026 audit", "Итог"],
        ["Reset Markov LCB", "1.330", "1.233; freq .879", "не прошёл gate", "нестабилен"],
        ["Reset EB + anchor", "1.115", "1.398", "1.262", "combined 1.243"],
    ], s, [48*mm, 31*mm, 36*mm, 29*mm, 30*mm]), Spacer(1, 3*mm)]
    story += [para(
        "Вывод не означает отсутствия полезного сигнала. Он означает, что CBR-only past "
        "не даёт воспроизводимого lift 1.30 в нужной полосе после учёта дрейфа. Следующий "
        "содержательный шаг - timestamped intraday market data до публикации, а не ещё один "
        "перебор архитектур на тех же дневных рядах.", s), PageBreak()]

    story += heading("4. Вторичная цель: окно закрывается", s)
    story += [para(
        "Target close_h5 равен единице, если v[t+5] > v[t]. На ранних блоках выбран trend "
        "anchor. Он переносится в положительную сторону после 2022 года, но не достигает 1.30.", s),
        table([
            ["Кандидат", "Lift final", "Частота", "Выгода", "Min год", "Min валюта"],
            ["Trend anchor, selected", "1.132", "1.629", "+25.0 б.п.", "1.098", "1.080"],
            ["Upper-range, retro best", "1.182", "1.458", "+32.1 б.п.", "1.119", "1.101"],
            ["ExtraTrees, selected #2", "1.162", "1.072", "+18.0 б.п.", "1.057", "1.094"],
        ], s, [44*mm, 25*mm, 27*mm, 31*mm, 24*mm, 24*mm]), Spacer(1, 5*mm),
        callout(
            "У upper-range правила 95% CI lift [1.021; 1.343]. Направление устойчиво "
            "положительное, но нижняя граница далека от 1.30. Лучший max-adjusted "
            "p по 18 close-политикам равен 0.165.", s, PALE_BLUE, BLUE)]
    story += [para("Leakage audit", s, "H2R3")]
    story += [para(
        "• Обычная и close матрицы заканчиваются на t.<br/>"
        "• Publication матрица использует ровно i+1 той же валюты и ничего позже.<br/>"
        "• Gate побитово совпадает с известным после публикации fav_h1.<br/>"
        "• Purge h=5 прошёл для каждого train/calibration/test года.<br/>"
        "• 54 теста репозитория прошли.", s
    )]
    story += [para("Решение", s, "H2R3")]
    story += [para(
        "• Заморозить conditional ExtraTrees как selected post-publication policy.<br/>"
        "• Logit + ExtraTrees оставить challenger до настоящего нового holdout.<br/>"
        "• До публикации оставить locked anchor benchmark; Markov - research-only.<br/>"
        "• Для close оставить upper-range правило challenger без обещания 1.30.<br/>"
        "• Логировать timestamp получения курса и timestamp выдачи каждого сигнала.", s
    )]
    story += [PageBreak()]

    story += heading("5. Воспроизводимость и карта результатов", s)
    story += [para("Ключевые файлы", s, "H2R3")]
    story += bullets([
        "research/round4_research.py - полный расчёт трёх контуров.",
        "research/round4_protocol.md - chronology и границы информации.",
        "results/research/round4/report.md - подробный текстовый отчёт.",
        "results/research/round4/headline_summary.csv - только выбранные на early blocks строки.",
        "results/research/round4/*_all_candidates_final_sensitivity.csv - retro diagnosis, не selection.",
        "results/research/round4/*_cross_period_robustness_diagnostic.csv - одна настройка через эпохи.",
        "results/research/round4/*_bootstrap.csv - четырёхнедельные интервалы.",
        "results/research/round4/leakage_audit.json - машинные проверки границы.",
        "results/research/round4/*_outputs.pkl - сохранённые OOF scores.",
    ], s)
    story += [Spacer(1, 4*mm), para("Источники", s, "H2R3")]
    story += bullets([
        '<link href="https://talenttrack.aitalenthub.ru/hackathon/cases/455">Условия кейса</link>.',
        '<link href="https://www.cbr.ru/faq/dkp/04/">Банк России: валютный рынок, публикация и срок действия курса</link>.',
        '<link href="https://www.cbr.ru/Reception/TopicalMessage/Page/2661">Банк России: время публикации</link>.',
        '<link href="https://proceedings.mlr.press/v130/gangrade21a.html">Selective classification via one-sided prediction</link>.',
        '<link href="https://proceedings.mlr.press/v266/retzlaff25a.html">Conformal coverage under nonstationarity</link>.',
    ], s)
    story += [Spacer(1, 5*mm), callout(
        "Короткий итог для защиты: после публикации новый условный ExtraTrees стабильно "
        "проходит порог с lift 2.459 и корректной частотой. До публикации и для close "
        "порог 1.30 не подтверждён; найденные максимумы честно маркированы как retro.",
        s, PALE_GREEN, GREEN)]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(PDF)
    return PDF


if __name__ == "__main__":
    build()
