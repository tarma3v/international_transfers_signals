"""Build the Russian round-6 model-search checkpoint PDF."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle

from research.build_round3_report import (
    BLUE,
    GRAY,
    GREEN,
    LIGHT,
    NAVY,
    ORANGE,
    PALE_BLUE,
    PALE_GREEN,
    PALE_ORANGE,
    RED,
    bullets,
    callout,
    get_styles,
    heading,
    para,
    register_fonts,
    table,
)


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "output" / "pdf" / "ivan_round6_model_search_checkpoint.pdf"


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LIGHT)
    canvas.line(20 * mm, 14 * mm, 190 * mm, 14 * mm)
    canvas.setFont("Arial", 7)
    canvas.setFillColor(GRAY)
    canvas.drawString(20 * mm, 9 * mm, "International transfers signals | round 6")
    canvas.drawRightString(190 * mm, 9 * mm, f"05.09.2026 | {doc.page}")
    canvas.restoreState()


def metric_cards(styles):
    rows = [
        [para("1.714", styles, "MetricR3"), para("1.892", styles, "MetricR3"),
         para("1.31", styles, "MetricR3")],
        [para("minimum lift", styles, "MetricSmallR3"),
         para("mean lift, 5 horizons", styles, "MetricSmallR3"),
         para("signals / corridor-week", styles, "MetricSmallR3")],
    ]
    result = Table(rows, colWidths=[58 * mm] * 3, rowHeights=[16 * mm, 11 * mm])
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
        title="Round 6: поиск лучшей модели сигналов международных переводов",
        author="international_transfers_signals",
        subject="Leakage-free multi-horizon checkpoint",
    )
    story = [
        Spacer(1, 12 * mm),
        para("Round 6: текущий чекпоинт поиска модели", s, "TitleR3"),
        para("Созвон 05.09: Q&A, честный лидер и новые intraday MOEX эксперименты", s, "SubtitleR3"),
        metric_cards(s), Spacer(1, 8 * mm),
        callout(
            "Лучший point score: среднее причинных рангов CBA-geometry и noon-MOEX "
            "HistGB. Оно проходит lift 1.30 на всех h=1/3/5/10/20 без знания "
            "следующего курса ЦБ. Формально CBA-geometry остаётся incumbent: paired "
            "CI прироста нового лидера пересекает ноль.",
            s, PALE_GREEN, GREEN,
        ),
        Spacer(1, 7 * mm),
        table([
            ["h", "Case lift", "Симметричная выгода", "Future-only выгода"],
            ["1", "1.714", "+19.7 б.п.", "+71.8 б.п."],
            ["3", "1.954", "+33.8 б.п.", "+82.2 б.п."],
            ["5", "1.961", "+41.2 б.п.", "+86.1 б.п."],
            ["10", "2.010", "+55.4 б.п.", "+89.4 б.п."],
            ["20", "1.819", "+77.1 б.п.", "+70.8 б.п."],
        ], s, [18 * mm, 34 * mm, 60 * mm, 62 * mm]),
        Spacer(1, 7 * mm),
        para("Ветка: ivan-experiments | данные до 03.09.2026 | следующий курс ЦБ не используется", s, "SmallR3"),
        PageBreak(),
    ]

    story += heading("1. Что именно требует кейс", s)
    story += [callout(
        "Результат модели — сигнал «сейчас выгодный момент», а не численный прогноз курса. "
        "Для каждого h текущий курс должен быть не хуже всех следующих h публикаций ЦБ.",
        s, PALE_BLUE, BLUE,
    )]
    story += [para("Официальная формула оценки", s, "H2R3"), table([
        ["Элемент", "Зафиксированная трактовка"],
        ["Коридоры", "TJS, UZS, KGS, AMD, KZT; итог нужен по всем"],
        ["Горизонты", "h=1, 3, 5, 10, 20 публикаций ЦБ; главного h нет"],
        ["Hit", "Сегодняшний нормированный курс не выше каждого будущего курса до h"],
        ["Lift", "Hit rate сигналов / random-day hit rate того же коридора и года"],
        ["Выгода кейса", "Курс дня сигнала против среднего в симметричном окне -h..+h"],
        ["Доп. выгода", "Курс дня сигнала против только будущих h публикаций"],
        ["Частота", "1–2 сигнала в неделю на коридор как self-check"],
        ["Режим проверки", "walk-forward по нескольким коридорам; все h равноправны"],
    ], s, [43 * mm, 131 * mm])]
    story += [para("Новые уточнения Q&A 04–05.09", s, "H2R3")]
    story += bullets([
        "Lift сравнивает hit rate сигналов со случайным днём того же коридора и периода; порог 1.30 не относится к одному выбранному h.",
        "MOEX прямо разрешён как открытый воспроизводимый intraday-индикатор; для каждого сигнала нужно восстановить данные as-of T.",
        "Следующий курс ЦБ можно использовать только если он уже опубликован на дату решения, но наш строгий трек намеренно его не использует.",
        "Кейс на 60–70% продуктовый: модель должна вести к объяснимому триггеру коммуникации, а не обещать будущий курс.",
    ], s)
    story += [para("Почему мы всё ещё смотрим future-only", s, "H2R3")]
    story += bullets([
        "Она отвечает на более строгий бизнес-вопрос: насколько хуже ждать после сигнала.",
        "Она не заменяет официальную симметричную метрику, а защищает от красивого lift без полезного движения после решения.",
        "Частоту 1–2 сообщений на клиента нужно решать отдельным аллокатором между коридорами; в данных кейса нет клиентского спроса.",
    ], s)
    story += [PageBreak()]

    story += heading("2. Как работает текущий point leader", s)
    story += [para("Архитектура", s, "H2R3")]
    story += bullets([
        "Первый эксперт — geometry75_cba_consensus_basis25: CNY market-state геометрия плюс 25% consensus basis из официальных RUB/USD/CNY курсов Центрального банка Армении.",
        "Второй эксперт — HistGB на 31 признаке CNYRUBF/USDRUBF до 12:00 MSK: overnight, intraday returns, range, volatility, slope, CBR basis и cross-basis.",
        "HistGB переобучается поквартально на fav_h5; метка входит в train только после наступления пятой будущей публикации.",
        "Финальный score — простое среднее causal percentile ranks двух экспертов. Формула выбрана на 2024 до открытия 2025–2026.",
        "Решение принимает rolling-20 threshold с целевой долей 22%; текущий score попадает в историю только после решения.",
    ], s)
    story += [para("Почему это не утечка", s, "H2R3"), table([
        ["Риск", "Защита"],
        ["Следующий курс ЦБ", "Не используется ни в признаках, ни при принятии решения"],
        ["CBA / MOEX as-of", "CBA effective date < T; свеча MOEX заканчивается до T 12:00"],
        ["Неразрешённые метки", "В train попадают только примеры, у которых полностью наступил соответствующий h"],
        ["Порог по test", "Только предыдущие scores; текущий score обновляет state после решения"],
        ["Случайная корреляция", "Stale20 controls, block bootstrap, circular shifts и Holm correction"],
    ], s, [44 * mm, 130 * mm])]
    story += [callout(
        "Freshness MOEX подтверждается matched control: задержка обоих рыночных экспертов "
        "снижает minimum/mean до 1.331/1.461. Но paired minimum-gain нового score над "
        "CBA incumbent равен +0.091 с CI [-0.137; +0.202], поэтому это challenger, не доказанная замена.",
        s, PALE_ORANGE, ORANGE,
    ), PageBreak()]

    story += heading("3. Новый intraday раунд после Q&A", s)
    story += [table([
        ["Направление", "Лучший перенос 2025–2026", "Вердикт"],
        ["Noon perpetual HistGB", "min / mean 1.636 / 1.793", "сильный свежий эксперт"],
        ["Noon + CBA rank mean", "1.714 / 1.892", "лучший point score"],
        ["State-space balance", "2024 оставил noon leader", "не добавляет"],
        ["Shared five-h learner", "2024 оставил noon leader", "не добавляет"],
        ["Noon spot ML", "screen best min 1.528", "хуже лидера"],
        ["Noon signed spot", "screen 1.644; later 1.604", "fresh, режим не переносится"],
        ["Online noon/spot", "min 1.714; mean 1.898", "point boost, CI не проходит"],
        ["15:30 CNY mean basis", "min / mean 1.708 / 1.883", "отдельный timed challenger"],
    ], s, [56 * mm, 59 * mm, 59 * mm], small=True)]
    story += [Spacer(1, 5 * mm), para("Что это нам говорит", s, "H2R3")]
    story += bullets([
        "Свежий intraday рынок несёт настоящий сигнал: noon signed-spot падает с later min 1.604 до 0.844 при задержке 20 строк; 15:30 score — с 1.708 до 0.782.",
        "Сложный ML на spot не лучше простого экономического знака basis. В этом источнике объяснимый partial-fixing полезнее универсального boosting.",
        "Простой noon spot силён на 2024, но хуже в 2025–2026. Online Hedge исправляет h20 и выгоду, но paired mean-gain CI [-0.045; +0.076] не доказывает boost.",
        "15:30 — не тюнинг часа: это отдельный методологически заданный продукт до обычной публикации курса; смешивать его метрики с 12:00 без оговорки нельзя.",
    ], s)
    story += [PageBreak()]

    story += heading("4. Два честных времени принятия решения", s)
    story += [table([
        ["h", "12:00 consensus lift", "15:30 CNY-mean lift", "Что меняется"],
        ["1", "1.714", "1.708", "почти одинаково"],
        ["3", "1.954", "1.889", "12:00 лучше"],
        ["5", "1.961", "1.943", "12:00 лучше"],
        ["10", "2.010", "1.948", "12:00 лучше"],
        ["20", "1.819", "1.927", "15:30 лучше"],
    ], s, [18 * mm, 48 * mm, 48 * mm, 60 * mm])]
    story += [Spacer(1, 5 * mm), table([
        ["Вариант", "2025 h5 / rate", "2026 h5 / rate", "Combined future benefit"],
        ["12:00 consensus", "2.062 / 1.25", "1.873 / 1.47", "+86.1 б.п."],
        ["15:30 CNY mean", "1.983 / 1.38", "1.901 / 1.34", "+90.3 б.п."],
    ], s, [48 * mm, 42 * mm, 42 * mm, 42 * mm])]
    story += [Spacer(1, 5 * mm), callout(
        "Для презентации: 12:00 consensus — лучший общий point score. 15:30 CNY "
        "session mean — более поздний объяснимый триггер с лучшим h20 и чуть большей "
        "future-only выгодой. Оба полностью исключают опубликованный курс на завтра.",
        s, PALE_BLUE, BLUE,
    )]
    story += [para("Архивы и причинность", s, "H2R3")]
    story += bullets([
        "Noon futures: 16 951 CNYRUBF и 16 953 USDRUBF часовых свечей.",
        "Noon spot: 13 427 CNYRUB_TOM и 8 719 USD000UTSTOM часовых свечей.",
        "15:30 spot: 74 442 CNYRUB_TOM и 46 547 USD000UTSTOM 10-минутных свечей.",
        "Во всех блоках физическая порча cutoff/future свечей оставляет прошлые признаки бит-в-бит неизменными.",
    ], s)
    story += [PageBreak()]

    story += heading("5. Статус и что говорить на созвоне", s)
    story += [table([
        ["Кандидат", "Роль", "Статус"],
        ["CBA geometry", "формальный statistical incumbent", "min 1.623 / mean 1.855"],
        ["12:00 CBA + noon HistGB", "лучший point score", "min 1.714 / mean 1.892"],
        ["15:30 CNY mean basis", "объяснимый timed product", "min 1.708 / mean 1.883"],
        ["Online noon/spot", "research point challenger", "mean 1.898; CI не проходит"],
        ["Known-next after publication", "отдельный сценарий", "не входит в strict track"],
    ], s, [61 * mm, 66 * mm, 47 * mm])]
    story += [para("Следующий раунд", s, "H2R3")]
    story += bullets([
        "Мы честно пробили 1.30 на всех пяти h и держим 1–2 сигнала в неделю; основной нерешённый вопрос — статистическое подтверждение прироста над сильным incumbent.",
        "Лучший score объединяет независимые источники: медленный cross-bank state и свежий MOEX intraday, а не ещё одну производную одной истории target.",
        "15:30 partial-fixing — объяснимый продуктовый вариант: не обещает курс, а сообщает, что наблюдаемая CNY-сессия делает текущий день сравнительно выгодным.",
        "Все оценки 2025–2026 ретроспективны. Настоящее подтверждение — только shadow на новых публикациях после freeze.",
        "На следующем шаге связать модельный сигнал с банковским клиентским курсом и общим лимитом коммуникаций на клиента.",
    ], s)
    story += [callout(
        "Итог для созвона: строго без курса ЦБ на завтра minimum lift = 1.714, mean = 1.892. "
        "Механизм воспроизводим и объясним, но прирост над прежним CBA incumbent пока "
        "не доказан paired CI — это нужно говорить прямо.",
        s, PALE_GREEN, GREEN,
    )]
    story += [Spacer(1, 5 * mm), para(
        "Воспроизводимость: research/round6_protocol.md; results/research/round6/report.md; "
        "полные CSV, protocol.json, архивы MOEX и OOF outputs сохранены. Все 103 теста "
        "прошли перед сборкой этого чекпоинта.",
        s, "SmallR3",
    )]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return PDF


if __name__ == "__main__":
    print(build())
