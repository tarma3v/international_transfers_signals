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
        [para("1.623", styles, "MetricR3"), para("1.855", styles, "MetricR3"),
         para("1.26", styles, "MetricR3")],
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
        para("Строго past-only, пять горизонтов кейса и 1–2 сигнала на коридор в неделю", s, "SubtitleR3"),
        metric_cards(s), Spacer(1, 8 * mm),
        callout(
            "Текущий лидер: geometry75_cba_consensus_basis25. Он проходит порог "
            "lift 1.30 на всех h=1/3/5/10/20 без знания следующего курса ЦБ. "
            "Это лучший замороженный ретроспективный кандидат, но не новый pristine holdout.",
            s, PALE_GREEN, GREEN,
        ),
        Spacer(1, 7 * mm),
        table([
            ["h", "Case lift", "Симметричная выгода", "Future-only выгода"],
            ["1", "1.623", "+18.0 б.п.", "+70.3 б.п."],
            ["3", "1.913", "+31.3 б.п.", "+81.5 б.п."],
            ["5", "1.931", "+38.1 б.п.", "+85.3 б.п."],
            ["10", "1.927", "+47.1 б.п.", "+87.5 б.п."],
            ["20", "1.879", "+70.3 б.п.", "+79.4 б.п."],
        ], s, [18 * mm, 34 * mm, 60 * mm, 62 * mm]),
        Spacer(1, 7 * mm),
        para("Ветка: ivan-experiments | тесты: 89 passed | prospective freeze: verified", s, "SmallR3"),
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
    ], s, [43 * mm, 131 * mm])]
    story += [para("Почему мы всё ещё смотрим future-only", s, "H2R3")]
    story += bullets([
        "Она отвечает на более строгий бизнес-вопрос: насколько хуже ждать после сигнала.",
        "Она не заменяет официальную симметричную метрику, а защищает от красивого lift без полезного движения после решения.",
        "Частоту 1–2 сообщений на клиента нужно решать отдельным аллокатором между коридорами; в данных кейса нет клиентского спроса.",
    ], s)
    story += [PageBreak()]

    story += heading("2. Как работает текущий лидер", s)
    story += [para("Архитектура", s, "H2R3")]
    story += bullets([
        "База — label-free геометрия нескольких независимых CNY-экспертов: ранги их согласия, минимума и максимума без обучения на будущих target-метках оцениваемого периода.",
        "Добавка 25% — consensus basis из официальных RUB/USD/CNY курсов Центрального банка Армении. Для даты T берётся только локальная котировка с effective date строго раньше T.",
        "Компоненты переводятся в причинные per-currency ранги и смешиваются фиксированно 75/25.",
        "Решение принимает rolling-20 threshold с целевой долей 22%; текущий score попадает в историю только после решения.",
    ], s)
    story += [para("Почему это не утечка", s, "H2R3"), table([
        ["Риск", "Защита"],
        ["Следующий курс ЦБ", "Не используется ни в признаках, ни при принятии решения"],
        ["Same-day рынок", "Внешняя локальная дата должна быть строго меньше signal date"],
        ["Неразрешённые метки", "В train попадают только примеры, у которых полностью наступил соответствующий h"],
        ["Порог по test", "Только предыдущие scores; текущий score обновляет state после решения"],
        ["Случайная корреляция", "Stale20 controls, block bootstrap, circular shifts и Holm correction"],
    ], s, [44 * mm, 130 * mm])]
    story += [callout(
        "Честность признака CBA подтверждена отдельно: aligned consensus basis имеет lift 1.392, "
        "а тот же ряд с задержкой 20 строк — 0.914. Но прирост финального 25% blend над "
        "сильной geometry-базой статистически пока не отделён от нуля.",
        s, PALE_ORANGE, ORANGE,
    ), PageBreak()]

    story += heading("3. Что проверили после новых ответов организаторов", s)
    story += [table([
        ["Направление", "Лучший перенос 2025–2026", "Вердикт"],
        ["Georgia NBG cross-rates", "min lift 0.974", "отвергнуто"],
        ["Belarus NBRB dense blends", "frozen 30%: 1.564", "переобучение"],
        ["Belarus tiny 10% point", "min lift 1.625", "только posthoc гипотеза"],
        ["Пять local-CB экспертов по валютам", "screen min 1.105", "отвергнуто"],
        ["Exponential threshold", "min / mean 1.594 / 1.819", "хуже rolling-20"],
        ["Frozen calendar prior", "min 1.611", "нет переноса"],
        ["Weekly cap + confidence slots", "min / mean 1.553 / 1.684", "хуже"],
        ["Gap regime", "later min 1.605; lifecycle 1.549", "эффект не переносится"],
    ], s, [56 * mm, 59 * mm, 59 * mm], small=True)]
    story += [Spacer(1, 5 * mm), para("Что это нам говорит", s, "H2R3")]
    story += bullets([
        "Идея «своя простая модель на каждую валюту, сверху глобальный blend» проверена напрямую. Локальные центральные банки не дали общего переносимого улучшения; исключение — армянский basis как слабая добавка к уже сильной геометрии.",
        "Красивые подгруппы ошибок, например дни после длинного календарного разрыва, легко выглядят сильными posthoc и разваливаются на lifecycle.",
        "Увеличение частоты до 1.5 сигнала в неделю само по себе не повышает качество: экспоненциальный threshold набрал больше слабых дней.",
        "Сезонный prior практически не помогает. Сильная часть задачи остаётся в текущем рыночном состоянии и согласии независимых экспертов.",
    ], s)
    story += [PageBreak()]

    story += heading("4. Статус и следующий честный шаг", s)
    story += [table([
        ["Кандидат", "Роль", "Статус"],
        ["geometry75_cba_consensus_basis25", "официальный five-horizon leader", "freeze / shadow"],
        ["logit50_extra50", "понятная CNY ML база", "freeze / shadow"],
        ["primary75_regime_logit25", "сильный h5 point estimate", "cadence/lift gain не доказаны"],
        ["wave_extra", "независимый path expert", "freshness доказана"],
        ["market_anchor_logit", "малый объяснимый fallback", "freeze"],
    ], s, [61 * mm, 66 * mm, 47 * mm])]
    story += [para("Следующий раунд", s, "H2R3")]
    story += bullets([
        "Не менять frozen кандидатов по уже просмотренным 2025–2026 данным.",
        "Продолжать искать независимый past-only источник сигнала, а не ещё одну трансформацию того же CNY rank.",
        "Оценивать каждый новый пакет сразу на пяти горизонтах, по годам, валютам и квартальной частоте.",
        "Любой новый adaptive router сначала выбирать на 2024, затем открывать 2025–2026 один раз; привлекательные posthoc точки сохранять только как будущие гипотезы.",
        "Новые реальные публикации после freeze — единственный способ превратить ретроспективного лидера в подтверждённый prospective результат.",
    ], s)
    story += [callout(
        "Итог: порог 1.30 уже пройден честно и одновременно на всех пяти горизонтах. "
        "Главная нерешённая задача теперь не ‘добыть красивее число’, а доказать перенос "
        "финального blend на новых данных и связать сигнал с фактическим клиентским курсом.",
        s, PALE_GREEN, GREEN,
    )]
    story += [Spacer(1, 5 * mm), para(
        "Воспроизводимость: research/round6_protocol.md; results/research/round6/report.md; "
        "полные CSV, protocol.json и OOF outputs сохранены. Все 89 тестов прошли.",
        s, "SmallR3",
    )]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return PDF


if __name__ == "__main__":
    print(build())
