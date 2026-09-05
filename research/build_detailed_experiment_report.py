"""Build the detailed Russian report over all completed experiment families."""
from __future__ import annotations

from pathlib import Path

from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
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
    PALE_RED,
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
PDF = ROOT / "output" / "pdf" / "ivan_detailed_experiment_report.pdf"


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LIGHT)
    canvas.line(20 * mm, 14 * mm, 190 * mm, 14 * mm)
    canvas.setFont("Arial", 7)
    canvas.setFillColor(GRAY)
    canvas.drawString(
        20 * mm, 9 * mm,
        "International transfers signals | ivan-experiments | detailed report",
    )
    canvas.drawRightString(190 * mm, 9 * mm, f"05.09.2026 | {doc.page}")
    canvas.restoreState()


def metric_cards(styles):
    rows = [
        [para("1.780", styles, "MetricR3"),
         para("2.059", styles, "MetricR3"),
         para("1.19", styles, "MetricR3")],
        [para("minimum lift across h", styles, "MetricSmallR3"),
         para("h=5 lift", styles, "MetricSmallR3"),
         para("signals / currency-week", styles, "MetricSmallR3")],
    ]
    result = Table(rows, colWidths=[58 * mm] * 3, rowHeights=[16 * mm, 11 * mm])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 1, BLUE),
        ("INNERGRID", (0, 0), (-1, -1), .4, colors.HexColor("#bfdbfe")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return result


def horizontal_bars(items, *, maximum=2.2, threshold=1.3,
                    width=174 * mm, height=83 * mm):
    drawing = Drawing(width, height)
    left, right, top, bottom = 57 * mm, 10 * mm, 7 * mm, 10 * mm
    chart_width = width - left - right
    chart_height = height - top - bottom
    row_height = chart_height / len(items)
    threshold_x = left + chart_width * threshold / maximum
    drawing.add(Line(
        threshold_x, bottom - 2, threshold_x, height - top + 1,
        strokeColor=RED, strokeWidth=1,
    ))
    drawing.add(String(
        threshold_x + 2, height - top - 2, "threshold 1.30",
        fontName="Arial", fontSize=6.4, fillColor=RED,
    ))
    for tick in (0.0, 0.5, 1.0, 1.5, 2.0):
        x = left + chart_width * tick / maximum
        drawing.add(Line(
            x, bottom - 2, x, height - top,
            strokeColor=LIGHT, strokeWidth=.4,
        ))
        drawing.add(String(
            x - 4, 1, f"{tick:.1f}", fontName="Arial",
            fontSize=6, fillColor=GRAY,
        ))
    for i, (label, value, color) in enumerate(items):
        y = height - top - (i + .72) * row_height
        drawing.add(String(
            1, y + 1, label, fontName="Arial", fontSize=6.6,
            fillColor=NAVY,
        ))
        drawing.add(Rect(
            left, y, chart_width * value / maximum, row_height * .48,
            fillColor=color, strokeColor=None,
        ))
        drawing.add(String(
            left + chart_width * value / maximum + 3, y + 1,
            f"{value:.3f}", fontName="Arial-Bold", fontSize=6.6,
            fillColor=NAVY,
        ))
    return drawing


def architecture(width=174 * mm, height=87 * mm):
    drawing = Drawing(width, height)
    boxes = [
        (2, 58, 37, 20, "Current CBR", "five target rates", PALE_BLUE, BLUE),
        (46, 58, 37, 20, "CNY 10-min", "10:00-15:30", PALE_GREEN, GREEN),
        (90, 58, 37, 20, "CNY basis", "market / current CBR", PALE_GREEN, GREEN),
        (134, 58, 37, 20, "Causal rank", "prior 250 rows", PALE_BLUE, BLUE),
        (46, 16, 37, 20, "Noon fallback", "3-view futures", PALE_ORANGE, ORANGE),
        (90, 16, 37, 20, "Availability", "market or fallback", PALE_ORANGE, ORANGE),
        (134, 16, 37, 20, "Alert policy", "22%, rolling 20", PALE_BLUE, BLUE),
    ]
    for x_mm, y_mm, w_mm, h_mm, title, subtitle, fill, border in boxes:
        x, y, w, h = x_mm * mm, y_mm * mm, w_mm * mm, h_mm * mm
        drawing.add(Rect(
            x, y, w, h, rx=4, ry=4, fillColor=fill,
            strokeColor=border, strokeWidth=1,
        ))
        drawing.add(String(
            x + 4, y + 12 * mm, title, fontName="Arial-Bold",
            fontSize=7, fillColor=NAVY,
        ))
        drawing.add(String(
            x + 4, y + 5 * mm, subtitle, fontName="Arial",
            fontSize=5.9, fillColor=GRAY,
        ))
    for x1, y1, x2, y2 in (
        (39, 68, 46, 68), (83, 68, 90, 68), (127, 68, 134, 68),
        (64, 58, 64, 36), (83, 26, 90, 26), (127, 26, 134, 26),
        (152, 58, 152, 36),
    ):
        drawing.add(Line(
            x1 * mm, y1 * mm, x2 * mm, y2 * mm,
            strokeColor=GRAY, strokeWidth=1,
        ))
    drawing.add(Circle(130 * mm, 26 * mm, 1.5, fillColor=GRAY, strokeColor=None))
    return drawing


def build() -> Path:
    register_fonts()
    styles = get_styles()
    PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=18 * mm,
        title="Подробный отчёт по моделям сигналов международных переводов",
        author="international_transfers_signals",
        subject="Leakage-free research summary through packet EO",
    )
    story = [
        Spacer(1, 11 * mm),
        para("Подробный отчёт по моделям сигналов", styles, "TitleR3"),
        para(
            "Международные переводы: лучшие подходы, главные бусты, устойчивость и честность проверки",
            styles, "SubtitleR3",
        ),
        metric_cards(styles), Spacer(1, 7 * mm),
        callout(
            "Лучший результат строго без знания следующего курса ЦБ - 15:30 availability-router. "
            "На 2025-2026 он даёт minimum lift 1.780 по пяти официальным горизонтам, "
            "h=5 lift 2.059 и 1.19 сигнала на валюту-неделю. Все пять валют проходят "
            "h=5 lift 1.30. Результат причинный, но ретроспективный: 2025-2026 уже использовались "
            "для подтверждения гипотез, поэтому окончательное доказательство требует shadow-периода.",
            styles, PALE_GREEN, GREEN,
        ), Spacer(1, 6 * mm),
        table([
            ["Место", "Кандидат", "Главная роль", "Результат 2025-2026"],
            ["1", "15:30 availability-router", "лучший point score по ТЗ", "min/mean 1.780/1.956; h5 2.059; rate 1.19"],
            ["2", "12:00 three-view consensus", "более раннее решение", "min/mean 1.714/1.892; h5 1.995; rate 1.31"],
            ["3", "CBA geometry", "frozen statistical incumbent", "min/mean 1.623/1.855; h5 1.947; rate 1.26"],
            ["4", "logit50_extra50", "лучший чистый classic-ML h5", "h5 1.846; rate 1.28; min FX 1.538"],
        ], styles, [13 * mm, 51 * mm, 47 * mm, 63 * mm], small=True),
        Spacer(1, 6 * mm),
        para(
            "Состояние: packet EO | 109 тестов | ветка ivan-experiments | push не выполнялся",
            styles, "SmallR3",
        ), PageBreak(),
    ]

    story += heading("1. Главный ответ и порядок приоритетов", styles)
    story += [callout(
        "Если нужен один понятный кандидат на демонстрацию по условиям кейса, показываем "
        "15:30 availability-router. Если решение должно приниматься раньше, показываем "
        "12:00 consensus. CBA geometry сохраняем как формальный incumbent, потому что "
        "парное преимущество более новых лидеров над ним или noon пока не доказано.",
        styles, PALE_BLUE, BLUE,
    )]
    story += [para("Почему лидер соответствует ТЗ", styles, "H2R3")]
    story += bullets([
        "Предсказывается бинарный выгодный момент по всем пяти валютам TJS, UZS, KGS, AMD и KZT, а не численное значение курса.",
        "Следующий курс ЦБ не используется. В 15:30 видны только завершившиеся до cutoff свечи и текущий уже известный курс ЦБ.",
        "Minimum lift по h=1/3/5/10/20 равен 1.780, то есть каждый официальный горизонт выше порога 1.30.",
        "Средняя частота h=5 равна 1.19 сигнала на валюту-неделю; по годам 1.19 и 1.28.",
        "Минимальный h=5 lift отдельной валюты равен 1.832; future-only benefit h=5 равен +95.2 б.п.",
    ], styles)
    story += [para("Что нельзя обещать", styles, "H2R3")]
    story += bullets([
        "Это не прогноз фактического банковского клиентского курса, комиссии или конверсии.",
        "2025-2026 - protocol-controlled retrospective, но уже не pristine holdout.",
        "Incremental lift router над noon равен +0.066 по minimum, однако paired CI [-0.123; +0.238] включает ноль.",
        "Минимальная квартальная частота равна 0.955, на волос ниже внутреннего строгого порога 1.00, хотя средняя частота ТЗ проходит.",
    ], styles)
    story += [PageBreak()]

    story += heading("2. Постановка задачи и метрики", styles)
    story += [table([
        ["Элемент", "Зафиксированная трактовка"],
        ["Целевые валюты", "TJS, UZS, KGS, AMD, KZT; модель оценивается по всем коридорам"],
        ["Наблюдение", "одна публикация официального курса ЦБ, а не календарный день"],
        ["Hit на h", "сегодняшний нормированный курс не выше каждого из следующих h курсов"],
        ["Горизонты", "h=1, 3, 5, 10, 20; одного официального главного h нет"],
        ["Case lift", "hit rate сигналов / hit rate случайного дня того же коридора и периода"],
        ["Выгода кейса", "день сигнала против среднего в симметричном окне -h..+h"],
        ["Future-only benefit", "день сигнала против только будущих h публикаций; строгая бизнес-диагностика"],
        ["Частота", "self-check 1-2 сигнала в неделю на валюту; клиентский общий cap решается отдельно"],
    ], styles, [44 * mm, 130 * mm])]
    story += [Spacer(1, 5 * mm), callout(
        "Lift 1.40 не означает доходность 40%. Он означает, что hit rate среди выбранных "
        "дней в 1.40 раза выше базовой вероятности случайного дня. Денежный эффект отдельно "
        "измеряется в базисных пунктах.", styles, PALE_ORANGE, ORANGE,
    )]
    story += [para("Почему h=5 всё равно выделяется в исследовании", styles, "H2R3")]
    story += bullets([
        "На h=5 обучались основные supervised-модели и строились residual/stacking схемы.",
        "Финальный официальный scorecard всегда перепроверяется сразу на h=1/3/5/10/20.",
        "Нельзя выбрать удачный h после просмотра результата: minimum across horizons защищает от этого.",
    ], styles)
    story += [PageBreak()]

    story += heading("3. Данные и информационные границы", styles)
    story += [table([
        ["Источник", "Что использовано", "Причинная граница"],
        ["Банк России", "целевые курсы, USD/CNY, RUONIA, ставка, 20 reference FX", "только публикации, известные к T"],
        ["MOEX daily", "CNYRUB_TOM, USD, EUR", "TRADEDATE < signal date"],
        ["MOEX hourly", "spot и perpetual futures до 12:00", "candle end < cutoff"],
        ["MOEX 10-minute", "74 442 CNY и 46 547 USD свечей, окно до 15:30", "candle end < 15:30"],
        ["Локальные ЦБ", "Армения, Беларусь, Казахстан, Кыргызстан, Узбекистан и др.", "effective/release date строго доступна"],
        ["Календарь", "weekday, месяц, праздники, paydays, gaps", "известен заранее"],
    ], styles, [39 * mm, 73 * mm, 62 * mm], small=True)]
    story += [para("Пропущенные дни", styles, "H2R3")]
    story += bullets([
        "Выходные не интерполируются как новые публикации: горизонт измеряется следующими публикациями ЦБ.",
        "Если CNY-сессии нет, строка не удаляется из знаменателя. Router переключается на noon-score.",
        "Нулевой market score допустим только как явный frozen sentinel старого baseline; новые history transforms пропуски исключают.",
        "Calendar gap хранится отдельным причинным признаком, но сам по себе не улучшил перенос.",
    ], styles)
    story += [para("Почему нет точного VWAP", styles, "H2R3"), para(
        "Методика ЦБ использует объёмно-взвешенные сделки CNYRUB_TOM 10:00-15:30. "
        "Публичный исторический реестр сделок требует подписки, а поля value/volume в нашем "
        "свечном архиве пусты. Поэтому используется честный unweighted candle proxy и нигде "
        "не заявляется реконструкция истинного VWAP.", styles)]
    story += [PageBreak()]

    story += heading("4. Разбиение train / validation / test", styles)
    story += [table([
        ["Блок", "Роль в ранних волнах", "Роль в последних packet-экспериментах"],
        ["2010-2016", "development и train-only EDA", "длинная история для lifecycle"],
        ["2017-2020", "general validation", "transport старых режимов"],
        ["2021", "калибровка перед shock", "исторический мост"],
        ["2022-2023", "shock validation и reset", "SVO/regime audit"],
        ["2024", "retrospective final в ранних волнах", "единственный screen новых fixed families"],
        ["2025-2026", "retrospective audit", "открывается один раз после выбора на 2024"],
        ["после 03.09.2026", "не использован", "замороженный prospective shadow"],
    ], styles, [31 * mm, 63 * mm, 80 * mm], small=True)]
    story += [para("Критические правила", styles, "H2R3")]
    story += bullets([
        "h=5 label входит в train только когда пятая будущая публикация уже произошла до refit.",
        "Переобучение выполняется поквартально; calibration и test сохраняют хронологию.",
        "Рабочий threshold использует только предыдущие scores той же валюты; текущий score добавляется после решения.",
        "Каждая новая семья заранее записывается в round6_protocol.md до просмотра её later block.",
        "Repeated research делает 2025-2026 ретроспективным подтверждением, а не новым независимым holdout.",
    ], styles)
    story += [callout(
        "Честная причинность отвечает: можно ли было посчитать сигнал в тот момент? "
        "Pristine holdout отвечает на другой вопрос: не выбрали ли мы идею из-за удачи на уже "
        "просмотренном периоде? Первое доказано тестами, второе сможет доказать только будущий shadow.",
        styles, PALE_RED, RED,
    ), PageBreak()]

    story += heading("5. Лучший подход: 15:30 availability-router", styles)
    story += [architecture(), Spacer(1, 3 * mm)]
    story += [para("Формула", styles, "H2R3")]
    story += bullets([
        "Для текущего дня берутся все CNYRUB_TOM 10-минутные свечи, завершившиеся в окне 10:00-15:30.",
        "Считается среднее candle close и логарифмический basis к текущему уже известному CBR CNY: 10 000 x log(mean_market / current_CBR).",
        "Высокий положительный basis означает, что рынок уже переоценил CNY вверх, а текущие target rates вероятнее окажутся выгодными относительно будущих.",
        "Basis переводится в percentile rank относительно только предыдущих 250 значений той же целевой валюты.",
        "Если CNY-сессии нет, применяется ранее замороженный noon three-view futures consensus; строки не выбрасываются.",
        "Сигнал возникает при попадании score в causal rolling top 22% за предыдущие 20 наблюдений.",
    ], styles)
    story += [callout(
        "Главное достоинство лидера - он не пытается предсказать точный курс. Он измеряет "
        "уже наблюдаемый разрыв рынка к текущему фиксингу и превращает его в редкий timing signal.",
        styles, PALE_GREEN, GREEN,
    ), PageBreak()]

    story += heading("6. Метрики лидера по горизонтам", styles)
    story += [table([
        ["h", "Case lift", "Symmetric benefit", "Future-only benefit", "Сигналов"],
        ["1", "1.780", "+21.9 б.п.", "+77.6 б.п.", "506"],
        ["3", "2.014", "+41.4 б.п.", "+93.5 б.п.", "506"],
        ["5", "2.053", "+47.9 б.п.", "+95.2 б.п.", "506"],
        ["10", "2.005", "+55.2 б.п.", "+90.6 б.п.", "501"],
        ["20", "1.928", "+78.8 б.п.", "+79.2 б.п.", "486"],
    ], styles, [17 * mm, 31 * mm, 42 * mm, 45 * mm, 29 * mm])]
    story += [Spacer(1, 5 * mm), horizontal_bars([
        ("15:30 availability", 1.780, GREEN),
        ("12:00 consensus", 1.714, BLUE),
        ("raw 15:30 basis", 1.708, ORANGE),
        ("CBA geometry", 1.623, colors.HexColor("#7c3aed")),
        ("stale fixing control", .755, RED),
    ], height=69 * mm)]
    story += [para(
        "Столбцы показывают minimum lift по пяти h на общем блоке 2025-2026. "
        "Именно minimum, а не лучший отдельный горизонт, определяет основной порядок.",
        styles, "SmallR3",
    ), PageBreak()]

    story += heading("7. Устойчивость лидера по времени и валютам", styles)
    story += [table([
        ["Разрез h=5", "Frequency", "Lift", "Future benefit"],
        ["2025", "1.187", "2.096", "+73.2 б.п."],
        ["2026", "1.276", "1.994", "+127.6 б.п."],
        ["2025Q1", "1.203", "3.067", "+46.5 б.п."],
        ["2025Q2", "0.955", "1.405", "+14.2 б.п."],
        ["2025Q3", "1.277", "1.639", "+112.4 б.п."],
        ["2025Q4", "1.400", "2.521", "+95.9 б.п."],
        ["2026Q1", "1.455", "2.323", "+123.7 б.п."],
        ["2026Q2", "1.136", "1.626", "+100.5 б.п."],
        ["2026Q3 partial", "1.300", "2.053", "+171.5 б.п."],
    ], styles, [53 * mm, 35 * mm, 32 * mm, 44 * mm], small=True)]
    story += [Spacer(1, 5 * mm), table([
        ["Валюта", "Signals", "Frequency", "h=5 lift", "Future benefit"],
        ["TJS", "96", "1.133", "1.948", "+94.9"],
        ["UZS", "100", "1.180", "2.113", "+100.1"],
        ["KGS", "107", "1.263", "2.235", "+92.1"],
        ["AMD", "102", "1.204", "2.205", "+99.0"],
        ["KZT", "101", "1.192", "1.832", "+90.0"],
    ], styles, [30 * mm, 28 * mm, 35 * mm, 35 * mm, 36 * mm])]
    story += [callout(
        "Слабое место - 2025Q2: lift всё ещё проходит 1.30, но частота 0.955. "
        "Попытки искусственно добрать cadence увеличивали число слабых сигналов и не улучшали итог.",
        styles, PALE_ORANGE, ORANGE,
    ), PageBreak()]

    story += heading("8. Лучшие объяснимые подходы до публикации", styles)
    story += [table([
        ["Подход", "Что делает", "2025-2026", "Вердикт"],
        ["15:30 availability-router", "fixing rank, noon fallback", "min/mean 1.780/1.956", "point leader"],
        ["15:20 router", "тот же механизм на 10 минут раньше", "1.770/1.952", "сильный ранний challenger"],
        ["Raw 15:30 mean basis", "среднее CNY close / current CBR", "1.708/1.883", "лучший standalone formula"],
        ["12:00 consensus", "CNY/USDRUB futures + CBA geometry", "1.714/1.892", "лучшее раннее решение"],
        ["CBA geometry", "lagged ARM CB cross-rate state", "1.623/1.855", "formal incumbent"],
        ["CNY waveform", "20 completed sessions compressed", "h5 1.827; rate 1.22", "fresh path signal"],
        ["CNY analogue", "nearest historical trajectories", "h5 1.625", "понятный, но слабее"],
    ], styles, [42 * mm, 62 * mm, 36 * mm, 34 * mm], small=True)]
    story += [para("Почему простая формула выиграла у сложных моделей", styles, "H2R3")]
    story += bullets([
        "Она напрямую повторяет механизм будущего CBR CNY fixing, а не ищет косвенную корреляцию.",
        "Общий RUB/CNY фактор действует сразу на все пять коридоров и увеличивает effective sample size.",
        "Percentile rank убирает масштаб и медленный drift, не подгоняя supervised coefficients.",
        "Fallback сохраняет правильный denominator в выходные и дни отсутствия рынка.",
    ], styles)
    story += [PageBreak()]

    story += heading("9. Классический ML: лучшие модели и признаки", styles)
    story += [table([
        ["Модель", "Признаки", "h=5 lift / rate", "Статус"],
        ["logit50_extra50", "19-feature logit + CNY ExtraTrees", "1.846 / 1.284", "лучший clean ML"],
        ["CNY ExtraTrees", "lagged daily CNY market + target panel", "1.776 / 1.265", "сильный nonlinear component"],
        ["19-feature logit", "CNY intraday + range/returns + currency", "1.673 / 1.339", "лучший small explainable ML"],
        ["Hierarchical logit", "currency interactions + shrinkage", "1.697 / 1.324", "частичный pooling полезен"],
        ["Spline GAM blend", "smooth nonlinear transforms", "1.892 / 1.221", "point gain, CI не проходит"],
        ["Residual HistGB/Extra", "anchor + global residual", "screen max 1.558", "хуже простой формулы"],
        ["Shared horizon ExtraTrees", "одна модель для 5 barriers", "около 1.40", "complementary, но слабее"],
    ], styles, [40 * mm, 71 * mm, 34 * mm, 29 * mm], small=True)]
    story += [para("Состав 19-feature CNY logit", styles, "H2R3")]
    story += bullets([
        "CNY overnight, close, WAP, open-close, intraday range, volatility, age and missingness.",
        "Целевая валюта: range positions 30/90/180 и returns 1/5/20.",
        "Пять one-hot currency indicators; имя валюты явно присутствует.",
        "Все refits поквартальные; в train только полностью разрешённые h=5 labels.",
    ], styles)
    story += [callout(
        "Главный ML-вывод: ExtraTrees полезен как независимый нелинейный взгляд, но максимальный "
        "перенос даёт простой 50/50 rank blend с логистикой. Усложнение второго уровня чаще "
        "подгоняет 2024, чем улучшает 2025-2026.", styles, PALE_GREEN, GREEN,
    ), PageBreak()]

    story += heading("10. Ансамбли, режимы и модели ошибок", styles)
    story += [table([
        ["Подход", "Point result", "Что полезно", "Почему не основной"],
        ["Primary + regime logit 75/25", "h5 1.941 / rate 1.254", "лучший h5 ensemble point", "lift gain CI пересекает 0; cadence 0.98"],
        ["Primary + ROCKET 75/25", "1.911 / 1.204", "fresh convolution diversity", "gain CI [-0.129;0.298]"],
        ["Primary + GAM 75/25", "1.892 / 1.221", "smooth interactions", "max-adjusted p=0.182"],
        ["Reliability LCB", "1.897 / 1.157", "uncertainty-aware neighbour score", "2026 слабее; paired CI crosses 0"],
        ["Primary + local FX 75/25", "1.867 / 1.265", "частичный per-currency pooling", "gain +0.021; CI crosses 0"],
        ["Online Hedge", "до mean 1.898", "weights update after labels resolve", "frequency drift; no stable gain"],
        ["Hard/soft routers", "обычно ниже equal blend", "диагностика режимов", "слишком мало независимых regimes"],
    ], styles, [47 * mm, 37 * mm, 48 * mm, 42 * mm], small=True)]
    story += [para("Что такое честный regime routing", styles, "H2R3"), para(
        "Роутер не знает, какая модель окажется лучшей в будущем. Он видит только уже разрешённые "
        "ошибки экспертов, их текущие causal ranks и заранее известные market-state признаки. "
        "Post-hoc выбор победителя года запрещён. Даже при такой защите режимных эпизодов мало, "
        "поэтому learned routers оказались менее надёжными, чем небольшой фиксированный вес.", styles)]
    story += [PageBreak()]

    story += heading("11. Классические временные ряды и нейросети", styles)
    story += [table([
        ["Семейство", "Лучший результат", "Диагноз"],
        ["Seasonal naive", "AUC 0.458", "календарная сезонность не переносится"],
        ["ETS", "AUC 0.478", "ошибка прогноза уровня не совпадает с tail target"],
        ["SARIMA", "AUC 0.523", "слишком слабый ranking редких выгодных дней"],
        ["GRU / RNN", "AUC 0.540", "мало независимых режимов, сложность не окупилась"],
        ["Quantile / future floor", "lift около 1.14", "multi-step errors размывают minimum event"],
        ["Discrete hazards", "final около 1.21", "ошибки пяти условных шагов накапливаются"],
        ["State-space / Markov", "до 1.32 retro", "частота и transport не проходят"],
    ], styles, [43 * mm, 38 * mm, 93 * mm])]
    story += [para("Почему direct forecast здесь проигрывает", styles, "H2R3")]
    story += bullets([
        "Target - экстремальное событие: сегодняшний курс должен пережить сразу h будущих публикаций.",
        "Хороший RMSE уровня не гарантирует правильный порядок редких лучших дней.",
        "На пять валют приходится много строк, но мало независимых макроэкономических режимов.",
        "Market basis ближе к механизму следующего официального fixing, чем экстраполяция собственного target ряда.",
    ], styles)
    story += [PageBreak()]

    story += heading("12. Внешние данные: что сработало", styles)
    story += [table([
        ["Источник", "Лучший эффект", "Итог"],
        ["MOEX CNY daily", "h5 ML 1.846; all-horizon geometry", "самый большой устойчивый новый источник"],
        ["MOEX CNY 10-min to 15:30", "router min 1.780; h5 2.059", "лучший explainable point leader"],
        ["MOEX perpetual futures to noon", "noon consensus min 1.714", "сильное более раннее решение"],
        ["Armenian CB RUB/USD/CNY", "CBA incumbent min 1.623", "лучший независимый local-CB source"],
        ["20 broad CBR currencies", "до 1.51 в отдельных blends", "самостоятельно нестабильно"],
        ["Другие local CB panels", "screen gains, later reversals", "не продвигать"],
        ["RUONIA / key rate / Brent", "около 1.0-1.2", "слишком медленно для h5 timing"],
    ], styles, [48 * mm, 56 * mm, 70 * mm])]
    story += [para("Главная закономерность источников", styles, "H2R3")]
    story += bullets([
        "Свежесть важнее ширины: недавний CNY рынок сильнее сотен медленных macro features.",
        "Независимый источник полезнее ещё одной производной target history.",
        "Aligned-versus-stale controls обязательны: без них внешняя серия может выглядеть умной только из-за общего тренда.",
        "Cross-bank dispersion и revisions содержат информацию, но она не дала переносимого прироста над CBA/noon лидерами.",
    ], styles)
    story += [PageBreak()]

    story += heading("13. Самые большие бусты", styles)
    story += [table([
        ["Ранг", "Приём", "Наблюдаемый boost", "Насколько доказан"],
        ["1", "Свежий 15:30 fixing proxy против stale20", "+0.926 minimum lift; CI [+0.638;+1.167]", "сильное freshness evidence"],
        ["2", "Добавить MOEX CNY к pre-MOEX stack", "h5 1.421 -> 1.846", "крупный point jump; later retro"],
        ["3", "Fresh 20-session waveform против stale20", "+0.579 lift; CI [+0.285;+0.937]", "статистически поддержан"],
        ["4", "Fresh random convolutions против stale20", "+0.325; CI [+0.056;+0.625]", "поддержан, но blend gain нет"],
        ["5", "CBA + noon market consensus", "minimum 1.623 -> 1.714", "paired CI прироста пересекает 0"],
        ["6", "Availability fallback", "minimum 1.714 -> 1.780", "полезный point gain; CI crosses 0"],
        ["7", "Resolved-error regime overlay", "h5 1.846 -> 1.941; benefit +10.4 б.п.", "benefit CI >0; lift CI crosses 0"],
        ["Отдельно", "Известный следующий курс после публикации", "до lift 2.459", "другой information set, не strict task"],
    ], styles, [14 * mm, 55 * mm, 61 * mm, 44 * mm], small=True)]
    story += [Spacer(1, 5 * mm), callout(
        "Самый важный практический boost дал не новый алгоритм, а новая своевременная информация: "
        "CNY market state до cutoff. После этого главные улучшения приходят от аккуратной комбинации "
        "независимых взглядов, а не от увеличения глубины модели.", styles, PALE_GREEN, GREEN,
    )]
    story += [PageBreak()]

    story += heading("14. Какие фичи оказались наиболее полезными", styles)
    story += [table([
        ["Группа", "Лучшие признаки / приёмы", "Роль"],
        ["CNY fixing", "session mean basis к текущему CBR", "главный объяснимый сигнал"],
        ["CNY intraday", "overnight, open-close, range, volatility, WAP deviations", "нелинейный ML-компонент"],
        ["Target history", "range positions 30/90/180; returns 1/5/20", "устойчивый multiscale anchor"],
        ["Currency identity", "пять one-hot indicators и локальные interactions", "разные base rates и чувствительности"],
        ["Cross-market", "CNY-via-USD basis, CNY/USDRUB futures agreement", "независимое подтверждение"],
        ["Path", "20-session waveform и fixed random convolutions", "дополнительная форма режима"],
        ["Resolved errors", "только уже завершённые outcomes экспертов", "безопасная regime диагностика"],
        ["Calendar", "weekday/month sin-cos, holidays, paydays, gap", "небольшой вспомогательный вклад"],
    ], styles, [37 * mm, 77 * mm, 60 * mm], small=True)]
    story += [para("Циклические признаки", styles, "H2R3"), para(
        "День недели и месяц кодируются sin/cos, а не порядковыми числами. Это правильно "
        "геометрически и устраняет искусственный разрыв воскресенье-понедельник, но отдельного "
        "крупного lift boost сезонность не дала. Бинарные holiday/payday/New-Year окна "
        "сохраняются как слабый контекст, а не ядро модели.", styles)]
    story += [para("Почему pct_range_90 не утечка", styles, "H2R3"), para(
        "Признак использует положение текущего значения только среди текущего и предыдущих "
        "90 публикаций. Ни одна будущая строка в окно не входит; это проверяется физическим "
        "обрезанием и порчей будущей части ряда.", styles)]
    story += [PageBreak()]

    story += heading("15. Частота сигналов и threshold policy", styles)
    story += [table([
        ["Политика", "Плюс", "Минус", "Решение"],
        ["Fixed train quantile", "простая и честная", "дрейф частоты", "baseline audits"],
        ["Rolling 20, rate 22%", "быстро адаптируется; causal", "шум на коротком окне", "основная политика"],
        ["Rolling 60/120", "ровнее шкала", "медленнее после regime shift", "для отдельных challengers"],
        ["Quarter reset", "убирает refit scale jump", "мало history в начале квартала", "не лидер"],
        ["Weekly cap/top-up", "контролирует коммуникации", "может добавлять слабые alerts", "downstream allocation"],
        ["Future test top-K", "красивая фиксированная частота", "прямая утечка test distribution", "запрещено"],
    ], styles, [43 * mm, 48 * mm, 48 * mm, 35 * mm], small=True)]
    story += [para("Почему 1-2 сигнала на валюту не равно 1-2 сообщения клиенту", styles, "H2R3"), para(
        "Модель даёт score по каждому коридору. Если клиент интересуется несколькими валютами, "
        "отдельный allocator должен выбрать 1-2 лучших сообщения суммарно. В данных кейса нет "
        "клиентских предпочтений и истории конверсии, поэтому это продуктовый слой поверх модели.", styles)]
    story += [callout(
        "Попытка поднять minimum-quarter cadence router с 0.955 выбрала rate 20%, но новый "
        "вариант опустился до 0.875 на later block. Частоту нельзя чинить по уже увиденному "
        "слабому кварталу - это превращается в post-hoc tuning.", styles, PALE_ORANGE, ORANGE,
    ), PageBreak()]

    story += heading("16. SVO и смена режима 2022", styles)
    story += [table([
        ["Период fixed 15:30 basis", "Minimum lift", "Mean lift", "h=5 lift", "Rate"],
        ["2022", "1.531", "1.656", "1.551", "1.19-1.38 annual band"],
        ["2023", "1.514", "1.726", "1.789", "в диапазоне 1-2"],
        ["2024", "1.578", "1.673", "1.668", "в диапазоне 1-2"],
        ["2025", "1.740", "1.925", "1.983", "1.38"],
        ["2026", "1.665", "1.843", "1.901", "1.34"],
        ["2022-2026", "1.601", "1.751", "1.751", "1.19-1.38"],
    ], styles, [50 * mm, 31 * mm, 31 * mm, 31 * mm, 31 * mm], small=True)]
    story += [para("Граница 24.02.2022", styles, "H2R3")]
    story += bullets([
        "03.01-23.02.2022: minimum/mean 1.471/1.691, frequency 1.465, всего 45 сигналов.",
        "24.02-31.12.2022: minimum/mean 1.533/1.635, frequency 1.269.",
        "Короткий pre-SVO кусок описательный: его недостаточно для причинного вывода о влиянии санкций.",
        "Механизм basis наблюдается по обе стороны границы, поэтому он не является только post-SVO артефактом.",
    ], styles)
    story += [callout(
        "Гипотеза о росте спроса на переводы после отключений SWIFT бизнесово правдоподобна, "
        "но в наших данных нет спроса, объёмов переводов и клиентских действий. Курсовой backtest "
        "не может подтвердить эту гипотезу.", styles, PALE_RED, RED,
    ), PageBreak()]

    story += heading("17. Что не улучшило лидера", styles)
    story += [table([
        ["Новая идея", "Screen 2024", "Later 2025-2026", "Вывод"],
        ["Global residual ML over fixing", "best new min 1.558 vs router 1.607", "selector оставил router", "capacity не добавляет signal"],
        ["Trailing robust-z normalization", "1.623 vs 1.607", "1.775/1.934 vs 1.780/1.956", "малый screen gain не переносится"],
        ["Intraday persistence / block minimum", "1.628 vs 1.607", "1.761/1.930", "фильтр удаляет полезные дни"],
        ["Target-to-CNY beta projection", "1.626/1.757 vs 1.607/1.653", "1.754/1.919", "2025Q2 lift падает до 1.068"],
        ["Static noon/fixing consensus", "оставил raw fixing", "нет открытия нового лидера", "agreement geometry лишняя"],
        ["Exact cutoff tuning", "15:20 ties 15:30", "h20 NI едва не проходит", "10 минут можно выиграть, не точность"],
    ], styles, [49 * mm, 42 * mm, 45 * mm, 38 * mm], small=True)]
    story += [para("Общий паттерн неудач", styles, "H2R3")]
    story += bullets([
        "Многие идеи слегка выигрывают на 2024 и откатываются на 2025-2026.",
        "Добавление currency-specific динамики часто улучшает cadence, но снижает precision в слабом квартале.",
        "Сложный feature engineering поверх уже сильного basis в основном переставляет одни и те же alerts.",
        "Отрицательные результаты сохранены, чтобы команда не повторяла те же ветки поиска.",
    ], styles)
    story += [PageBreak()]

    story += heading("18. Leakage audit и статистическая строгость", styles)
    story += [table([
        ["Риск", "Защита", "Проверка"],
        ["Будущий target в feature", "все rolling окна past/current only", "physical future corruption"],
        ["Неразрешённый h label", "reach date < refit date", "training logs + unit tests"],
        ["Same-day close после cutoff", "candle end < decision time", "timestamp corruption tests"],
        ["Порог по test", "shifted rolling history", "future-score mutation test"],
        ["Удаление missing rows", "availability fallback сохраняет scope", "exact row-count tests"],
        ["Post-hoc model winner", "screen 2024, later once", "protocol written before results"],
        ["Перекрывающиеся horizons", "four-week block bootstrap", "cluster-preserving CI"],
        ["Много гипотез", "Holm / max-shift diagnostics", "multiplicity reports"],
    ], styles, [43 * mm, 72 * mm, 59 * mm], small=True)]
    story += [para("Что статистически поддержано", styles, "H2R3")]
    story += bullets([
        "15:30 fresh fixing против stale20: minimum gain +0.926, CI полностью выше нуля.",
        "Waveform fresh против stale20: +0.579, CI [+0.285;+0.937].",
        "Random convolutions fresh против stale20: +0.325, CI [+0.056;+0.625].",
        "Сам router лучше случайного выбора: h=5 bootstrap CI [1.727;2.434], p=0.00025.",
    ], styles)
    story += [para("Что пока не доказано", styles, "H2R3")]
    story += bullets([
        "Superiority router над noon или CBA incumbent.",
        "Superiority regime/GAM/ROCKET blends над logit50_extra50.",
        "Сохранение текущих результатов на новом проспективном периоде.",
    ], styles)
    story += [PageBreak()]

    story += heading("19. Карта всех рассмотренных категорий", styles)
    story += [table([
        ["Категория", "Проверенные семейства", "Лучший итог"],
        ["Простые правила", "range, momentum, streak, percentile, seasonality", "multiscale range anchor"],
        ["Linear ML", "logit, elastic regularization, interactions", "19-feature CNY logit"],
        ["Tree ML", "ExtraTrees, HistGB, XGBoost, CatBoost", "CNY ExtraTrees / consensus"],
        ["Time series", "naive, ETS, SARIMA, quantile paths", "ниже task threshold"],
        ["Neural", "GRU / sequence encoding", "AUC около 0.54; не окупилось"],
        ["Analogue", "KNN paths, reliability surfaces", "LCB 1.897 point"],
        ["Additive", "global/local spline GAM", "primary75 GAM25: 1.892"],
        ["Convolution", "fixed random kernels over CNY path", "primary75 ROCKET25: 1.911"],
        ["Regime", "resolved-error stack, online Hedge, routers", "primary75 regime25: 1.941"],
        ["External macro", "broad CBR, rates, oil, local CB", "CBA geometry strongest"],
        ["Market intraday", "daily/hourly/10-min spot and futures", "15:30 router strongest"],
        ["Policy", "rolling/reset/weekly cap/cadence", "rolling 22% / 20"],
    ], styles, [37 * mm, 82 * mm, 55 * mm], small=True)]
    story += [PageBreak()]

    story += heading("20. Отдельный сценарий после публикации курса", styles)
    story += [callout(
        "Этот раздел не относится к strict pre-publication target. Если следующий эффективный "
        "курс ЦБ уже опубликован, первая часть будущего окна известна законно. Тогда conditional "
        "ExtraTrees достигал lift около 2.459 при частоте около 1.07.",
        styles, PALE_ORANGE, ORANGE,
    )]
    story += [table([
        ["Условие", "Почему обязательно"],
        ["Фактический publication timestamp наступил", "иначе v[t+1] является прямой утечкой"],
        ["Банк ещё исполняет сегодняшний клиентский курс", "иначе сигнал нельзя монетизировать"],
        ["Есть latency SLA", "переоценка банка может съесть окно"],
        ["Логируется фактический customer rate", "официальный курс ЦБ не равен цене перевода"],
    ], styles, [61 * mm, 113 * mm])]
    story += [para("Лучшие признаки post-publication", styles, "H2R3")]
    story += bullets([
        "known margin v[t+1]-v[t], нормированный на недавнюю volatility;",
        "направление известного первого шага и peer movement пяти валют;",
        "позиция нового опубликованного курса в коротком историческом диапазоне;",
        "past-only market context как дополнительная, а не заменяющая часть.",
    ], styles)
    story += [para(
        "Эту цифру нельзя смешивать с headline 1.780: information set и момент решения другие.",
        styles,
    ), PageBreak()]

    story += heading("21. Рекомендуемая финальная схема", styles)
    story += [table([
        ["Время", "Основной score", "Роль", "Что заморозить"],
        ["12:00", "three-view futures + CBA consensus", "ранний сигнал", "формулы, cutoffs, 22%/20"],
        ["15:20", "early availability-router", "операционный challenger", "не менять cutoff после later"],
        ["15:30", "fixing availability-router", "основной point leader", "mean basis, fallback, 22%/20"],
        ["после публикации", "conditional known-next model", "отдельный продукт", "hard timestamp gate"],
    ], styles, [27 * mm, 57 * mm, 43 * mm, 47 * mm], small=True)]
    story += [para("Что показывать на защите", styles, "H2R3")]
    story += bullets([
        "Сначала условия кейса: пять горизонтов, lift против случайного дня, частота 1-2.",
        "Затем главный результат 15:30: min 1.780, h5 2.059, rate 1.19, все валюты выше 1.30.",
        "Показать экономический механизм CNY basis и availability fallback, а не список алгоритмов.",
        "Отдельно показать stale20 ablation: именно свежая информация даёт крупнейший подтверждённый boost.",
        "Честно проговорить retrospective status и план prospective shadow.",
    ], styles)
    story += [para("Что запускать в shadow", styles, "H2R3")]
    story += bullets([
        "15:30 router как основной; 12:00 consensus и 15:20 router как frozen challengers.",
        "logit50_extra50 как независимый classic-ML benchmark.",
        "Логировать score, threshold history, fired flag, все публикационные timestamps и customer rate.",
        "Не менять веса, окна и признаки до заранее назначенной даты аудита.",
    ], styles)
    story += [PageBreak()]

    story += heading("22. Воспроизводимость и артефакты", styles)
    story += [table([
        ["Артефакт", "Назначение"],
        ["EXPERIMENTS_SUMMARY.md", "русский навигатор по лучшим и отрицательным результатам"],
        ["results/research/round6/report.md", "полный журнал packet A-EO"],
        ["research/round6_protocol.md", "precommit-описание каждой новой гипотезы"],
        ["results/research/round6/*/protocol.json", "точные frozen параметры и selection period"],
        ["results/research/round6/*/*.csv", "screen, later, breakdown, bootstrap и audits"],
        ["results/research/round6/*/outputs.pkl", "OOF scores для повторной проверки"],
        ["tests/", "109 correctness, leakage, as-of и causality тестов"],
        ["round6_prospective_shadow_protocol.md", "граница будущего независимого подтверждения"],
    ], styles, [65 * mm, 109 * mm])]
    story += [para("Ключевые внешние источники", styles, "H2R3")]
    story += bullets([
        '<link href="https://talenttrack.aitalenthub.ru/hackathon/cases/455">Страница кейса</link>.',
        '<link href="https://www.cbr.ru/currency_base/dynamics/">Динамика официальных курсов Банка России</link>.',
        '<link href="https://cbr.ru/Content/Document/File/162004/metod_6290-u.pdf">Методология установления официальных курсов</link>.',
        '<link href="https://www.moex.com/files/4be999zbzp80bx2bgmwayrtyx0">MOEX ISS developer manual</link>.',
    ], styles)
    story += [callout(
        "Итог: целевой уровень уже честно пройден. Наиболее сильное и объяснимое решение - "
        "своевременный CNY fixing basis с причинным fallback и rolling policy. Следующий "
        "качественный шаг - не ещё один retrospective тюнинг, а неизменяемый shadow на новых данных.",
        styles, PALE_GREEN, GREEN,
    )]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return PDF


if __name__ == "__main__":
    print(build())
