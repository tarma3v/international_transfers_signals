"""Build a plain-language explainer for the strongest signal approaches."""
from __future__ import annotations

from pathlib import Path

from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
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
PDF = ROOT / "output" / "pdf" / "ivan_best_approach_explained_simply.pdf"


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LIGHT)
    canvas.line(20 * mm, 14 * mm, 190 * mm, 14 * mm)
    canvas.setFont("Arial", 7)
    canvas.setFillColor(GRAY)
    canvas.drawString(
        20 * mm, 9 * mm,
        "International transfers signals | simple model explainer",
    )
    canvas.drawRightString(190 * mm, 9 * mm, f"05.09.2026 | {doc.page}")
    canvas.restoreState()


def metric_cards(styles):
    rows = [
        [para("2.059", styles, "MetricR3"),
         para("1.19", styles, "MetricR3"),
         para("279 / 506", styles, "MetricR3")],
        [para("pooled h=5 lift", styles, "MetricSmallR3"),
         para("signals / currency-week", styles, "MetricSmallR3"),
         para("successful h=5 alerts", styles, "MetricSmallR3")],
    ]
    result = Table(rows, colWidths=[58 * mm] * 3, rowHeights=[16 * mm, 11 * mm])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 1, BLUE),
        ("INNERGRID", (0, 0), (-1, -1), .4, colors.HexColor("#bfdbfe")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return result


def _box(drawing, x, y, w, h, title, subtitle, fill, border):
    drawing.add(Rect(
        x * mm, y * mm, w * mm, h * mm, rx=4, ry=4,
        fillColor=fill, strokeColor=border, strokeWidth=1,
    ))
    drawing.add(String(
        (x + 3) * mm, (y + h - 7) * mm, title,
        fontName="Arial-Bold", fontSize=7.5, fillColor=NAVY,
    ))
    drawing.add(String(
        (x + 3) * mm, (y + 4) * mm, subtitle,
        fontName="Arial", fontSize=6.1, fillColor=GRAY,
    ))


def _arrow(drawing, x1, y1, x2, y2, color=GRAY):
    drawing.add(Line(x1 * mm, y1 * mm, x2 * mm, y2 * mm,
                     strokeColor=color, strokeWidth=1.1))
    if x2 >= x1:
        drawing.add(Polygon([
            x2 * mm, y2 * mm,
            (x2 - 2.2) * mm, (y2 + 1.3) * mm,
            (x2 - 2.2) * mm, (y2 - 1.3) * mm,
        ], fillColor=color, strokeColor=None))


def simple_story(width=174 * mm, height=62 * mm):
    drawing = Drawing(width, height)
    _box(drawing, 2, 27, 39, 23, "Старая цена ЦБ", "ценник еще не сменился", PALE_BLUE, BLUE)
    _box(drawing, 49, 27, 39, 23, "Рынок CNY", "уже торгуется сегодня", PALE_GREEN, GREEN)
    _box(drawing, 96, 27, 34, 23, "Разрыв", "рынок выше ЦБ", PALE_ORANGE, ORANGE)
    _box(drawing, 138, 27, 34, 23, "Сигнал", "сегодня может быть дешево", PALE_GREEN, GREEN)
    for x1, x2 in ((41, 49), (88, 96), (130, 138)):
        _arrow(drawing, x1, 38.5, x2, 38.5)
    drawing.add(String(
        2 * mm, 10 * mm,
        "Идея похожа на магазин: официальный ценник меняют раз в день, а оптовая цена уже сдвинулась.",
        fontName="Arial", fontSize=7.5, fillColor=NAVY,
    ))
    return drawing


def market_timeline(width=174 * mm, height=70 * mm):
    drawing = Drawing(width, height)
    y = 34
    drawing.add(Line(8 * mm, y * mm, 168 * mm, y * mm,
                     strokeColor=NAVY, strokeWidth=1.4))
    for x, label, sub, color in (
        (10, "10:00", "рынок открылся", BLUE),
        (65, "12:00", "ранний score", ORANGE),
        (112, "15:30", "конец окна ЦБ", GREEN),
        (162, "до 18:00", "обычно публикация", RED),
    ):
        drawing.add(Circle(x * mm, y * mm, 3.3, fillColor=color, strokeColor=None))
        drawing.add(String((x - 7) * mm, (y + 8) * mm, label,
                           fontName="Arial-Bold", fontSize=7, fillColor=NAVY))
        drawing.add(String((x - 12) * mm, (y - 11) * mm, sub,
                           fontName="Arial", fontSize=6, fillColor=GRAY))
    drawing.add(String(
        8 * mm, 60 * mm,
        "До публикации: завтрашний курс неизвестен. После фактической публикации: он уже разрешенная информация.",
        fontName="Arial-Bold", fontSize=7.4, fillColor=NAVY,
    ))
    return drawing


def router_diagram(width=174 * mm, height=100 * mm):
    drawing = Drawing(width, height)
    _box(drawing, 58, 76, 58, 19, "Есть свечи CNY до 15:30?", "проверяем только доступность", PALE_BLUE, BLUE)
    _box(drawing, 7, 42, 55, 20, "Да: fixing score", "рынок CNY / текущий ЦБ", PALE_GREEN, GREEN)
    _box(drawing, 112, 42, 55, 20, "Нет: noon fallback", "резервный ранний score", PALE_ORANGE, ORANGE)
    _box(drawing, 59, 6, 56, 21, "Общий causal rank", "сравнение только с прошлым", PALE_BLUE, BLUE)
    drawing.add(Line(87 * mm, 76 * mm, 35 * mm, 62 * mm,
                     strokeColor=GRAY, strokeWidth=1.1))
    drawing.add(Line(87 * mm, 76 * mm, 139 * mm, 62 * mm,
                     strokeColor=GRAY, strokeWidth=1.1))
    drawing.add(String(49 * mm, 68 * mm, "ДА", fontName="Arial-Bold",
                       fontSize=7, fillColor=GREEN))
    drawing.add(String(121 * mm, 68 * mm, "НЕТ", fontName="Arial-Bold",
                       fontSize=7, fillColor=ORANGE))
    drawing.add(Line(35 * mm, 42 * mm, 75 * mm, 27 * mm,
                     strokeColor=GRAY, strokeWidth=1.1))
    drawing.add(Line(139 * mm, 42 * mm, 101 * mm, 27 * mm,
                     strokeColor=GRAY, strokeWidth=1.1))
    return drawing


def lift_math(width=174 * mm, height=76 * mm):
    drawing = Drawing(width, height)
    _box(drawing, 2, 38, 48, 25, "Обычный день", "26.78% удачных", PALE_BLUE, BLUE)
    _box(drawing, 63, 38, 48, 25, "Наши сигналы", "55.14% удачных", PALE_GREEN, GREEN)
    _box(drawing, 124, 38, 48, 25, "Lift", "55.14 / 26.78 = 2.059", PALE_ORANGE, ORANGE)
    _arrow(drawing, 50, 50, 63, 50)
    _arrow(drawing, 111, 50, 124, 50)
    drawing.add(String(
        4 * mm, 17 * mm,
        "Из 506 сигналов успешными оказались 279. При случайном выборе ожидалось бы примерно 136.",
        fontName="Arial-Bold", fontSize=8.1, fillColor=NAVY,
    ))
    return drawing


def horizon_bars(items, width=174 * mm, height=86 * mm):
    drawing = Drawing(width, height)
    left, bottom, top, maximum = 45 * mm, 9 * mm, 8 * mm, 2.25
    chart_width = width - left - 9 * mm
    row_h = (height - bottom - top) / len(items)
    x_threshold = left + chart_width * 1.30 / maximum
    drawing.add(Line(x_threshold, bottom - 2, x_threshold, height - top,
                     strokeColor=RED, strokeWidth=1))
    drawing.add(String(x_threshold + 2, height - top - 1, "порог 1.30",
                       fontName="Arial", fontSize=6, fillColor=RED))
    for i, (label, value, color) in enumerate(items):
        y = height - top - (i + .72) * row_h
        drawing.add(String(2, y + 1, label, fontName="Arial-Bold",
                           fontSize=7, fillColor=NAVY))
        drawing.add(Rect(left, y, chart_width * value / maximum, row_h * .48,
                         fillColor=color, strokeColor=None))
        drawing.add(String(left + chart_width * value / maximum + 3, y + 1,
                           f"{value:.3f}", fontName="Arial-Bold",
                           fontSize=7, fillColor=NAVY))
    return drawing


def information_modes(width=174 * mm, height=93 * mm):
    drawing = Drawing(width, height)
    drawing.add(Rect(2 * mm, 8 * mm, 80 * mm, 76 * mm, rx=5, ry=5,
                     fillColor=PALE_GREEN, strokeColor=GREEN, strokeWidth=1.2))
    drawing.add(Rect(92 * mm, 8 * mm, 80 * mm, 76 * mm, rx=5, ry=5,
                     fillColor=PALE_BLUE, strokeColor=BLUE, strokeWidth=1.2))
    drawing.add(String(8 * mm, 72 * mm, "РЕЖИМ A: 15:30", fontName="Arial-Bold",
                       fontSize=9, fillColor=GREEN))
    drawing.add(String(98 * mm, 72 * mm, "РЕЖИМ B: ПОСЛЕ ПУБЛИКАЦИИ", fontName="Arial-Bold",
                       fontSize=8.1, fillColor=BLUE))
    left = [
        "Завтрашний курс еще неизвестен",
        "Используем рынок CNY и текущий ЦБ",
        "Lift h=5: 2.059",
        "Можно отправить раньше",
    ]
    right = [
        "Завтрашний курс уже опубликован",
        "Использовать его разрешено",
        "Lift h=5: 2.459",
        "Нужен жесткий timestamp gate",
    ]
    for i, text in enumerate(left):
        drawing.add(String(8 * mm, (58 - i * 12) * mm, text,
                           fontName="Arial", fontSize=6.8, fillColor=NAVY))
    for i, text in enumerate(right):
        drawing.add(String(98 * mm, (58 - i * 12) * mm, text,
                           fontName="Arial", fontSize=6.8, fillColor=NAVY))
    return drawing


def build() -> Path:
    register_fonts()
    styles = get_styles()
    PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=18 * mm,
        title="Как работает лучший сигнал международных переводов",
        author="international_transfers_signals",
        subject="Plain-language explanation of the 15:30 availability router and the publication-time boundary",
    )

    story = [
        Spacer(1, 12 * mm),
        para("Как работает лучший сигнал", styles, "TitleR3"),
        para(
            "Объяснение 15:30 availability-router простыми словами: откуда берутся числа, почему возникает преимущество и когда разрешен курс ЦБ на завтра",
            styles, "SubtitleR3",
        ),
        metric_cards(styles), Spacer(1, 7 * mm),
        callout(
            "Главная мысль: официальный курс меняется дискретно, а рынок CNY/RUB движется внутри дня. "
            "Если к 15:30 рынок уже заметно выше текущего курса ЦБ, текущий официальный курс часто оказывается "
            "удачным относительно следующих публикаций. На неторговый день модель берет заранее зафиксированный "
            "резервный сигнал, а не удаляет наблюдение.",
            styles, PALE_GREEN, GREEN,
        ), Spacer(1, 6 * mm),
        simple_story(), Spacer(1, 4 * mm),
        para(
            "Документ сначала объясняет строгий pre-publication вариант. В конце отдельно разобран более сильный "
            "и допустимый по кейсу вариант после фактической публикации следующего эффективного курса.",
            styles, "SmallR3",
        ), PageBreak(),
    ]

    story += heading("1. Задача на примере пяти конфет", styles)
    story += [para(
        "Представим, что сегодняшний курс - это цена конфеты. Клиент хочет купить сегодня, только если завтра и "
        "еще несколько дней конфета не станет дешевле. Мы не угадываем точную будущую цену. Мы отвечаем на "
        "более простой вопрос: сегодняшний день похож на достаточно хороший момент или нет?",
        styles,
    )]
    story += [table([
        ["Обозначение", "Перевод на простой язык"],
        ["v[t]", "нормированный официальный курс нужной валюты сегодня"],
        ["h=5", "пять следующих публикаций курса, а не обязательно пять календарных дней"],
        ["Попадание", "v[t] не выше каждого из v[t+1] ... v[t+5]"],
        ["Сигнал", "решение написать клиенту: сейчас момент выглядит выгодным"],
    ], styles, [42 * mm, 132 * mm])]
    story += [Spacer(1, 5 * mm), callout(
        "Пример: сегодня 10.00, следующие пять значений 10.10 / 10.08 / 10.15 / 10.20 / 10.12. "
        "Сегодняшний день - попадание. Если хотя бы одно будущее значение равно 9.95, попадания нет.",
        styles, PALE_BLUE, BLUE,
    )]
    story += [para("Что именно предсказываем", styles, "H2R3")]
    story += bullets([
        "Отдельный бинарный ответ для TJS, UZS, KGS, AMD и KZT.",
        "Главный внутренний target - fav_h5, но официальный scorecard также считается для h=1/3/5/10/20.",
        "Чем ниже курс иностранной валюты в рублях, тем больше валюты получит адресат за те же рубли.",
        "Точная численная траектория курса не является обязательным результатом кейса.",
    ], styles)
    story += [PageBreak()]

    story += heading("2. Почему вообще помогает рынок CNY", styles)
    story += [simple_story(), Spacer(1, 4 * mm)]
    story += bullets([
        "Официальный курс ЦБ похож на ценник, который обновляется один раз в сутки.",
        "Биржа похожа на живой оптовый рынок: цены CNY/RUB меняются каждые минуты.",
        "Если живой рынок уже ушел выше старого официального ценника, это признак ослабления рубля внутри дня.",
        "Тогда текущий официальный курс валюты получателя нередко еще дешевле тех курсов, которые появятся в следующих публикациях.",
        "У пяти коридоров общая рублевая сторона. Поэтому сильное движение рубля видно одновременно в TJS, UZS, KGS, AMD и KZT, хотя локальная динамика валют различается.",
    ], styles)
    story += [callout(
        "Это не магия и не прогноз нейросети. Лидер - объяснимое правило синхронизации двух часов: "
        "медленных официальных курсов и быстрого внутридневного рынка.",
        styles, PALE_GREEN, GREEN,
    )]
    story += [para("Важная оговорка", styles, "H2R3")]
    story += [para(
        "CNY - не единственная причина движения всех валют. Это ликвидный публичный индикатор общего состояния "
        "рубля. Поэтому сигнал иногда ошибается, а минимальный lift у KZT ниже, чем у остальных коридоров.", styles,
    ), PageBreak()]

    story += heading("3. Почему выбрано именно 15:30", styles)
    story += [market_timeline(), Spacer(1, 5 * mm)]
    story += [callout(
        "15:30 не выбирали перебором по лучшему результату 2025-2026. Это граница из официальной методики "
        "Банка России: для CNY/RUB TOM используется окно сделок 10:00 <= t < 15:30 по Москве.",
        styles, PALE_BLUE, BLUE,
    )]
    story += bullets([
        "На строке дня T допускаются только 10-минутные свечи, завершившиеся строго раньше T 15:30.",
        "Свеча с окончанием ровно в 15:30 и все более поздние данные исключены.",
        "Текущий уже действующий курс ЦБ известен; следующий эффективный курс в этом режиме не загружается.",
        "Физический тест умножает все свечи с cutoff и позже на 100 и проверяет, что прошлые признаки не изменились ни на бит.",
    ], styles)
    story += [para("Откуда могла возникнуть исследовательская предвзятость", styles, "H2R3")]
    story += [para(
        "Само время 15:30 взято из методики, а proxy и router выбирались на 2024. Но к моменту позднего раунда "
        "2025-2026 уже анализировались в других экспериментах. Поэтому результат причинный, но весь процесс "
        "исследования нельзя назвать новым pristine holdout.", styles,
    ), PageBreak()]

    story += heading("4. Что именно считается в 15:30", styles)
    story += [para("Шаг 1. Собираем цены", styles, "H2R3")]
    story += [para(
        "Берем все завершившиеся 10-минутные свечи CNYRUB_TOM между 10:00 и 15:30. В архиве нет пригодных "
        "исторических volume/value, поэтому настоящий VWAP восстановить нельзя. Мы честно используем простой "
        "детерминированный proxy.", styles,
    )]
    story += [para("Шаг 2. Получаем рыночный уровень", styles, "H2R3")]
    story += [callout(
        "market_level = геометрическое среднее цен закрытия завершившихся свечей. Все пять проверенных "
        "невзвешенных proxy дали одинаковый ranking; сохраненное имя победителя - geometric_mean_close.",
        styles, PALE_BLUE, BLUE,
    )]
    story += [para("Шаг 3. Сравниваем его с текущим ЦБ", styles, "H2R3")]
    story += [callout(
        "basis_bps = 10 000 x ln(market_level / current_CBR_CNY). Положительный basis означает: рынок CNY "
        "уже выше текущего официального уровня.",
        styles, PALE_ORANGE, ORANGE,
    )]
    story += [para("Игрушечный числовой пример", styles, "H2R3")]
    story += [table([
        ["Величина", "Пример", "Смысл"],
        ["Текущий CBR CNY", "11.50 RUB", "официальный ценник"],
        ["Средний рынок до 15:30", "11.62 RUB", "живой рынок уже выше"],
        ["Basis", "около +104 б.п.", "примерно +1.04% к текущему ЦБ"],
    ], styles, [55 * mm, 38 * mm, 81 * mm])]
    story += [para(
        "Числа 11.50 и 11.62 здесь учебные, а не конкретная строка датасета. Они показывают механику знака.",
        styles, "SmallR3",
    ), PageBreak()]

    story += heading("5. Как разные валюты получают сопоставимый score", styles)
    story += [para(
        "Один и тот же basis в 100 б.п. может быть обычным в бурный период и огромным в спокойный. Поэтому "
        "абсолютное число не сравнивается напрямую с вечным порогом.", styles,
    )]
    story += [table([
        ["Действие", "Что происходит", "Зачем"],
        ["Берем историю", "только предыдущие 250 score той же валюты", "никакого будущего"],
        ["Считаем percentile", "какую долю прошлых score текущий превышает", "масштаб 0...1"],
        ["Минимум истории", "до 20 прошлых наблюдений score не активен", "не делать вывод из пары дней"],
        ["Раздельно по валютам", "TJS сравнивается с прошлым TJS, KZT с прошлым KZT", "учесть разную волатильность"],
    ], styles, [38 * mm, 75 * mm, 61 * mm])]
    story += [Spacer(1, 5 * mm), callout(
        "Percentile 0.90 означает не 'курс вырастет с вероятностью 90%'. Он означает только: сегодняшний "
        "рыночный разрыв выше 90% недавних разрывов этой валюты.",
        styles, PALE_ORANGE, ORANGE,
    )]
    story += [para("Почему percentile полезен", styles, "H2R3")]
    story += bullets([
        "Не требует знать будущие значения для масштабирования.",
        "Автоматически приспосабливается к смене режима и величины колебаний.",
        "Позволяет применять одну коммуникационную политику ко всем пяти коридорам.",
    ], styles)
    story += [PageBreak()]

    story += heading("6. Зачем нужен availability-router", styles)
    story += [router_diagram(), Spacer(1, 4 * mm)]
    story += bullets([
        "В рабочий день с завершившимися CNY-свечами используем fixing score.",
        "В выходной, праздник или при пропуске рынка используем заранее замороженный noon-consensus.",
        "Router смотрит только на наличие рыночной строки. Он не знает target и не выбирает модель по будущему исходу.",
        "Ни один целевой день не удаляется. Поэтому база случайного дня остается сопоставимой.",
    ], styles)
    story += [callout(
        "Детская аналогия: если школьный термометр работает, читаем его. Если школа закрыта, берем прогноз "
        "с резервной станции. Мы не выбрасываем день только потому, что основной датчик молчит.",
        styles, PALE_GREEN, GREEN,
    )]
    story += [para("Что такое noon-consensus", styles, "H2R3")]
    story += [para(
        "Это более ранний, уже зафиксированный независимый score из доступных к полудню рыночных и "
        "кросс-валютных представлений. В данном подходе он важен как fallback; его параметры не выбираются "
        "заново по пропущенным дням.", styles,
    ), PageBreak()]

    story += heading("7. Как из score получается редкий сигнал", styles)
    story += [table([
        ["Параметр", "Значение", "Простой смысл"],
        ["rate", "22%", "срабатывают примерно самые сильные 22% недавних score"],
        ["rolling", "20", "порог считается по 20 предыдущим score валюты"],
        ["cooldown", "0", "отдельной паузы нет; редкость дает threshold"],
        ["target cadence", "1-2 / неделю", "проверяется после расчета, а не подгоняется на тесте"],
    ], styles, [42 * mm, 32 * mm, 100 * mm])]
    story += [Spacer(1, 5 * mm), para(
        "Для каждого нового дня алгоритм сначала видит только историю score. Он находит границу верхних 22% "
        "среди предыдущих 20 значений и затем сравнивает с ней сегодняшний score. Текущий исход добавится в "
        "историю только позже, когда станет известен.", styles,
    )]
    story += [callout(
        "Важно: 22% - не вероятность успеха и не обещание клиенту. Это регулятор редкости, чтобы сильные "
        "события не превращались в ежедневный спам.",
        styles, PALE_ORANGE, ORANGE,
    )]
    story += [para("Почему получилось 1.19, а не ровно 1.54", styles, "H2R3")]
    story += bullets([
        "22% применяется к причинному скользящему окну, поэтому фактическая доля меняется вместе с режимом.",
        "В конце ряда некоторые h=5 исходы еще не разрешены и не входят в оценку.",
        "Даты публикаций и пропущенные торговые сессии распределены неравномерно.",
    ], styles)
    story += [PageBreak()]

    story += heading("8. Откуда взялось число lift 2.059", styles)
    story += [lift_math(), Spacer(1, 4 * mm)]
    story += [table([
        ["Шаг", "Число 2025-2026", "Что оно означает"],
        ["Оценочный scope", "2020 строк", "все доступные h=5 строки пяти валют"],
        ["Сигналы", "506", "дни, выбранные router и threshold"],
        ["Успешные сигналы", "279", "сегодняшний курс не стал хуже в следующих 5 публикациях"],
        ["Hit rate сигналов", "279 / 506 = 55.14%", "точность выбранных дней"],
        ["Base rate", "26.78%", "точность случайного дня в том же scope"],
        ["Pooled lift", "55.14 / 26.78 = 2.059", "примерно в 2.06 раза лучше случайного дня"],
    ], styles, [44 * mm, 52 * mm, 78 * mm])]
    story += [Spacer(1, 5 * mm), callout(
        "Lift 2.059 не означает +105.9% доходности и не означает вероятность 205.9%. Это отношение двух "
        "вероятностей попадания.",
        styles, PALE_RED, RED,
    )]
    story += [para("Почему рядом встречается 2.053", styles, "H2R3")]
    story += [para(
        "2.059 - простой pooled lift по всем строкам. В официальном многогоризонтном scorecard используется "
        "corridor-period adjusted aggregation; для h=5 она дает 2.053. Разница около 0.006 возникает только "
        "из-за способа усреднения, а не из-за другой модели.", styles,
    ), PageBreak()]

    story += heading("9. Что получилось на всех горизонтах", styles)
    story += [horizon_bars([
        ("h=1", 1.780, GREEN),
        ("h=3", 2.014, BLUE),
        ("h=5", 2.053, BLUE),
        ("h=10", 2.005, ORANGE),
        ("h=20", 1.928, ORANGE),
    ]), Spacer(1, 5 * mm)]
    story += [table([
        ["Горизонт", "Adjusted lift", "Сигналов", "Future-only выгода"],
        ["1", "1.780", "506", "+77.6 б.п."],
        ["3", "2.014", "506", "+93.5 б.п."],
        ["5", "2.053", "506", "+95.2 б.п."],
        ["10", "2.005", "501", "+90.6 б.п."],
        ["20", "1.928", "486", "+79.2 б.п."],
    ], styles, [36 * mm, 43 * mm, 38 * mm, 57 * mm])]
    story += [callout(
        "Минимум по пяти официальным горизонтам равен 1.780. Мы не выбираем самый красивый h после "
        "просмотра: даже слабейший горизонт выше целевого 1.30.",
        styles, PALE_GREEN, GREEN,
    )]
    story += [PageBreak()]

    story += heading("10. Проверка по годам и валютам", styles)
    story += [table([
        ["Период h=5", "Частота", "Lift", "Future-only выгода"],
        ["2025", "1.187", "2.096", "+73.2 б.п."],
        ["2026", "1.276", "1.994", "+127.6 б.п."],
        ["2025-2026", "1.195", "2.059 pooled", "+95.2 б.п."],
    ], styles, [52 * mm, 38 * mm, 38 * mm, 46 * mm])]
    story += [Spacer(1, 5 * mm), table([
        ["Валюта", "Сигналов", "Частота", "h=5 lift", "Future benefit"],
        ["TJS", "96", "1.133", "1.948", "+94.9"],
        ["UZS", "100", "1.180", "2.113", "+100.1"],
        ["KGS", "107", "1.263", "2.235", "+92.1"],
        ["AMD", "102", "1.204", "2.205", "+99.0"],
        ["KZT", "101", "1.192", "1.832", "+90.0"],
    ], styles, [31 * mm, 34 * mm, 36 * mm, 36 * mm, 37 * mm])]
    story += [Spacer(1, 5 * mm), callout(
        "Каждый год и каждая валюта проходят lift 1.30. Самое слабое место - KZT с 1.832; это все равно "
        "существенно выше порога.",
        styles, PALE_BLUE, BLUE,
    )]
    story += [para("Частота во времени", styles, "H2R3")]
    story += [para(
        "Самый тихий квартал - 2025Q2: 0.955 сигнала на валюту-неделю и lift 1.405. Средняя частота ТЗ "
        "проходит, но обещать строго не менее одного сигнала в каждом квартале нельзя.", styles,
    ), PageBreak()]

    story += heading("11. Почему мы считаем эффект настоящим", styles)
    story += [table([
        ["Проверка", "Результат", "Зачем она нужна"],
        ["4-недельный block bootstrap", "h=5 CI [1.727; 2.434]", "учесть зависимость соседних дней"],
        ["p(lift <= 1)", "0.00025", "случайное качество крайне маловероятно"],
        ["Circular-shift max test", "null q95 = 1.274", "поправка на пять записанных политик"],
        ["Fresh vs stale-20", "minimum gain +1.025; CI [0.674; 1.311]", "проверить ценность свежей информации"],
        ["Router vs noon", "+0.066; CI [-0.123; +0.238]", "добавочный gain router пока не доказан"],
    ], styles, [47 * mm, 56 * mm, 71 * mm], small=True)]
    story += [Spacer(1, 5 * mm), callout(
        "Самое сильное доказательство механизма: если искусственно состарить только CNY-часть на 20 строк, "
        "h=5 lift падает примерно с 2.059 до 0.989. Значит, решает именно свежая рыночная информация.",
        styles, PALE_GREEN, GREEN,
    )]
    story += [para("Что еще не доказано", styles, "H2R3")]
    story += bullets([
        "Что router статистически лучше сильного noon-consensus, а не просто удачнее на этом блоке.",
        "Что тот же результат сохранится на полностью новом периоде, который никто не видел при генерации гипотез.",
        "Что курс приложения банка повторяет курс ЦБ и дает клиенту ровно такую же экономию.",
    ], styles)
    story += [PageBreak()]

    story += heading("12. Где именно исключено будущее", styles)
    story += [table([
        ["Возможная утечка", "Защита"],
        ["Свеча 15:30 и позднее", "берутся только candle end < T 15:30"],
        ["Следующий курс ЦБ", "не загружается в pre-publication score"],
        ["Будущая шкала score", "percentile только по предыдущим 250 строкам"],
        ["Текущий threshold", "rolling cutoff только по прошлым 20 score"],
        ["Пропущенный день", "availability fallback, а не удаление после просмотра outcome"],
        ["Будущая метка", "используется только для оценки; в признаки не попадает"],
    ], styles, [63 * mm, 111 * mm])]
    story += [Spacer(1, 5 * mm), para("Физические тесты", styles, "H2R3")]
    story += bullets([
        "Изменяем все рыночные значения после cutoff и требуем полного совпадения более ранних признаков.",
        "Проверяем равенство test_idx у router и fallback, чтобы пропуски не меняли scope.",
        "Проверяем, что causal percentile не может переписать прошлое будущим score.",
        "Полный набор репозитория: 109 correctness, as-of и leakage тестов прошел.",
    ], styles)
    story += [callout(
        "Будущее разрешено только в target после завершения эксперимента. Иначе невозможно проверить, был "
        "ли старый сигнал правильным. Запрет относится к информации, которой располагал алгоритм в момент решения.",
        styles, PALE_ORANGE, ORANGE,
    )]
    story += [PageBreak()]

    story += heading("13. Но можно ли использовать курс ЦБ на завтра?", styles)
    story += [information_modes(), Spacer(1, 4 * mm)]
    story += [callout(
        "Да - если Банк России уже фактически опубликовал его к моменту сигнала. Нет - если мы хотим отправить "
        "сигнал раньше публикации. Правило кейса: на дату T разрешено все, что реально доступно на T.",
        styles, PALE_GREEN, GREEN,
    )]
    story += [para("Что прямо сказано в материалах кейса", styles, "H2R3")]
    story += bullets([
        "В описании данных: курс на завтра публикуется сегодня, поэтому сигнал опирается на последний опубликованный курс.",
        "В Q&A: индексирование по дате публикации и отсчет горизонта с T+1 разрешено как явно описанное допущение команды.",
        "Момент отправки является частью решения команды; гранулярность не фиксируется.",
        "Дисквалифицируется не слово 'завтра', а любое значение, которое еще не было доступно в фактический timestamp решения.",
    ], styles)
    story += [para("Почему мы раньше запрещали его", styles, "H2R3")]
    story += [para(
        "Это было сознательное более строгое исследовательское ограничение: доказать, что модель умеет находить "
        "момент еще до следующей публикации ЦБ. Оно делает эксперимент труднее, но не является запретом кейсодателя.",
        styles,
    ), PageBreak()]

    story += heading("14. Лучший вариант после публикации", styles)
    story += [para(
        "Если продукт готов отправлять пуш только после появления нового значения ЦБ, самый сильный честно "
        "отделенный кандидат - pub_extra_7y. Он использует уже опубликованный следующий эффективный курс и "
        "пытается понять, сохранится ли хороший момент еще на оставшихся публикациях горизонта h=5.", styles,
    )]
    story += [table([
        ["Метрика 2024-2026", "Значение"],
        ["Сигналов", "732"],
        ["Частота", "1.069 на валюту-неделю"],
        ["Base rate", "29.45%"],
        ["Hit rate", "72.40%"],
        ["Lift", "2.459"],
        ["Future-only выгода", "+138.3 б.п."],
        ["95% block-bootstrap CI", "[2.160; 2.797]"],
        ["Минимальный lift валюты", "2.371"],
    ], styles, [80 * mm, 94 * mm])]
    story += [Spacer(1, 5 * mm), para("Почему он настолько силен", styles, "H2R3")]
    story += bullets([
        "Если уже объявленный v[t+1] ниже v[t], target fav_h5 точно невозможен: первый будущий шаг уже проигран.",
        "Если v[t+1] выше v[t], известный запас показывает, насколько остальные четыре курса могут откатиться.",
        "Простой gate v[t+1] >= v[t] уже дает lift около 1.959; ML уточняет силу запаса и режим рынка.",
        "Главный признак - known_margin_vol: объявленный запас, деленный на недавнюю волатильность.",
    ], styles)
    story += [PageBreak()]

    story += heading("15. Это не чит, если поставить правильные часы", styles)
    story += [table([
        ["Ситуация", "Вердикт", "Почему"],
        ["Сигнал в 15:30, а значение опубликовано позже", "УТЕЧКА", "информации еще нет"],
        ["Сигнал после фактического события публикации", "РАЗРЕШЕНО", "новое значение уже публично"],
        ["Исторически считаем, будто публикация всегда была в 18:00", "РИСК", "точное время не регламентировано"],
        ["Live-сервис проверяет смену значения и записывает timestamp", "ПРАВИЛЬНО", "availability доказуема"],
        ["Пуш обещает, что курс вырастет", "НЕЛЬЗЯ", "комплаенс запрещает утверждения о будущем"],
        ["Пуш сообщает факт о текущем положении", "МОЖНО", "вывод о будущем делает клиент"],
    ], styles, [57 * mm, 33 * mm, 84 * mm], small=True)]
    story += [Spacer(1, 5 * mm), callout(
        "Технически самый важный production-признак - не сам курс, а published_at. Если timestamp отсутствует, "
        "post-publication модель должна быть заблокирована и система остается в pre-publication режиме.",
        styles, PALE_RED, RED,
    )]
    story += [para("Историческая оговорка", styles, "H2R3")]
    story += [para(
        "Архив курсов надежно хранит значения и даты действия, но точное историческое время публикации может быть "
        "недоступно. Поэтому post-publication backtest доказывает условие 'если курс уже опубликован', а не то, "
        "что одинаковый пуш можно было отправить в фиксированную минуту каждого исторического дня.", styles,
    ), PageBreak()]

    story += heading("16. Какой вариант рекомендовать команде", styles)
    story += [table([
        ["Продуктовый режим", "Основной score", "Когда отправлять", "Headline"],
        ["Ранний сигнал", "15:30 availability-router", "в 15:30 до нового ЦБ", "h5 2.059; rate 1.19"],
        ["Максимальная точность", "pub_extra_7y", "сразу после фактической публикации", "h5 2.459; rate 1.07"],
        ["Резерв", "noon-consensus", "если нужен более ранний cutoff", "h5 около 1.995"],
    ], styles, [39 * mm, 48 * mm, 52 * mm, 35 * mm], small=True)]
    story += [Spacer(1, 5 * mm), para("Рекомендуемая логика сервиса", styles, "H2R3")]
    story += bullets([
        "До фактической публикации: работает 15:30 router, следующий курс в features отсутствует.",
        "Появилось новое значение и записан timestamp: разрешается post-publication модель.",
        "Не появилось или timestamp сомнителен: post-publication контур не запускается.",
        "Пуш формулируется как факт о настоящем и прошлом, без обещания будущего курса.",
        "На защите две цифры показываются отдельно, потому что их information set различается.",
    ], styles)
    story += [callout(
        "Самая честная формулировка: 'До публикации мы получили lift 2.059. После публикации, когда курс на "
        "завтра уже является публичным фактом, отдельная модель получила lift 2.459. Эти режимы нельзя смешивать.'",
        styles, PALE_GREEN, GREEN,
    )]
    story += [PageBreak()]

    story += heading("17. Как пересказать подход за одну минуту", styles)
    story += [callout(
        "ЦБ обновляет официальный курс раз в сутки, а рынок CNY/RUB движется постоянно. В 15:30 мы сравниваем "
        "средний биржевой уровень CNY за официальное расчетное окно с еще действующим курсом ЦБ. Если разрыв "
        "необычно большой относительно предыдущих наблюдений, сегодняшний официальный курс пяти валютных "
        "коридоров часто оказывается хорошим относительно следующих публикаций. В выходной берем независимый "
        "резервный noon-score. Порог считается только по прошлым score и дает около 1.19 сигнала в неделю. "
        "На 2025-2026 успешны 279 из 506 сигналов против базовых 26.78%, поэтому lift равен 2.059.",
        styles, PALE_BLUE, BLUE,
    )]
    story += [para("Если спросят про курс на завтра", styles, "H2R3")]
    story += [callout(
        "Кейс его не запрещает после публикации. Запрещено пользоваться им раньше, чем он появился. Наш 15:30 "
        "результат специально построен без него; отдельный post-publication результат 2.459 использует его "
        "легально при наличии timestamp gate.",
        styles, PALE_ORANGE, ORANGE,
    )]
    story += [para("Если спросят, почему не нейросеть", styles, "H2R3")]
    story += [para(
        "Здесь выиграла не большая модель, а свежая информация, напрямую связанная с механизмом формирования "
        "следующего курса. Простое правило легче объяснить, проверить на утечку и превратить в факт для пуша.", styles,
    )]
    story += [PageBreak()]

    story += heading("18. Источники и воспроизводимость", styles)
    story += [table([
        ["Источник", "Что подтверждает"],
        ["Страница кейса Talent Track /cases/455", "информационный cutoff, метрики, frequency, walk-forward"],
        ["Q&A 04.09 и 05.09, стр. 3-6", "курс на завтра публикуется сегодня; допущение по индексации"],
        ["Методика ЦБ 6290-У", "окно CNYRUB_TOM 10:00 <= t < 15:30"],
        ["results/.../fixing_availability_router", "числа 15:30 leader и breakdown"],
        ["results/research/round4", "отдельная post-publication модель"],
        ["tests/test_round6_research.py", "физические cutoff и causality checks"],
    ], styles, [70 * mm, 104 * mm], small=True)]
    story += [Spacer(1, 5 * mm), para("Ключевые файлы", styles, "H2R3")]
    story += bullets([
        "research/round6_fixing_availability_router.py",
        "research/round6_fixing_proxies.py",
        "research/round6_moex_spot_1530_features.py",
        "research/round4_research.py",
        "results/research/round6/fixing_availability_router/later_by_horizon.csv",
        "results/research/round6/fixing_availability_router/standard_h5_results.csv",
        "results/research/round4/headline_summary.csv",
    ], styles)
    story += [callout(
        "Итог: лучший объяснимый pre-publication сигнал строится на свежем CNY market-to-CBR basis. "
        "Лучший доступный результат по буквальным правилам кейса может использовать уже опубликованный курс на "
        "завтра, но только как отдельный post-publication продукт с проверяемым timestamp.",
        styles, PALE_GREEN, GREEN,
    )]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return PDF


if __name__ == "__main__":
    print(build())
