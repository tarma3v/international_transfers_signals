"""Build the consolidated PDF across main, version_b and ivan-experiments."""
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
PDF = ROOT / "output" / "pdf" / "international_transfers_full_research_report.pdf"


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LIGHT)
    canvas.line(20 * mm, 14 * mm, 190 * mm, 14 * mm)
    canvas.setFont("Arial", 7)
    canvas.setFillColor(GRAY)
    canvas.drawString(20 * mm, 9 * mm, "International transfers signals | consolidated research")
    canvas.drawRightString(190 * mm, 9 * mm, f"04.09.2026 | {doc.page}")
    canvas.restoreState()


def metric_cards(styles):
    cells = [
        [para("1.295", styles, "MetricR3"), para("1.307", styles, "MetricR3"),
         para("2.459", styles, "MetricR3")],
        [para("locked past-only", styles, "MetricSmallR3"),
         para("version_b causal challenger", styles, "MetricSmallR3"),
         para("after-publication selected", styles, "MetricSmallR3")],
    ]
    result = Table(cells, colWidths=[58 * mm] * 3, rowHeights=[16 * mm, 11 * mm])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 1, BLUE),
        ("INNERGRID", (0, 0), (-1, -1), .4, colors.HexColor("#bfdbfe")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return result


def branch_flow(styles):
    cells = [[
        para("main<br/><font size='6'>продукт, базовые правила и ML</font>", styles, "CalloutR3"),
        para("version_b<br/><font size='6'>rolling logit benchmark</font>", styles, "CalloutR3"),
        para("ivan-experiments<br/><font size='6'>deep research rounds 1-4</font>", styles, "CalloutR3"),
    ]]
    result = Table(cells, colWidths=[56 * mm, 56 * mm, 62 * mm], rowHeights=[26 * mm])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), PALE_GREEN),
        ("BACKGROUND", (1, 0), (1, 0), PALE_ORANGE),
        ("BACKGROUND", (2, 0), (2, 0), PALE_BLUE),
        ("BOX", (0, 0), (0, 0), 1, GREEN),
        ("BOX", (1, 0), (1, 0), 1, ORANGE),
        ("BOX", (2, 0), (2, 0), 1, BLUE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    return result


def bar_chart(items, *, maximum=2.8, target=1.3, width=174 * mm, height=92 * mm):
    """Compact horizontal lift chart with an explicit threshold line."""
    d = Drawing(width, height)
    left, right, top, bottom = 48 * mm, 9 * mm, 6 * mm, 10 * mm
    chart_w = width - left - right
    chart_h = height - top - bottom
    row_h = chart_h / len(items)
    x = left + chart_w * target / maximum
    d.add(Line(x, bottom - 2, x, height - top + 1, strokeColor=RED, strokeWidth=1))
    d.add(String(x + 2, height - top - 2, "threshold 1.30", fontName="Arial", fontSize=6.5,
                 fillColor=RED))
    for tick in (0, 1, 2):
        tx = left + chart_w * tick / maximum
        d.add(Line(tx, bottom - 2, tx, height - top, strokeColor=LIGHT, strokeWidth=.4))
        d.add(String(tx - 3, 1, str(tick), fontName="Arial", fontSize=6, fillColor=GRAY))
    for i, (label, value, color) in enumerate(items):
        y = height - top - (i + .72) * row_h
        d.add(String(1, y + 1, label, fontName="Arial", fontSize=6.7, fillColor=NAVY))
        d.add(Rect(left, y, chart_w * value / maximum, row_h * .48,
                   fillColor=color, strokeColor=None))
        d.add(String(left + chart_w * value / maximum + 3, y + 1, f"{value:.3f}",
                     fontName="Arial-Bold", fontSize=6.7, fillColor=NAVY))
    return d


def stability_chart(width=174 * mm, height=82 * mm):
    years = [2024, 2025, 2026]
    series = [
        ("Locked anchor", [1.251, .508, 1.172], BLUE),
        ("Reset XGB", [1.290, 1.188, 1.437], GREEN),
        ("Online Hedge", [1.288, 1.043, 1.422], ORANGE),
        ("version_b fixed q20", [1.133, 1.472, 1.822], colors.HexColor("#7c3aed")),
        ("Post-pub ExtraTrees", [2.268, 2.754, 2.288], RED),
    ]
    d = Drawing(width, height)
    left, right, top, bottom = 21 * mm, 8 * mm, 8 * mm, 14 * mm
    chart_w, chart_h = width - left - right, height - top - bottom
    ymin, ymax = .4, 2.9
    for value in (.5, 1.0, 1.3, 2.0, 2.5):
        y = bottom + chart_h * (value - ymin) / (ymax - ymin)
        d.add(Line(left, y, width - right, y,
                   strokeColor=RED if value == 1.3 else LIGHT,
                   strokeWidth=1 if value == 1.3 else .4))
        d.add(String(1, y - 2, f"{value:.1f}", fontName="Arial", fontSize=6, fillColor=GRAY))
    xs = [left + chart_w * i / 2 for i in range(3)]
    for x, year in zip(xs, years):
        d.add(String(x - 9, 2, str(year), fontName="Arial", fontSize=7, fillColor=NAVY))
    for label, values, color in series:
        points = []
        for x, value in zip(xs, values):
            y = bottom + chart_h * (value - ymin) / (ymax - ymin)
            points.append((x, y))
            d.add(Circle(x, y, 2.2, fillColor=color, strokeColor=colors.white, strokeWidth=.5))
        for a, b in zip(points[:-1], points[1:]):
            d.add(Line(a[0], a[1], b[0], b[1], strokeColor=color, strokeWidth=1.6))
    legend_y = height - 5
    legend_x = left
    for i, (label, _, color) in enumerate(series):
        x = legend_x + (i % 3) * 52 * mm
        y = legend_y - (i // 3) * 8 * mm
        d.add(Rect(x, y - 2, 9, 4, fillColor=color, strokeColor=None))
        d.add(String(x + 12, y - 3, label, fontName="Arial", fontSize=6.2, fillColor=NAVY))
    return d


def architecture_diagram(width=174 * mm, height=67 * mm):
    d = Drawing(width, height)
    box_w, box_h = 35 * mm, 18 * mm
    xs = [2 * mm, 47 * mm, 92 * mm, 137 * mm]
    labels = [
        ("CBR publications", "five corridors"),
        ("Past-only features", "range / trend / vol"),
        ("Frozen scorers", "anchor / logit / XGB"),
        ("Alert policy", "causal threshold"),
    ]
    fills = [PALE_BLUE, PALE_GREEN, PALE_ORANGE, PALE_BLUE]
    borders = [BLUE, GREEN, ORANGE, BLUE]
    y = 35 * mm
    for i, (x, (title, sub)) in enumerate(zip(xs, labels)):
        d.add(Rect(x, y, box_w, box_h, rx=4, ry=4, fillColor=fills[i],
                   strokeColor=borders[i], strokeWidth=1))
        d.add(String(x + 4, y + 11 * mm, title, fontName="Arial-Bold", fontSize=7,
                     fillColor=NAVY))
        d.add(String(x + 4, y + 5 * mm, sub, fontName="Arial", fontSize=6,
                     fillColor=GRAY))
        if i < 3:
            d.add(Line(x + box_w, y + box_h / 2, xs[i + 1] - 3, y + box_h / 2,
                       strokeColor=GRAY, strokeWidth=1))
    d.add(Rect(47 * mm, 2 * mm, 80 * mm, 19 * mm, rx=4, ry=4,
               fillColor=PALE_RED, strokeColor=RED, strokeWidth=1))
    d.add(String(51 * mm, 14 * mm, "Separate post-publication contour",
                 fontName="Arial-Bold", fontSize=7, fillColor=NAVY))
    d.add(String(51 * mm, 7 * mm, "known v[t+1] + runtime timestamp gate",
                 fontName="Arial", fontSize=6.2, fillColor=GRAY))
    return d


def build() -> Path:
    register_fonts()
    s = get_styles()
    PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=18 * mm,
        title="Полный отчёт по прогнозированию выгодного курса",
        author="international_transfers_signals",
        subject="Consolidated research across main, version_b and ivan-experiments",
    )
    story = []

    # 1. Cover
    story += [
        Spacer(1, 13 * mm),
        para("Полный отчёт по прогнозированию выгодного курса", s, "TitleR3"),
        para("Данные, постановка, все полезные подходы, честность метрик и итоговая стратегия", s, "SubtitleR3"),
        metric_cards(s), Spacer(1, 8 * mm),
        callout(
            "Главная картина: до публикации следующего курса честный locked benchmark даёт "
            "lift 1.295; logistic regression из version_b после исправления порога остаётся "
            "интересным recent-regime challenger около 1.307; сильнейший отдельный продуктовый "
            "сценарий после публикации следующего курса даёт 2.459.",
            s, PALE_GREEN, GREEN,
        ), Spacer(1, 7 * mm), branch_flow(s), Spacer(1, 8 * mm),
        para("Ветка отчёта: ivan-experiments | состояние данных: 04.09.2026", s, "SmallR3"),
        para("Все новые оценки на 2024-2026 ретроспективны: этот период уже изучался.", s, "SmallR3"),
        PageBreak(),
    ]

    # 2. Executive view
    story += heading("1. Ответ без прикрас", s)
    story += [callout(
        "У нас нет доказанного универсального past-only решения, которое стабильно держит "
        "lift выше 1.30 во всех эпохах при 1-2 сигналах на валюту в неделю. Есть несколько "
        "сильных challengers и один очень сильный условный сценарий после публикации.",
        s, PALE_ORANGE, ORANGE,
    )]
    story += [para("Что считать лучшим сейчас", s, "H2R3"), table([
        ["Вопрос", "Ответ", "Lift / частота", "Источник", "Статус"],
        ["Честный baseline до публикации", "Multiscale range anchor", "1.295 / 1.02", "ivan-experiments", "locked"],
        ["Объяснимый recent challenger", "Causal logistic regression", "1.307 / 1.11", "version_b + audit", "заморозить"],
        ["Ровный post-2022 ML", "Reset XGBoost", "1.288 / 1.07", "ivan-experiments", "challenger"],
        ["Максимум past-only", "Online local Hedge", "1.434 / 1.63", "ivan-experiments", "retro, нестабилен"],
        ["После публикации", "Conditional ExtraTrees", "2.459 / 1.07", "ivan-experiments", "selected"],
        ["Окно закрывается", "Upper-range rule", "1.182 / 1.46", "ivan-experiments", "не проходит"],
    ], s, [39*mm, 39*mm, 29*mm, 34*mm, 33*mm], small=True)]
    story += [Spacer(1, 5 * mm), para("Практическое решение", s, "H2R3")]
    story += bullets([
        "Если сигнал разрешён только до следующей публикации: anchor остаётся benchmark; causal logit и reset XGB работают параллельно как challengers.",
        "Если после публикации завтрашнего курса ещё доступна операция по сегодняшнему: conditional ExtraTrees становится основным кандидатом.",
        "Никакую новую цифру на 2024-2026 больше не использовать для настройки; подтверждение возможно только на новых публикациях.",
        "Lift описывает вероятность удачного момента, а не процент доходности. Денежная выгода считается отдельно в базисных пунктах.",
    ], s)
    story += [PageBreak()]

    # 3. Task and data
    story += heading("2. Что именно предсказывается", s)
    story += [para(
        "Для TJS, UZS, KGS, AMD и KZT используется нормированный официальный курс ЦБ: "
        "рубли за одну единицу валюты получателя. Меньшее значение выгоднее отправителю рублей.", s)]
    story += [callout(
        "fav_h5(t) = 1, если v[t] <= min(v[t+1], ..., v[t+5]). Иными словами, сегодня "
        "не будет побито более низким курсом в следующих пяти публикациях ЦБ.", s, PALE_BLUE, BLUE)]
    story += [para("Данные", s, "H2R3"), table([
        ["Элемент", "Содержание"],
        ["Основные ряды", "TJS, UZS, KGS, AMD, KZT"],
        ["Рыночный контекст", "USD и CNY; в отдельных опытах RUONIA, ставка, Brent, broad USD"],
        ["Длинная история", "2010-01-01 - 2026-09-02; около 4.1 тыс. публикаций на валюту"],
        ["Единица времени", "публикация ЦБ, не календарный день"],
        ["Пропущенные дни", "выходные не заполняются; calendar gap хранится отдельным признаком"],
        ["Частота продукта", "целевой коридор 1-2 сигнала на валюту в неделю"],
    ], s, [47*mm, 127*mm])]
    story += [para("Как считается lift", s, "H2R3"), para(
        "Lift = hit rate среди сигналов / base rate среди всех допустимых тестовых строк. "
        "При случайном выборе того же числа дней ожидаемый lift равен 1. Например, post-publication "
        "ExtraTrees на 2024-2026 имеет base rate 29.45%, hit rate 72.40% и lift 2.459.", s)]
    story += [para("Дополнительные метрики", s, "H2R3")]
    story += bullets([
        "Future-only benefit: достижимое преимущество сегодняшнего курса относительно следующих пяти публикаций.",
        "Симметричная +/-5 выгода: локальность минимума с учётом уже прошедших значений; это не то же самое, что доступная клиенту выгода.",
        "Минимумы lift по годам и валютам, macro-year lift и фактическая частота сигналов.",
    ], s)
    story += [PageBreak()]

    # 4. Protocol
    story += heading("3. Что означает честный тест", s)
    story += [table([
        ["Этап", "Роль", "Что запрещено"],
        ["Train", "обучение параметров", "labels, не успевшие разрешиться до границы"],
        ["Calibration", "шкала score и порог", "scores или target будущего test"],
        ["Test", "только применение", "подбор модели, окна, порога или частоты"],
        ["Retrospective audit", "диагностика гипотез", "называть результат новым holdout"],
    ], s, [36*mm, 62*mm, 76*mm])]
    story += [Spacer(1, 5 * mm), para("Замороженная хронология ivan-experiments", s, "H2R3")]
    story += bullets([
        "Development до 2016; general validation 2017-2020; transition/calibration 2021; shock validation 2022-2023; retrospective final 2024-2026.",
        "Для каждого тестового года порог строится по предыдущему году отдельно для каждой валюты.",
        "h=5 строка допускается в обучение только после фактической даты пятой будущей публикации.",
        "Rolling-порог использует только предыдущие scores и обязательный shift(1).",
        "Четырёхнедельный block bootstrap сохраняет перекрытие targets и общие движения валют.",
    ], s)
    story += [callout(
        "Причинный прогноз ещё не гарантирует честный backtest. Можно не передавать будущие курсы "
        "в модель, но всё равно подсмотреть в test при выборе порога, гиперпараметров или победителя.",
        s, PALE_RED, RED,
    ), Spacer(1, 5 * mm)]
    story += [table([
        ["Маркер в отчёте", "Смысл"],
        ["locked / selected earlier", "конфигурация выбрана до оцениваемого блока"],
        ["causal retrospective", "каждый сигнал воспроизводим онлайн, но период уже просмотрен"],
        ["posthoc / retro max", "гипотеза выбрана после просмотра результата"],
        ["non-causal threshold", "решение использует будущую шкалу test scores"],
    ], s, [52*mm, 122*mm])]
    story += [PageBreak()]

    # 5. Provenance
    story += heading("4. Карта веток и происхождение результатов", s)
    story += [branch_flow(s), Spacer(1, 7 * mm), table([
        ["Ветка", "Вклад", "Ключевые результаты", "Как читать"],
        ["main", "продуктовая постановка, базовые правила, Logit/RF/GB/Cat/XGB, интерфейс", "h5 logit 1.14; upper-range 1.39 при freq 0.60", "отдельный ранний протокол"],
        ["version_b", "36m rolling fit, 6m test, расширенные признаки, logistic regression", "заявлено mean lift 1.450 через future test top-15%", "rank diagnostic, не online policy"],
        ["ivan-experiments", "2010-2026, leakage audits, режимы, ансамбли, внешние данные, state и post-publication модели", "locked 1.295; post-publication 2.459", "основной research protocol"],
        ["version_b + ivan audit", "та же logit, но причинные пороги и long-history test", "recent 1.307-1.347; long 1.085-1.225", "causal, но retrospective"],
    ], s, [31*mm, 55*mm, 53*mm, 35*mm], small=True)]
    story += [Spacer(1, 5 * mm), callout(
        "Цифры из разных веток нельзя ранжировать как одну leaderboard без оговорок: "
        "различаются начало истории, длина train, способ calibration и агрегирование валют.",
        s, PALE_ORANGE, ORANGE,
    )]
    story += [para("Авторы веток по git history", s, "H2R3")]
    story += bullets([
        "main и ivan-experiments: Aleksandr Tarmaev.",
        "version_b: Daniil Nedaiborsch; audited commit aa44f10.",
        "Этот PDF собирает результаты, но не переносит код version_b в main и не скрывает исходную ветку.",
    ], s)
    story += [PageBreak()]

    # 6. Data findings
    story += heading("5. Что полезного нашли в данных", s)
    story += [table([
        ["Наблюдение", "Факт", "Следствие для модели"],
        ["Общий валютный фактор", "PC1 движений объясняет ~70% в 2017-2020, 85% в 2022-2023, 92% в 2024-2026", "глобальное обучение по пяти валютам оправдано"],
        ["Смена режима после 2022", "тренды и волатильность резко отличаются", "rolling/decay/reset полезнее бинарной даты"],
        ["Сезонность", "порядок сильных месяцев меняется между эпохами и валютами", "sin/cos и праздники только слабые covariates"],
        ["Пропуски", "это календарные выходные, не missing observation", "не размножать forward-filled labels"],
        ["Перекрытие target", "соседние h=5 labels имеют четыре общих будущих точки", "нужен block, а не iid bootstrap"],
        ["Общий спрос", "истории переводов и клиентского спроса нет", "нельзя доказать эффект SWIFT или бизнес uplift спроса"],
    ], s, [42*mm, 68*mm, 64*mm], small=True)]
    story += [para("Самые сильные past-only признаки", s, "H2R3")]
    story += bullets([
        "Положение текущего курса в прошлых диапазонах 30/90/180 публикаций.",
        "Доходности и тренды 5/20/60, расстояния до прошлых минимумов и максимумов.",
        "Короткая и длинная волатильность, их отношение, streak up/down.",
        "Идентификатор валюты, общий фактор пяти валют и индивидуальное отклонение от него.",
        "Past-only USD/CNY контекст; календарный gap; слабые циклические признаки дня недели и года.",
    ], s)
    story += [callout(
        "Главная фича оказалась простой: pct_range_90 не смотрит вперёд. Она отвечает, "
        "насколько низко сегодняшний курс расположен между прошлым минимумом и максимумом.",
        s, PALE_GREEN, GREEN,
    )]
    story += [PageBreak()]

    # 7. Past-only leaderboard
    story += heading("6. Past-only: карта лучших результатов", s)
    story += [bar_chart([
        ("Locked anchor [ivan]", 1.295, BLUE),
        ("Causal logit [version_b audit]", 1.307, colors.HexColor("#7c3aed")),
        ("Reset XGB [ivan]", 1.288, GREEN),
        ("Geometric consensus [ivan]", 1.264, GREEN),
        ("Soft router [ivan, retro]", 1.325, ORANGE),
        ("Trend anchor [ivan, posthoc]", 1.406, ORANGE),
        ("Online Hedge [ivan, retro]", 1.434, RED),
        ("Post-publication [ivan]", 2.459, RED),
    ])]
    story += [para("Сводка для h=5; результаты 2024-2026", s, "H2R3"), table([
        ["Подход", "Lift", "Freq", "Min year", "Future bps", "Источник / статус"],
        ["Multiscale anchor", "1.295", "1.02", "0.508", "+35.0", "ivan / locked"],
        ["Causal logit rolling q20", "1.307", "1.11", "1.050*", "+51.8", "version_b + audit / retro"],
        ["Reset XGBoost", "1.288", "1.07", "1.188", "+32.7", "ivan / current regime"],
        ["Geometric consensus", "1.264", "1.11", "1.138", "+33.4", "ivan / challenger"],
        ["Soft regime router", "1.325", "1.54", "1.161", "+32.0", "ivan / retro"],
        ["Trend anchor", "1.406", "1.03", "1.227", "+47.4", "ivan / posthoc"],
        ["Online local Hedge", "1.434", "1.63", "1.043", "+42.2", "ivan / retro unstable"],
    ], s, [48*mm, 19*mm, 19*mm, 21*mm, 24*mm, 43*mm], small=True)]
    story += [para("* Для causal logit строка min year включает 2022-2026; на 2024-2026 minimum aggregate year равен 1.102.", s, "SmallR3")]
    story += [PageBreak()]

    # 8. Stability and explainability
    story += heading("7. Почему общий lift скрывает нестабильность", s)
    story += [stability_chart(), Spacer(1, 4 * mm), para(
        "Линия 1.30 показывает порог. Post-publication модель стоит отдельно: у неё другая "
        "информация в момент решения. Online Hedge получает высокий aggregate, потому что "
        "частота сигналов распределена по годам неравномерно; version_b logit особенно силён "
        "в 2025-2026, но не в 2022-2024.", s)]
    story += [table([
        ["Семейство", "Объяснимость", "Что можно сказать пользователю"],
        ["Range anchor / known-next gate", "очень высокая", "точная формула и причина каждого сигнала"],
        ["Logistic regression", "высокая", "вклад стандартизированных признаков и знак зависимости"],
        ["Empirical Bayes / Markov", "средняя-высокая", "вероятность состояния, shrinkage и последовательность движений"],
        ["ExtraTrees / XGBoost", "средняя-низкая", "global importance/SHAP, но не короткая формула"],
        ["Equal/geometric ensemble", "средняя", "согласие нескольких понятных экспертов"],
        ["Learned router / Hedge", "низкая", "веса меняются по режиму и прошлым ошибкам"],
        ["GRU", "низкая", "скрытое состояние без устойчивого прироста"],
    ], s, [45*mm, 34*mm, 95*mm], small=True)]
    story += [PageBreak()]

    # 9. version_b audit
    story += heading("8. Отдельный аудит version_b", s)
    story += [callout(
        "Оригинальные 1.450 воспроизводятся точно, но сигнал выбирается как top 15% "
        "внутри уже целиком известного шестимесячного test-fold. Future labels не используются, "
        "однако онлайн 10 января нельзя знать scores февраля-июня.",
        s, PALE_ORANGE, ORANGE,
    )]
    story += [para("Apples-to-apples на исходной истории 2019-2026", s, "H2R3"), table([
        ["Политика порога", "Mean lift", "Aggregate", "Freq", "Min FX", "Вердикт"],
        ["Future test top-15%", "1.450", "1.452", "0.73", "1.319", "non-causal"],
        ["30m fit / 6m calib / fixed q20", "1.347", "1.366", "1.08", "1.212", "causal retro"],
        ["30m fit / 6m calib / rolling120 q20", "1.307", "1.306", "1.11", "1.198", "causal retro"],
        ["Past OOF rolling120 q30", "1.335", "1.344", "1.29", "1.160", "causal retro"],
    ], s, [62*mm, 24*mm, 24*mm, 20*mm, 20*mm, 24*mm], small=True)]
    story += [para("Длинная история 2013-2026 без перенастройки идеи", s, "H2R3"), table([
        ["Политика", "Mean lift", "Freq", "95% aggregate CI"],
        ["Future top-15%, diagnostic", "1.319", "0.73", "[1.188; 1.454]"],
        ["Past OOF expanding q20", "1.225", "1.04", "[1.099; 1.370]"],
        ["Nested fixed q20", "1.085", "1.16", "[0.982; 1.202]"],
        ["Nested rolling120 q20", "1.137", "1.09", "[1.038; 1.252]"],
    ], s, [72*mm, 31*mm, 27*mm, 44*mm])]
    story += [Spacer(1, 4 * mm), callout(
        "Итог: ranker содержит реальный recent-regime signal, но 1.450 нельзя считать "
        "устойчивой рабочей цифрой. Для forward test заморожен причинный rolling q20 challenger.",
        s, PALE_GREEN, GREEN,
    )]
    story += [PageBreak()]

    # 10. Post-publication
    story += heading("9. Самый сильный контур: после публикации", s)
    story += [para(
        "После публикации следующего эффективного курса известен v[t+1]. Target h=5 не меняется, "
        "но первый из пяти будущих шагов уже наблюдается. Если v[t+1] < v[t], текущий день "
        "структурно не может быть fav_h5. Если выше, известен запас перед четырьмя неизвестными шагами.", s)]
    story += [bar_chart([
        ("Known-next gate", 1.959, BLUE),
        ("Known margin only", 2.342, GREEN),
        ("Gate + past logit", 2.363, GREEN),
        ("Published t+1 logit", 2.493, ORANGE),
        ("t+1 + margin logit", 2.515, ORANGE),
        ("Logit + ExtraTrees retro", 2.553, RED),
    ], maximum=2.8, target=1.3, height=78*mm)]
    story += [table([
        ["Selected ExtraTrees", "2017-2020", "2022-2023", "2024-2026", "95% CI final"],
        ["Lift", "2.632", "2.395", "2.459", "[2.160; 2.797]"],
        ["Frequency", "1.055", "1.090", "1.069", "target met"],
        ["Future benefit", "+92.9 bps", "+278.5 bps", "+138.3 bps", "[+114; +160]"],
    ], s, [48*mm, 31*mm, 31*mm, 31*mm, 33*mm])]
    story += [PageBreak()]

    # 11. Why post-publication works and caveat
    story += heading("10. Механика post-publication и бизнес-граница", s)
    story += [para("Главные признаки", s, "H2R3")]
    story += bullets([
        "known_margin_vol = (v[t+1] - v[t]) / недавняя волатильность; главный фактор logit и ExtraTrees.",
        "Размер общего объявленного движения по пяти валютам и peer mean/min.",
        "Позиция нового опубликованного курса в коротком диапазоне.",
        "Past-only тренд, волатильность, USD/CNY и слабые циклические признаки.",
    ], s)
    story += [callout(
        "Это не утечка при условии правильного timestamp: v[t+1] разрешён только после фактической "
        "публикации ЦБ. Любой score раньше обновления должен быть технически заблокирован.",
        s, PALE_BLUE, BLUE,
    ), Spacer(1, 6 * mm)]
    story += [table([
        ["Необходимое условие", "Почему критично", "Статус"],
        ["Новый курс уже опубликован", "иначе v[t+1] является будущим", "проверять timestamp"],
        ["Сегодняшний клиентский курс ещё доступен", "иначе сигнал невозможно монетизировать", "нужно подтвердить у бизнеса"],
        ["Есть история фактического спреда банка", "курс ЦБ не равен цене операции", "данных пока нет"],
        ["Сигнал успевает до смены прайса", "задержка может съесть преимущество", "нужен latency SLA"],
    ], s, [48*mm, 78*mm, 48*mm])]
    story += [para(
        "Банк России указывает, что точное время публикации не регламентировано, обычно курсы "
        "размещаются до 18:00 МСК и вступают в силу на следующий календарный день. Это делает "
        "сценарий правдоподобным, но не доказывает доступность банковского клиентского курса.", s)]
    story += [PageBreak()]

    # 12. Other targets
    story += heading("11. Другие цели и две разные цены дня", s)
    story += [para("Window-closing target", s, "H2R3"), para(
        "close_h5(t)=1, если v[t+5] > v[t]. Это более слабая постановка: требуется только, "
        "чтобы пятая публикация была хуже сегодняшней, а не все пять.", s)]
    story += [table([
        ["Подход", "Lift", "Freq", "Benefit", "Источник", "Статус"],
        ["Trend anchor selected", "1.132", "1.629", "+25.0 bps", "ivan", "не проходит"],
        ["Upper-range retro best", "1.182", "1.458", "+32.1 bps", "ivan", "CI [1.021;1.343]"],
        ["ExtraTrees", "1.162", "1.072", "+18.0 bps", "ivan", "не проходит"],
    ], s, [47*mm, 22*mm, 22*mm, 29*mm, 24*mm, 30*mm])]
    story += [para("Future-only против симметричной +/-5", s, "H2R3"), table([
        ["Политика 2024-2026", "Future-only", "Симметричная +/-5", "Интерпретация"],
        ["Locked anchor", "+35.0", "-26.9", "будущее лучше, но день не локальный минимум"],
        ["Geometric consensus", "+33.4", "+4.3", "обе метрики положительны"],
        ["Online Hedge", "+42.2", "+7.4", "сильнее, но frequency drift"],
        ["Reset XGB", "+32.7", "+11.7", "ровнее современный режим"],
        ["Known-next gate", "+77.8", "+48.3", "другой timestamp"],
    ], s, [52*mm, 28*mm, 35*mm, 59*mm], small=True)]
    story += [callout(
        "Для продуктового прогноза главнее future-only: прошлый курс уже нельзя получить. "
        "Симметричная метрика полезна как описание локального минимума, но может награждать запоздалый сигнал.",
        s, PALE_ORANGE, ORANGE,
    )]
    story += [PageBreak()]

    # 13. Experiments I
    story += heading("12. Что пробовали: модели временных рядов и ML", s)
    story += [table([
        ["Семейство", "Лучшее наблюдение", "Почему не финал", "Источник"],
        ["Seasonal naive", "AUC 0.458", "календарный профиль не переносится", "ivan"],
        ["ETS", "AUC 0.478", "прогноз уровня плохо совпадает с barrier target", "ivan"],
        ["SARIMA", "AUC 0.523", "сигнал слишком слабый", "ivan"],
        ["GRU", "AUC 0.540", "сложность не окупилась на малом числе режимов", "ivan"],
        ["Main logit h5", "lift 1.14; freq 1.22", "полезный baseline, ниже 1.30", "main"],
        ["version_b logit", "1.450 diagnostic", "future test top-K; causal recent 1.307", "version_b + audit"],
        ["Global ExtraTrees", "shock 1.367", "final 1.254", "ivan"],
        ["XGB ranker", "general до 1.375", "shock/final около 1.0", "ivan"],
        ["Quantile future floor", "около 1.14", "нестабильный минимум будущего", "ivan"],
        ["KNN trajectories", "слабый отдельно", "полезен только разнообразием ошибок", "ivan"],
    ], s, [45*mm, 37*mm, 65*mm, 27*mm], small=True)]
    story += [para("Главный вывод", s, "H2R3"), para(
        "Сложность архитектуры не коррелирует с lift. Простое положение в диапазоне и "
        "линейный ranker нередко переносятся лучше, чем direct multistep forecast или RNN. "
        "Мы оптимизируем редкий верхний хвост качества сигналов, а не среднюю ошибку прогноза курса.", s)]
    story += [PageBreak()]

    # 14. Experiments II
    story += heading("13. Что пробовали: режимы, башни и внешние данные", s)
    story += [table([
        ["Идея", "Результат", "Диагноз", "Источник"],
        ["Local logit -> global XGB residual", "shock 1.075; final 1.110", "residual переносит устаревшие ошибки", "ivan"],
        ["Короткие окна ExtraTrees", "shock 1.384; freq 0.864", "сильный, но слишком редкий", "ivan"],
        ["Equal mix 6 experts", "shock 1.287; final 1.328", "простая смесь сильнее learned gate", "ivan"],
        ["Soft regime router", "final 1.325", "лучше по min-year, но retro", "ivan"],
        ["Online Hedge", "final 1.434", "Simpson inflation через frequency drift", "ivan"],
        ["Post-2022 reset XGB", "final 1.288; min-year 1.188", "самый ровный current model", "ivan"],
        ["Binary post-2022 feature", "около 1.09-1.18", "дата разрыва слишком груба", "ivan"],
        ["RUONIA/rate/Brent/USD", "shock до 1.03", "causal lagged proxies не помогают h5", "ivan"],
        ["Cross-sectional rank", "general 1.371; shock 1.038", "ломается после режима", "ivan"],
        ["Per-currency champions", "general 1.342; shock 0.888", "переобучение валюты и эпохи", "ivan"],
        ["Pooled discrete hazard", "final 1.206", "ошибки шагов накапливаются", "ivan"],
        ["Empirical Bayes states", "selected final 1.228", "не проходит 1.30", "ivan"],
        ["Markov + anchor", "retro 1.316; freq 0.961", "частота и старые эпохи не проходят", "ivan"],
    ], s, [52*mm, 40*mm, 55*mm, 27*mm], small=True)]
    story += [PageBreak()]

    # 15. Lessons
    story += heading("14. Что реально дало самый большой прирост", s)
    story += [table([
        ["Ранг", "Приём", "Что дал", "Почему работает"],
        ["1", "Изменить information set: известный v[t+1]", "gate 1.959 -> model 2.459", "часть target уже разрешена"],
        ["2", "Нормировать known margin на volatility", "главная post-pub feature", "измеряет запас в единицах обычного шума"],
        ["3", "Multiscale position 30/90/180", "locked 1.295", "совмещает реакцию и устойчивость"],
        ["4", "Global pooling пяти валют", "ExtraTrees shock 1.367", "общий фактор движений велик"],
        ["5", "Забывание старого режима", "reset XGB min-year 1.188", "2022 изменил распределение"],
        ["6", "Простое смешивание экспертов", "equal mix final 1.328", "снижает модельный риск"],
        ["7", "Честный rolling threshold", "убирает false win", "контролирует online частоту без test quantile"],
    ], s, [17*mm, 55*mm, 43*mm, 59*mm], small=True)]
    story += [para("Что не стоит переоценивать", s, "H2R3")]
    story += bullets([
        "Праздники и sin/cos календаря допустимы, но самостоятельного boost не дали.",
        "Один чемпион на валюту звучит естественно, однако независимых режимов слишком мало.",
        "Learned routing красив концептуально, но equal weights пока надёжнее.",
        "После 2022 не следует жёстко выбрасывать всё старое без sensitivity: recency лучше бинарного разрыва.",
        "AUC около 0.55-0.62 может быть полезен для верхнего хвоста; средний AUC не заменяет lift/frequency audit.",
    ], s)
    story += [PageBreak()]

    # 16. Statistics
    story += heading("15. Насколько результат доказан статистически", s)
    story += [table([
        ["Утверждение", "Данные", "Вердикт"],
        ["Locked anchor лучше random", "lift 1.295; CI [0.964;1.596]", "нестрого на 95%"],
        ["Новые ordinary models лучше anchor", "CI разностей пересекают 0", "не доказано"],
        ["Online Hedge стабильно >1.30", "macro-year 1.251; freq 0.48-3.09", "нет"],
        ["version_b causal recent >1.30", "CI fixed [1.210;1.565]; rolling [1.153;1.502]", "1.30 внутри CI"],
        ["version_b causal long >1.30", "лучший CI [1.099;1.370]", "нет"],
        ["Post-publication selected >1", "CI [2.160;2.797]; adjusted p=0.00025", "сильное подтверждение"],
        ["Window closing >1.30", "best CI [1.021;1.343]; adjusted p=0.165", "нет"],
    ], s, [62*mm, 69*mm, 43*mm], small=True)]
    story += [Spacer(1, 5 * mm), para("Цена перебора", s, "H2R3"), para(
        "В ordinary исследовании записано 157 политик. Если после просмотра final выбрать "
        "максимум, обычный bootstrap уже не исправляет selection bias. Поэтому отчёт различает "
        "locked, selected earlier и retro max; circular-shift audit оценивает максимум под null.", s)]
    story += [callout(
        "Правильная headline цифра - не самый большой найденный lift, а результат конфигурации, "
        "замороженной до соответствующего тестового блока.", s, PALE_GREEN, GREEN,
    )]
    story += [PageBreak()]

    # 17. Architecture and decisions
    story += heading("16. Рекомендуемая финальная схема", s)
    story += [architecture_diagram(), Spacer(1, 3 * mm)]
    story += [table([
        ["Контур", "Основной", "Challenger", "Что заморозить"],
        ["До публикации", "Multiscale anchor", "version_b causal logit; reset XGB; geometric consensus", "features, model, q20/rolling120, per-FX thresholds"],
        ["После публикации", "Conditional ExtraTrees 7y", "logit + ExtraTrees ensemble", "q22, rolling250, actual publication gate"],
        ["Window closing", "нет production winner", "upper-range rule", "только research monitoring"],
    ], s, [38*mm, 39*mm, 54*mm, 43*mm], small=True)]
    story += [para("Порядок пилота", s, "H2R3")]
    story += bullets([
        "Параллельно считать все frozen policies, но пользователю показывать один согласованный сигнал.",
        "Логировать publication timestamp, scoring timestamp, показ уведомления и фактический клиентский курс.",
        "Не менять параметры по новым данным до заранее назначенной даты аудита.",
        "На аудите сравнить lift, frequency, future bps, minimum year/currency и бизнес-конверсию.",
        "Post-publication включать только после подтверждения, что прежний курс ещё исполним.",
    ], s)
    story += [PageBreak()]

    # 18. Limits and reproducibility
    story += heading("17. Ограничения, следующий holdout и воспроизводимость", s)
    story += [para("Что пока неизвестно", s, "H2R3")]
    story += bullets([
        "Фактический банковский rate, spread, комиссии и момент обновления прайса.",
        "История спроса и переводов: официальные курсы не позволяют проверить гипотезу о росте спроса после отключений SWIFT.",
        "Реакция пользователей на частые и кластерные уведомления.",
        "Сохранится ли режим 2025-2026 на новых публикациях.",
    ], s)
    story += [para("Замороженный будущий тест", s, "H2R3"), callout(
        "После 04.09.2026 не менять признаки, окна, веса и пороги frozen candidates. "
        "Иначе следующий период снова станет development, а не holdout.", s, PALE_ORANGE, ORANGE)]
    story += [para("Ключевые артефакты", s, "H2R3")]
    story += bullets([
        "EXPERIMENTS_SUMMARY.md - короткий навигатор по всем волнам.",
        "results/research/round2/report-source.md - данные, модели, внешние факторы и ensembles.",
        "results/research/round3/report-source.md - режимы, online mixtures и statistical audit.",
        "results/research/round4/report.md - conditional publication, state models и closing target.",
        "results/research/version_b_honest_audit/report.md - причинный аудит ветки version_b.",
        "research/* - воспроизводимый код; results/research/* - полные grid, OOF и bootstrap таблицы.",
        "56 автоматических тестов проверяют корректность, as-of семантику и leakage boundaries.",
    ], s)
    story += [para("Внешние ссылки", s, "H2R3")]
    story += bullets([
        '<link href="https://talenttrack.aitalenthub.ru/hackathon/cases/455">Условия кейса</link>.',
        '<link href="https://www.cbr.ru/currency_base/dynamics/">Официальная динамика курсов Банка России</link>.',
        '<link href="https://www.cbr.ru/faq/dkp/04/">Банк России: срок действия официального курса</link>.',
        '<link href="https://www.cbr.ru/Reception/TopicalMessage/Page/2661">Банк России: время публикации курса</link>.',
    ], s)

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(PDF)
    return PDF


if __name__ == "__main__":
    build()
