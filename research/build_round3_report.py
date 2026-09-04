"""Build the verified PDF for the third forecasting research round."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "research" / "round3"
PDF = ROOT / "output" / "pdf" / "ivan_deep_research_round3_full.pdf"

NAVY = colors.HexColor("#0f172a")
BLUE = colors.HexColor("#2563eb")
GREEN = colors.HexColor("#059669")
ORANGE = colors.HexColor("#d97706")
RED = colors.HexColor("#dc2626")
GRAY = colors.HexColor("#475569")
LIGHT = colors.HexColor("#e2e8f0")
PALE_BLUE = colors.HexColor("#eff6ff")
PALE_GREEN = colors.HexColor("#ecfdf5")
PALE_ORANGE = colors.HexColor("#fff7ed")
PALE_RED = colors.HexColor("#fef2f2")


def clean(value: object) -> str:
    return (str(value).replace("—", "-").replace("–", "-")
            .replace("−", "-").replace("‑", "-"))


def register_fonts() -> None:
    base = Path("/System/Library/Fonts/Supplemental")
    pdfmetrics.registerFont(TTFont("Arial", str(base / "Arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(base / "Arial Bold.ttf")))


def get_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "TitleR3", fontName="Arial-Bold", fontSize=24, leading=28,
        textColor=NAVY, alignment=TA_LEFT, spaceAfter=9,
    ))
    styles.add(ParagraphStyle(
        "SubtitleR3", fontName="Arial", fontSize=11.5, leading=16,
        textColor=GRAY, spaceAfter=13,
    ))
    styles.add(ParagraphStyle(
        "H1R3", fontName="Arial-Bold", fontSize=16.5, leading=20,
        textColor=NAVY, spaceBefore=3, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        "H2R3", fontName="Arial-Bold", fontSize=11.5, leading=14,
        textColor=NAVY, spaceBefore=7, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "BodyR3", fontName="Arial", fontSize=9.15, leading=12.7,
        textColor=NAVY, spaceAfter=5.5,
    ))
    styles.add(ParagraphStyle(
        "SmallR3", fontName="Arial", fontSize=7.35, leading=9.3,
        textColor=GRAY, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        "CalloutR3", fontName="Arial-Bold", fontSize=9.9, leading=13.5,
        textColor=NAVY,
    ))
    styles.add(ParagraphStyle(
        "CellR3", fontName="Arial", fontSize=6.85, leading=8.6,
        textColor=NAVY,
    ))
    styles.add(ParagraphStyle(
        "CellSmallR3", fontName="Arial", fontSize=6.15, leading=7.7,
        textColor=NAVY,
    ))
    styles.add(ParagraphStyle(
        "CellHeaderR3", fontName="Arial-Bold", fontSize=6.65, leading=8.2,
        textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        "MetricR3", fontName="Arial-Bold", fontSize=18, leading=21,
        alignment=TA_CENTER, textColor=BLUE,
    ))
    styles.add(ParagraphStyle(
        "MetricSmallR3", fontName="Arial", fontSize=6.9, leading=8.5,
        alignment=TA_CENTER, textColor=GRAY,
    ))
    return styles


def para(text: str, styles, style: str = "BodyR3") -> Paragraph:
    return Paragraph(clean(text), styles[style])


def heading(title: str, styles) -> list:
    return [para(title, styles, "H1R3")]


def table(data, styles, widths=None, small=False, header=True):
    cooked = []
    for i, row in enumerate(data):
        style = "CellHeaderR3" if header and i == 0 else (
            "CellSmallR3" if small else "CellR3"
        )
        cooked.append([Paragraph(clean(cell), styles[style]) for cell in row])
    result = Table(cooked, colWidths=widths, repeatRows=1 if header else 0,
                   hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), .35, LIGHT),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
    for i in range(1 if header else 0, len(cooked)):
        if i % 2 == 0:
            commands.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8fafc")))
    result.setStyle(TableStyle(commands))
    return result


def callout(text: str, styles, fill=PALE_BLUE, border=BLUE):
    result = Table([[para(text, styles, "CalloutR3")]], colWidths=[174 * mm])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("BOX", (0, 0), (-1, -1), 1, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return result


def bullets(items: list[str], styles) -> list:
    return [para(f"• {item}", styles) for item in items]


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LIGHT)
    canvas.line(20 * mm, 14 * mm, 190 * mm, 14 * mm)
    canvas.setFont("Arial", 7)
    canvas.setFillColor(GRAY)
    canvas.drawString(20 * mm, 9 * mm,
                      "international_transfers_signals | ivan-experiments")
    canvas.drawRightString(190 * mm, 9 * mm, f"04.09.2026 | {doc.page}")
    canvas.restoreState()


def build() -> Path:
    register_fonts()
    s = get_styles()
    PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=18 * mm,
        title="Третье глубокое исследование прогнозирования h=5",
        author="international_transfers_signals",
    )
    story = []

    # Cover
    story += [Spacer(1, 17 * mm), para("Третье глубокое исследование", s, "TitleR3"),
              para("Все эксперименты, честный аудит lift и две оценки цены дня", s, "SubtitleR3")]
    metrics = [
        [para("1.434*", s, "MetricR3"), para("1.295", s, "MetricR3"),
         para("1.959", s, "MetricR3")],
        [para("максимум past-only", s, "MetricSmallR3"),
         para("locked past-only", s, "MetricSmallR3"),
         para("после публикации", s, "MetricSmallR3")],
    ]
    mt = Table(metrics, colWidths=[58 * mm] * 3, rowHeights=[15 * mm, 10 * mm])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 1, BLUE),
        ("INNERGRID", (0, 0), (-1, -1), .4, colors.HexColor("#bfdbfe")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [mt, Spacer(1, 7 * mm), callout(
        "Честный итог: новый past-only максимум 1.434 найден, но не доказан и частично "
        "вызван неравномерной частотой по годам. Строго зафиксированный результат остаётся "
        "1.295. Наиболее стабильные новые challengers - geometric consensus и post-2022 "
        "reset XGB. Lift 1.959 относится к отдельному режиму, где следующий эффективный курс "
        "ЦБ уже опубликован.", s, PALE_GREEN, GREEN), Spacer(1, 5 * mm),
              para("* Ретроспективно на 2024-2026; этот период уже просматривался.", s, "SmallR3"),
              para("Ветка ivan-experiments. Push не выполнялся.", s, "SmallR3"), PageBreak()]

    # Executive answer
    story += heading("1. Ответ без прикрас", s)
    story += [callout(
        "Мы не получили честно подтверждённый past-only lift выше 1.40. Максимум 1.434 "
        "интересен как гипотеза, но его superiority над locked anchor статистически не "
        "подтверждена, а годовая частота дрейфует от 0.48 до 3.09.", s, PALE_ORANGE, ORANGE)]
    story += [para("Что считать лучшим в разных смыслах", s, "H2R3"), table([
        ["Критерий", "Победитель", "Результат", "Оговорка"],
        ["Честный locked final", "Multiscale anchor", "lift 1.295", "плохой 2025: 0.508"],
        ["Максимальный headline", "Online local Hedge", "lift 1.434", "ретро; Simpson gap +0.183"],
        ["Перенос по режимам", "Geometric consensus", "1.375 / 1.189 / 1.264", "не достигает 1.40"],
        ["Текущий режим", "Post-2022 reset XGB", "lift 1.288; min year 1.188", "нет нового holdout"],
        ["После публикации", "Known-next-rate", "lift 1.959", "другая информация в момент решения"],
    ], s, [38 * mm, 40 * mm, 43 * mm, 53 * mm])]
    story += [Spacer(1, 4 * mm), para("Почему одна цифра недостаточна", s, "H2R3")]
    story += bullets([
        "Общий lift взвешивает годы числом сигналов. Модель может почти молчать в трудном году.",
        "2024-2026 больше не unseen holdout для новых идей третьей волны.",
        "157 записанных политик создают multiple-testing bias.",
        "Future-only и симметричная +-h выгода отвечают на разные вопросы.",
    ], s)
    story += [para("Рекомендация: заморозить consensus и reset XGB как два challenger, "
                   "сохранить locked anchor как benchmark и собирать новый проспективный поток.", s),
              PageBreak()]

    # Task and data
    story += heading("2. Задача, данные и target", s)
    story += [para(
        "На каждой публикации ЦБ для TJS, UZS, KGS, AMD и KZT выбирается удачный день "
        "отправки рублей. Меньший нормированный курс означает более выгодный момент. История: "
        "01.01.2010-02.09.2026. Номиналы 10/100 нормируются до одной единицы валюты.", s)]
    story += [table([
        ["Объект", "Точное определение"],
        ["fav_h5", "1, если v[t] <= min(v[t+1], ..., v[t+5])"],
        ["Горизонт", "5 следующих публикаций ЦБ, не календарных дней"],
        ["Lift", "hit rate среди сигналов / base rate на том же OOS-блоке"],
        ["Рабочая частота", "1-2 сигнала на валютный коридор в неделю"],
        ["Base rate 2024-2026", "29.45%; lift 1.40 соответствует hit rate около 41.2%"],
    ], s, [44 * mm, 130 * mm])]
    story += [para("Три оценки цены дня", s, "H2R3"), table([
        ["Метрика", "Окно", "Интерпретация"],
        ["Future-only target", "t+1 ... t+h", "основная задача классификации"],
        ["Future-only benefit", "сегодня против среднего будущего", "достижимая клиентом выгода"],
        ["Симметричная benefit", "t-h ... t+h", "условие кейса; включает уже случившееся"],
    ], s, [43 * mm, 54 * mm, 77 * mm])]
    story += [Spacer(1, 4 * mm), callout(
        "Положительный future-only benefit при отрицательной симметричной метрике не ошибка. "
        "Past-only модель может выбрать день, который хорош относительно будущего, но не является "
        "локальным минимумом относительно уже прошедшей части окна.", s),
              para('<link href="https://talenttrack.aitalenthub.ru/hackathon/cases/455">Условия кейса</link> | '
                   '<link href="https://www.cbr.ru/currency_base/dynamics/">данные Банка России</link>', s, "SmallR3"),
              PageBreak()]

    # Protocol
    story += heading("3. Разбиение и защита от утечек", s)
    story += [table([
        ["Период", "Роль"],
        ["2010-2016", "development и train-only EDA"],
        ["2017-2020", "general validation семейств"],
        ["2021", "калибровка перед shock-блоком"],
        ["2022-2023", "shock/adaptation validation"],
        ["2024-2026", "ретроспективный аудит; для новых идей не holdout"],
    ], s, [42 * mm, 132 * mm])]
    story += [para("Causal walk-forward", s, "H2R3")]
    story += bullets([
        "Для тестового года Y обучение заканчивается раньше Y-1; Y-1 калибрует порог.",
        "Purging учитывает дату, когда полностью разрешается h=5 target.",
        "Rolling threshold видит только прошлые scores.",
        "Online Hedge/SGD обновляются только после получения завершённого target.",
        "Future corruption tests требуют неизменности ранних features и сигналов.",
        "pct_range_90 строится только по текущей и предыдущим 89 публикациям.",
    ], s)
    story += [para("Выходные и 2022", s, "H2R3"), para(
        "Выходные не считаются пропусками: основная ось - публикации ЦБ, а календарный разрыв "
        "входит как gap_days. Дата 24.02.2022 применена только в reset-sensitivity. Сам факт "
        "структурного сдвига виден, но причинную связь с санкциями или SWIFT одни курсы не доказывают.", s)]
    story += [callout(
        "Главное ограничение: 2024-2026 уже многократно просмотрен. Любое новое улучшение на нём "
        "называется retrospective, даже если все признаки и вычисления причинны.", s, PALE_RED, RED),
              PageBreak()]

    # Earlier experiments
    story += heading("4. База до третьей волны", s)
    story += [para("Временные ряды и нейросеть", s, "H2R3"), table([
        ["Модель", "Walk-forward AUC h=5", "Итог"],
        ["Seasonal naive-5", "0.458", "сезонность не переносится"],
        ["ETS-5", "0.478", "слабее случайного ранжирования"],
        ["SARIMA-5", "0.523", "небольшой сигнал"],
        ["SARIMA-20", "0.488", "длиннее - хуже"],
        ["Drift-20", "0.544", "лучше ETS/SARIMA"],
        ["GRU classifier", "0.540", "нет преимущества над простым ML"],
        ["pct_range_90", "0.572", "сильнейшая простая anchor-фича"],
    ], s, [53 * mm, 41 * mm, 80 * mm])]
    story += [para("Классический ML и round 2", s, "H2R3"), table([
        ["Семейство", "2022-23", "2024-26*", "Вывод"],
        ["Global ExtraTrees", "1.367", "1.254", "лучший отдельный эксперт"],
        ["Local logit -> global XGB residual", "1.075", "1.110", "идея реализована, перенос слабый"],
        ["XGB ranker", "до 1.073", "до 1.076", "старый lift 1.375 не перенёсся"],
        ["Short-window Extra mix", "1.384; f=.864", "1.277", "сильный, но не прошёл frequency gate"],
        ["Equal mix 6 experts", "1.287", "1.328", "простая смесь сильнее gate"],
        ["Soft regime router", "1.243", "1.325", "лучше по min-year"],
        ["RUONIA/key/Brent/USD", "до 1.032", "до 1.105", "timestamp-safe macro не помог"],
    ], s, [51 * mm, 27 * mm, 29 * mm, 67 * mm], small=True)]
    story += [para("* Retrospective для идей, появившихся после просмотра блока.", s, "SmallR3"),
              PageBreak()]

    # Round 3 A
    story += heading("5. Третья волна: ансамбли и новые targets", s)
    story += [Image(str(DATA / "report_transfer.png"), width=174 * mm, height=76 * mm)]
    story += [para("Consensus и online Hedge", s, "H2R3"), para(
        "Одиннадцать причинных экспертов объединены по нормированным рангам. Geometric consensus "
        "даёт 1.375 / 1.189 / 1.264 на general / shock / final. Delayed-feedback Hedge меняет "
        "веса только после завершения h=5. Local Hedge достигает final 1.434, но не переносится "
        "на shock (1.170) и сильно меняет годовую частоту.", s)]
    story += [para("Direct path/barrier", s, "H2R3"), para(
        "Ridge/Extra/Hist прогнозируют пять накопленных log-return, затем совместное empirical "
        "residual distribution оценивает вероятность не пересечь барьер. Лучшее: 1.140 / 1.076 / "
        "1.224. Правильная постановка пути не компенсировала ошибку пяти прогнозов.", s)]
    story += [para("Delayed labels и partial barrier", s, "H2R3"), para(
        "Использованы только уже разрешившиеся h=5 target-rates и известная часть четырёх "
        "незавершённых окон. Лучшее: 1.316 / 1.159 / 1.184. Признак запаздывает за быстрым drift.", s),
              PageBreak()]

    # Round 3 B
    story += heading("6. Третья волна: режимы и локализация", s)
    story += [para("Post-2022 reset XGB", s, "H2R3"), para(
        "Все target-строки до 24.02.2022 отброшены. Rate 20% и rolling-120 дают самый ровный "
        "современный профиль.", s), table([
        ["Год", "Lift", "Частота", "Future, б.п.", "+-h, б.п."],
        ["2024", "1.290", "1.016", "+43.0", "+15.8"],
        ["2025", "1.188", "1.167", "-8.1", "+6.9"],
        ["2026", "1.437", "1.120", "+85.2", "+13.6"],
        ["Все", "1.288", "1.070", "+32.7", "+11.7"],
    ], s, [33 * mm, 32 * mm, 38 * mm, 36 * mm, 35 * mm])]
    story += [para("Cross-sectional state", s, "H2R3"), para(
        "Среднее, spread, breadth и rank текущих признаков среди пяти валют дали ExtraTrees "
        "1.371 на general, но 1.038 на shock и 1.173 final. Пять рядов недостаточны для "
        "устойчивого cross-sectional ранга.", s)]
    story += [para("Pooled discrete hazard", s, "H2R3"), para(
        "At-risk развёртка в пять шагов и произведение условных survival probabilities: "
        "1.292 / 1.089 / 1.206. Ошибка на последовательных hazards накапливается.", s)]
    story += [para("Per-currency champions", s, "H2R3"), para(
        "Четыре валюты выбрали pairwise ranker, UZS - ExtraTrees window-3. General 1.342, "
        "затем shock 0.888 и final 1.044: явное переобучение к старому режиму.", s), PageBreak()]

    # Round 3 C
    story += heading("7. Третья волна: online, смеси и отбор", s)
    story += [para("Causal online logistic SGD", s, "H2R3"), para(
        "Global/local partial_fit обновлялся лишь после разрешения target. Лучший shock lift 1.047, "
        "final 1.082. Линейная граница и единичные обновления слишком шумны.", s)]
    story += [para("Ансамбли reset + anchor + online", s, "H2R3"), para(
        "Лучший вариант, прошедший 2025 gate, - 25% reset XGB + 75% multiscale anchor: final 1.221. "
        "Другие смеси поднимали отдельный 2026 до 1.43, но ломали 2025 или частоту.", s)]
    story += [para("Frequency-balanced selection", s, "H2R3"), para(
        "Кандидаты ранжировались не только по aggregate lift, но и по macro-year lift, min-year, "
        "min-corridor и диапазону годовой частоты. Balanced equal mix имеет final 1.348, но macro "
        "1.206 и frequency 0.68-2.66. Geometric consensus сохраняет headline 1.264 и macro 1.245, "
        "frequency 0.90-1.37, поэтому выбран стабильным benchmark.", s)]
    story += [callout(
        "Главный урок round 3: сложность полезна только когда остаётся переносимой. Direct barrier, "
        "hazard, online SGD и learned routing выглядят содержательно, но на пяти валютах и нескольких "
        "режимах их variance выше выигрыша.", s, PALE_GREEN, GREEN), Spacer(1, 5 * mm)]
    story += [para("Почему online Hedge показал 1.434", s, "H2R3"), table([
        ["Год", "Lift", "Сигналов / валюта / неделя", "Future, б.п."],
        ["2024", "1.288", "3.089", "+41.5"],
        ["2025", "1.043", "0.477", "-18.3"],
        ["2026", "1.422", "1.325", "+79.0"],
        ["Aggregate", "1.434", "1.628", "+42.2"],
        ["Macro-year", "1.251", "-", "-"],
    ], s, [36 * mm, 31 * mm, 67 * mm, 40 * mm])]
    story += [PageBreak()]

    # Main leaderboard
    story += heading("8. Past-only leaderboard: 2024-2026", s)
    story += [Image(str(DATA / "report_final_comparison.png"), width=174 * mm, height=91 * mm)]
    story += [table([
        ["Политика", "Статус", "Lift", "Macro", "Freq", "Min year", "Future"],
        ["Online local Hedge", "retro", "1.434", "1.251", "1.628", "1.043", "+42.2"],
        ["Trend anchor", "posthoc", "1.406", "1.381", "1.026", "1.227", "+47.4"],
        ["Balanced equal mix", "retro", "1.348", "1.206", "1.629", "1.016", "+38.3"],
        ["Equal mix original", "retro", "1.328", "1.166", "1.546", "0.882", "+38.2"],
        ["Soft regime router", "retro", "1.325", "1.276", "1.542", "1.161", "+32.0"],
        ["Locked multiscale", "locked", "1.295", "0.977", "1.022", "0.508", "+35.0"],
        ["Post-2022 reset XGB", "retro", "1.288", "1.305", "1.070", "1.188", "+32.7"],
        ["Short-window mix", "retro", "1.277", "1.178", "1.165", "1.096", "+39.8"],
        ["Geometric consensus", "retro", "1.264", "1.245", "1.114", "1.138", "+33.4"],
        ["Global ExtraTrees", "candidate", "1.254", "1.371", "0.988", "1.061", "+27.3"],
    ], s, [49 * mm, 25 * mm, 19 * mm, 19 * mm, 19 * mm, 22 * mm, 21 * mm], small=True)]
    story += [para(
        "Headline и macro расходятся, когда политика распределяет сигналы между годами неравномерно. "
        "Поэтому reset XGB и consensus привлекательнее своего места в сортировке по одному lift.", s),
              PageBreak()]

    # Annual stability
    story += heading("9. Стабильность по годам", s)
    story += [Image(str(DATA / "report_annual_stability.png"), width=174 * mm, height=76 * mm)]
    story += [table([
        ["Политика", "2024 lift/freq", "2025 lift/freq", "2026 lift/freq", "Диагноз"],
        ["Locked anchor", "1.251 / 1.36", "0.508 / 0.28", "1.172 / 1.78", "провал и почти тишина в 2025"],
        ["Equal mix", "1.225 / 2.42", "0.882 / 0.73", "1.392 / 1.64", "частота следует сложности года"],
        ["Consensus", "1.138 / 1.37", "1.142 / 0.90", "1.454 / 1.17", "самый ровный перенос"],
        ["Online Hedge", "1.288 / 3.09", "1.043 / 0.48", "1.422 / 1.33", "headline раздут weighting"],
        ["Reset XGB", "1.290 / 1.02", "1.188 / 1.17", "1.437 / 1.12", "лучший current-regime profile"],
    ], s, [38 * mm, 29 * mm, 29 * mm, 29 * mm, 49 * mm], small=True)]
    story += [Spacer(1, 5 * mm), callout(
        "Если бизнесу нужен стабильный поток уведомлений, online Hedge нельзя выбирать только по "
        "1.434. Его 2024 частота выше guardrail, а 2025 ниже. Reset XGB и consensus лучше контролируют "
        "операционное поведение.", s, PALE_ORANGE, ORANGE), PageBreak()]

    # Benefit metrics
    story += heading("10. Цена дня: учитывать прошлое или нет", s)
    story += [Image(str(DATA / "report_benefit_comparison.png"), width=174 * mm, height=76 * mm)]
    story += [para(
        "Future-only benefit сравнивает сегодня со средним следующих пяти публикаций. Симметричная "
        "метрика сравнивает с десятью соседями: пятью прошлыми и пятью будущими. Вторая соответствует "
        "формулировке локального минимума, но включает недостижимую прошлую половину.", s)]
    story += [table([
        ["Политика", "Future-only, б.п.", "Симметричная +-5, б.п.", "Комментарий"],
        ["Locked anchor", "+35.0", "-26.9", "будущее лучше, локального минимума нет"],
        ["Equal mix", "+38.2", "-1.4", "почти нейтрально в полном окне"],
        ["Soft router", "+32.0", "+0.8", "согласованный знак"],
        ["Consensus", "+33.4", "+4.3", "согласованный знак"],
        ["Online Hedge", "+42.2", "+7.4", "положительно, но нестабильна частота"],
        ["Reset XGB", "+32.7", "+11.7", "лучший баланс двух выгод"],
    ], s, [41 * mm, 35 * mm, 42 * mm, 56 * mm])]
    story += [callout(
        "Для точности прогноза primary остаётся future-only fav_h5. Для отчёта заказчику нужно "
        "добавлять симметричную выгоду отдельной колонкой, а не смешивать эти цели в одну цифру.", s),
              PageBreak()]

    # Published price scenario
    story += heading("11. Сценарий после публикации курса", s)
    story += [callout(
        "Если следующий эффективный курс уже официально опубликован, первый шаг h=5 становится "
        "известным. Это законный timestamp-aware продуктовый сигнал после публикации, но утечка для "
        "любого решения, которое должно быть принято раньше.", s, PALE_GREEN, GREEN)]
    story += [Spacer(1, 5 * mm), table([
        ["Период", "h", "Lift", "Частота", "Future, б.п.", "+-h, б.п.", "Min year", "Min FX"],
        ["2022-2023", "5", "1.911", "1.394", "+143.1", "+77.3", "1.745", "1.847"],
        ["2024-2026", "5", "1.959", "1.369", "+77.8", "+48.3", "1.819", "1.801"],
    ], s, [34 * mm, 12 * mm, 21 * mm, 25 * mm, 25 * mm, 24 * mm, 19 * mm, 14 * mm])]
    story += [para("Почему результат такой высокий", s, "H2R3")]
    story += bullets([
        "Условие первого будущего шага уже проверено фактом публикации.",
        "Cooldown три календарных дня удерживает частоту около 1.37-1.39.",
        "Оставшиеся четыре публикации всё ещё неизвестны, поэтому lift не равен механическим 100% для h=5.",
        "Минимумы по годам и валютам выше 1.8 на финальном блоке.",
    ], s)
    story += [para(
        'Банк России сообщает ориентир публикации до 18:00 МСК, но точное время не гарантировано. '
        'Поэтому production должен проверять факт появления нового значения, а не доверять часам. '
        '<link href="https://www.cbr.ru/Reception/TopicalMessage/Page/2661">Разъяснение Банка России</link>.', s),
              PageBreak()]

    # Statistics
    story += heading("12. Неопределённость и цена перебора", s)
    story += [para("Четырёхнедельный block bootstrap", s, "H2R3"), table([
        ["Политика", "Lift", "95% CI", "Разница с anchor, 95% CI", "Future CI, б.п."],
        ["Online local Hedge", "1.434", "[1.227; 1.732]", "[-0.114; 0.514]", "[14.0; 72.1]"],
        ["Trend anchor posthoc", "1.406", "[1.132; 1.684]", "[-0.202; 0.432]", "[11.9; 81.7]"],
        ["Balanced equal mix", "1.348", "[1.146; 1.615]", "[-0.185; 0.368]", "[8.7; 67.4]"],
        ["Soft regime router", "1.325", "[1.150; 1.551]", "[-0.231; 0.361]", "[6.1; 57.2]"],
        ["Locked anchor", "1.295", "[0.964; 1.596]", "reference", "[-14.0; 67.3]"],
        ["Reset XGB", "1.288", "[1.012; 1.587]", "[-0.384; 0.425]", "[-2.7; 65.9]"],
        ["Geometric consensus", "1.264", "[1.042; 1.615]", "[-0.325; 0.378]", "[8.4; 65.6]"],
    ], s, [42 * mm, 18 * mm, 31 * mm, 49 * mm, 34 * mm], small=True)]
    story += [para(
        "Target-окна перекрываются, а пять валют одного дня коррелированы, поэтому iid bootstrap "
        "не использовался. Новые модели имеют lift CI выше единицы, но интервалы разницы с locked "
        "anchor у всех пересекают ноль.", s)]
    story += [para("Multiplicity", s, "H2R3"), para(
        "Для 157 recorded policies все исходы одновременно циклически сдвигались по датам. "
        "Наблюдаемый максимум унифицированной сетки: 1.392; 95-й процентиль максимума под null: "
        "1.409; минимальное max-adjusted p=0.067. Это диагностический Reality-Check analogue, "
        "а не формальный SPA, но он запрещает уверенное заявление о 5% superiority.", s)]
    story += [para(
        '<link href="https://doi.org/10.1111/1468-0262.00152">White Reality Check</link> | '
        '<link href="https://doi.org/10.1198/073500105000000063">Hansen SPA</link> | '
        '<link href="https://doi.org/10.3982/ECTA5771">Model Confidence Set</link>', s, "SmallR3"),
              PageBreak()]

    # Drivers and failures
    story += heading("13. Что реально дало boost", s)
    story += [table([
        ["Фактор", "Эффект", "Что делать"],
        ["Trailing range position", "сильнее ETS/SARIMA/GRU", "оставить multiscale anchor"],
        ["Global pooling 5 FX", "ExtraTrees 1.367 на shock", "учить общий нелинейный слой"],
        ["Простое усреднение", "часто сильнее learned gate", "shrinkage к равным весам"],
        ["Recency/reset", "адаптация после 2022", "контролировать variance и окно"],
        ["Rolling frequency", "выявляет Simpson inflation", "годовой guardrail 1-2"],
        ["Known next rate", "lift до 1.959", "только runtime timestamp gate"],
    ], s, [43 * mm, 61 * mm, 70 * mm])]
    story += [para("Что не сработало и почему", s, "H2R3")]
    story += bullets([
        "ETS/SARIMA/seasonal naive: target является прохождением барьера, сезонность нестабильна.",
        "GRU: мало независимых режимов; сложность не окупилась.",
        "Ranker и per-currency champions: сильное переобучение к 2017-2020.",
        "Residual XGB tower: глобальный слой перенёс устаревшие ошибки локальных моделей.",
        "External macro: доступные RUONIA/key/Brent/USD прокси слишком грубы для h=5.",
        "Barrier and hazard: последовательные ошибки на пяти шагах накапливаются.",
        "Online SGD: линейность и noisy delayed updates.",
        "Hard regime routing: слишком мало независимых regimes для обучения gate.",
    ], s)
    story += [callout(
        "Самый большой модельный boost пришёл не от более сложной архитектуры, а от согласования "
        "формы target с trailing range, global pooling и дисциплины калибровки. Самый большой общий "
        "boost 1.96 пришёл от изменения времени решения после публикации.", s, PALE_BLUE, BLUE),
              PageBreak()]

    # Decision
    story += heading("14. Финальное решение", s)
    story += [table([
        ["Роль", "Замороженная политика", "Почему"],
        ["Champion", "Locked multiscale anchor", "единственный заранее зафиксированный final lift 1.295"],
        ["Stable challenger", "Geometric consensus; rate .20; rolling 120", "лучший перенос и ровная частота"],
        ["Current challenger", "Post-2022 reset XGB; rate .20; rolling 120", "лучший современный min-year/frequency"],
        ["Research only", "Online local Hedge", "headline 1.434, но frequency drift"],
        ["Separate product", "After-publication known-next-rate", "lift 1.959 при строгом timestamp"],
    ], s, [34 * mm, 64 * mm, 76 * mm])]
    story += [para("Правила следующего теста", s, "H2R3")]
    story += bullets([
        "После 04.09.2026 не менять features, weights, target rate и rolling window.",
        "Логировать timestamp публикации ЦБ, timestamp сигнала и клиентский execution price.",
        "Primary: future-only lift h=5; co-primary: future-only benefit.",
        "Guardrails: 1-2 сигнала/валюта/неделю, min-year, min-currency, clustering 7d.",
        "Симметричную +-h выгоду показывать отдельно как метрику кейса.",
        "Победителя выбирать парным block-bootstrap сравнением с locked anchor.",
    ], s)
    story += [para("Что ещё нужно для бизнес-модели", s, "H2R3"), para(
        "История реального банковского курса и спреда, комиссия, суммы переводов, реакция на "
        "уведомления и контрольная группа. Без этого результат описывает тайминг официального курса "
        "ЦБ, но не доказывает денежный uplift продукта.", s)]
    story += [callout(
        "Короткая формулировка для защиты: past-only baseline 1.295 подтверждён; новые модели дают "
        "до 1.434 ретроспективно, но не проходят строгую проверку superiority. Продуктовый "
        "after-publication сценарий даёт 1.959 и должен маркироваться отдельно.", s, PALE_GREEN, GREEN),
              PageBreak()]

    # Sources and artifacts
    story += heading("15. Источники и воспроизводимость", s)
    sources = [
        ('Условия кейса', 'https://talenttrack.aitalenthub.ru/hackathon/cases/455'),
        ('Банк России: динамика курсов', 'https://www.cbr.ru/currency_base/dynamics/'),
        ('Банк России: публикация курса', 'https://www.cbr.ru/Reception/TopicalMessage/Page/2661'),
        ('Rossi: Exchange Rate Predictability', 'https://doi.org/10.1257/jel.51.4.1063'),
        ('Rossi: parameter instability', 'https://doi.org/10.1017/S1365100506050085'),
        ('Montero-Manso and Hyndman: global forecasting', 'https://doi.org/10.1016/j.ijforecast.2021.03.028'),
        ('Joulani et al.: delayed feedback', 'https://proceedings.mlr.press/v28/joulani13.html'),
        ('Flaspohler et al.: optimism and delay', 'https://proceedings.mlr.press/v139/flaspohler21a.html'),
        ('Tahmasbi et al.: DriftSurf', 'https://proceedings.mlr.press/v139/tahmasbi21a.html'),
        ('Forecast averaging under structural breaks', 'https://doi.org/10.1007/s00181-021-02137-w'),
        ('Discrete-time survival analysis', 'https://doi.org/10.1093/acprof:oso/9780195337518.003.0003'),
        ('White: Reality Check', 'https://doi.org/10.1111/1468-0262.00152'),
        ('Hansen: SPA', 'https://doi.org/10.1198/073500105000000063'),
        ('Hansen-Lunde-Nason: MCS', 'https://doi.org/10.3982/ECTA5771'),
        ('Giacomini-White: conditional predictive ability', 'https://doi.org/10.1111/j.1468-0262.2006.00718.x'),
    ]
    story += [para(f'{i}. <link href="{url}">{label}</link>', s, "SmallR3")
              for i, (label, url) in enumerate(sources, 1)]
    story += [Spacer(1, 4 * mm), para("Главные локальные артефакты", s, "H2R3")]
    story += bullets([
        "results/research/round3/master_policy_metrics.csv - единая таблица политик.",
        "results/research/round3/master_final_breakdown.csv - годовые и валютные срезы.",
        "results/research/round3/round3_block_bootstrap.csv - интервалы.",
        "results/research/round3/round3_circular_shift_multiplicity.csv - перебор 157 политик.",
        "results/research/publication_timing_h1.csv - after-publication по всем h.",
        "results/research/round3/report-source.md - полный текст и журнал выводов.",
        "results/research/round3/claim-source-ledger.md - связь утверждений с источниками.",
    ], s)
    story += [para("Запуск: `.venv/bin/python -m research.round3_master_audit`, затем "
                   "`.venv/bin/python -m research.build_round3_report`; тесты: "
                   "`.venv/bin/python -m pytest -c pytest.ini`.", s),
              Spacer(1, 5 * mm), callout(
        "Полный перебор параметров сохранён в отдельных stage1, stage2_2022_2023 и "
        "final_2024_2026_retrospective CSV. Отчёт сознательно агрегирует их по семействам, "
        "чтобы не скрывать отрицательные эксперименты и не выдавать лучший из сотен запусков за "
        "единственную заранее выбранную модель.", s)]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(PDF)
    return PDF


if __name__ == "__main__":
    build()
