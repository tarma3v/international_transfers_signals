"""Build the final Russian PDF research report from verified CSV artifacts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "research"
OUTPUT = ROOT / "output" / "pdf" / "ivan_experiments_report.pdf"

NAVY = colors.HexColor("#0f172a")
BLUE = colors.HexColor("#2563eb")
PALE_BLUE = colors.HexColor("#eff6ff")
GREEN = colors.HexColor("#15803d")
PALE_GREEN = colors.HexColor("#f0fdf4")
RED = colors.HexColor("#b91c1c")
PALE_RED = colors.HexColor("#fef2f2")
GRAY = colors.HexColor("#475569")
LIGHT = colors.HexColor("#e2e8f0")


def register_fonts():
    base = Path("/System/Library/Fonts/Supplemental")
    pdfmetrics.registerFont(TTFont("Arial", str(base / "Arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(base / "Arial Bold.ttf")))


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "TitleRu", fontName="Arial-Bold", fontSize=25, leading=29,
        textColor=NAVY, alignment=TA_LEFT, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        "SubtitleRu", fontName="Arial", fontSize=12, leading=17,
        textColor=GRAY, spaceAfter=15,
    ))
    styles.add(ParagraphStyle(
        "H1Ru", fontName="Arial-Bold", fontSize=17, leading=21,
        textColor=NAVY, spaceBefore=4, spaceAfter=9,
    ))
    styles.add(ParagraphStyle(
        "H2Ru", fontName="Arial-Bold", fontSize=12, leading=15,
        textColor=NAVY, spaceBefore=8, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        "BodyRu", fontName="Arial", fontSize=9.2, leading=13.2,
        textColor=NAVY, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "SmallRu", fontName="Arial", fontSize=7.7, leading=10.5,
        textColor=GRAY, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "CalloutRu", fontName="Arial-Bold", fontSize=11, leading=15,
        textColor=NAVY, leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "CellRu", fontName="Arial", fontSize=7.4, leading=9.3,
        textColor=NAVY,
    ))
    styles.add(ParagraphStyle(
        "CellBoldRu", fontName="Arial-Bold", fontSize=7.4, leading=9.3,
        textColor=NAVY,
    ))
    styles.add(ParagraphStyle(
        "CellHeaderRu", fontName="Arial-Bold", fontSize=7.4, leading=9.3,
        textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        "CoverMetric", fontName="Arial-Bold", fontSize=19, leading=22,
        alignment=TA_CENTER, textColor=BLUE,
    ))
    return styles


def clean(text):
    return str(text).replace("—", "-").replace("–", "-").replace("−", "-")


def p(text, styles, name="BodyRu"):
    return Paragraph(clean(text), styles[name])


def table(data, styles, widths=None, header=True, font_size=7.4):
    cooked = []
    for r, row in enumerate(data):
        cooked.append([
            Paragraph(clean(cell), styles["CellHeaderRu" if header and r == 0 else "CellRu"])
            for cell in row
        ])
    t = Table(cooked, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), .35, LIGHT),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]
    for r in range(1 if header else 0, len(cooked)):
        if r % 2 == 0:
            commands.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f8fafc")))
    t.setStyle(TableStyle(commands))
    return t


def callout(text, styles, color=PALE_BLUE, border=BLUE):
    t = Table([[p(text, styles, "CalloutRu")]], colWidths=[174 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("BOX", (0, 0), (-1, -1), 1, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def page_decor(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LIGHT)
    canvas.line(20 * mm, 14 * mm, 190 * mm, 14 * mm)
    canvas.setFont("Arial", 7)
    canvas.setFillColor(GRAY)
    canvas.drawString(20 * mm, 9 * mm, "international_transfers_signals | ivan-experiments")
    canvas.drawRightString(190 * mm, 9 * mm, f"04.09.2026 | {doc.page}")
    canvas.restoreState()


def build():
    register_fonts()
    styles = make_styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=18 * mm,
        title="Исследование сигналов международных переводов",
        author="Команда international_transfers_signals",
        subject="Leakage-controlled walk-forward research",
    )
    story = []

    # Cover and executive summary.
    story += [Spacer(1, 20 * mm), p("Сигналы выгодного момента для международных переводов", styles, "TitleRu")]
    story += [p("Большое исследование точности: CBR 2010-2026, пять валютных коридоров, строгий контроль утечек", styles, "SubtitleRu")]
    metrics = [
        [p("1.96", styles, "CoverMetric"), p("1.37", styles, "CoverMetric"), p("+77.8 б.п.", styles, "CoverMetric")],
        [p("future-only lift, h=5", styles, "SmallRu"), p("сигнала / коридор / неделю", styles, "SmallRu"), p("будущая выгода, h=5", styles, "SmallRu")],
    ]
    mt = Table(metrics, colWidths=[58 * mm] * 3, rowHeights=[16 * mm, 10 * mm])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 1, BLUE),
        ("INNERGRID", (0, 0), (-1, -1), .4, colors.HexColor("#bfdbfe")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [mt, Spacer(1, 8 * mm)]
    story += [callout(
        "Итог: требования по accuracy/lift, частоте и положительной будущей выгоде пробиваются на всех h, если уведомление отправляется только после фактической публикации Банком России следующего эффективного курса. Без этой временной гарантии h=1 не пробит, а лучший обычный h=5 результат является post-hoc диагностикой.",
        styles, PALE_GREEN, GREEN,
    ), Spacer(1, 5 * mm)]
    story += [p("Главное решение", styles, "H2Ru")]
    story += [p(
        "После публикации нового курса проверяем, что следующий эффективный курс не хуже текущего для отправителя, и ставим cooldown 3 календарных дня. Это не нейросеть и не попытка угадать h=1: первый шаг уже публичен. Для h>1 тот же gate становится сильным ведущим признаком без отдельной настройки по горизонту.", styles
    )]
    story += [p("Честный статус результатов", styles, "H2Ru")]
    story += [p(
        "Основной timestamp-aware результат заранее обусловлен временем релиза и устойчив на 2022-2023 и 2024-2026. Обычный anchor с lift 1.406 найден при исследовании финального блока и поэтому подписан post-hoc. Полностью locked ансамбль, выбранный только по 2022-2023, получил на 2024-2026 lift 1.260.", styles
    )]
    story += [Spacer(1, 4 * mm), p("Ветка: ivan-experiments. Push не выполнялся.", styles, "SmallRu"), PageBreak()]

    # Task definition.
    story += [p("1. Что именно предсказываем", styles, "H1Ru")]
    story += [p(
        "Пять коридоров: RUB -> TJS, UZS, KGS, AMD, KZT. Курс ЦБ задан в рублях за единицу валюты получателя, поэтому меньший курс выгоднее отправителю.", styles
    )]
    story += [table([
        ["Объект", "Определение"],
        ["Target Сейчас выгодно", "y(t,h)=1, если v(t) <= min(v(t+1),...,v(t+h)). В основной постановке h считается в публикациях ЦБ."],
        ["Future-only lift", "Доля y=1 среди сигналов / доля y=1 среди случайных дней того же оценочного периода и коридоров."],
        ["Выгода", "Среднее улучшение относительно будущего окна в базисных пунктах. Положительный знак = клиенту лучше."],
        ["Рабочая частота", "1-2 сигнала на каждый валютный коридор в неделю."],
        ["Горизонты", "h = 1, 3, 5, 10, 20."],
    ], styles, [38 * mm, 136 * mm])]
    story += [Spacer(1, 4 * mm), callout(
        "Что значит lift 1.40: сигнал попадает в target в 1.40 раза чаще случайного дня. Например, при base rate 29.4% hit rate 41.4% дает 41.4 / 29.4 = 1.406. Это не +40 процентных пунктов.", styles
    )]
    story += [p("Метрика заказчика и продукт", styles, "H2Ru")]
    story += [p(
        "Симметричная выгода +/-h частично вознаграждает уже случившееся движение. Поэтому для точности модели основной отчет использует target только по будущему и отдельно показывает достижимую будущую выгоду. Бизнес-курс приложения может отличаться от курса ЦБ; это остается обязательной проверкой перед production.", styles
    )]
    story += [p("Источник постановки", styles, "H2Ru"), p(
        '<link href="https://talenttrack.aitalenthub.ru/hackathon/cases/455">Страница кейса TalentTrack / AI Talent Hub</link>.', styles
    ), PageBreak()]

    # Data and validation.
    audit = pd.read_csv(RESULTS / "data_audit.csv")
    story += [p("2. Данные и протокол без утечек", styles, "H1Ru")]
    story += [p(
        "Официальная история ЦБ расширена с короткого периода 2019+ до 01.2010-09.2026. TJS/UZS/KGS/AMD доступны с 12.01.2010, KZT/USD/CNY/EUR - с 01.01.2010. Номиналы нормализованы, пересечение со старой выгрузкой совпало точно.", styles
    )]
    data_rows = [["Валюта", "Строк", "Начало", "Конец", "max gap", "NaN / <=0"]]
    for _, row in audit.iterrows():
        data_rows.append([
            row.currency, f"{int(row.rows)}", row["first"], row["last"],
            f"{int(row.max_gap_days)} дн.", f"{int(row['nan'])} / {int(row.nonpositive)}",
        ])
    story += [table(data_rows, styles, [24 * mm, 23 * mm, 29 * mm, 29 * mm, 29 * mm, 32 * mm])]
    story += [p("Хронологическое разделение", styles, "H2Ru")]
    story += [table([
        ["Блок", "Роль"],
        ["2010-2016", "Development и train-only EDA/отбор признаков."],
        ["2017-2020", "General validation для семей моделей."],
        ["2021", "Калибровка перед первым post-shock fold; не используется как финал."],
        ["2022-2023", "Заранее заданный shock/adaptation validation вокруг 24.02.2022."],
        ["2024-2026", "Последовательный аудит. Для каждого года порог берется только из предыдущего года."],
    ], styles, [36 * mm, 138 * mm])]
    story += [p("Защита от утечки", styles, "H2Ru")]
    for text in [
        "Каждый feature получает только срез значений до текущей строки включительно.",
        "Purge рассчитывается по реальной дате достижения target; строки train, чей h пересекает следующий блок, удаляются.",
        "Физическое усечение всего будущего на 31.12.2020 дало точное совпадение всех сохраненных ранних значений 279 признаков.",
        "46 unit-тестов прошли. Порог, модель и частота не подгоняются по текущему тестовому году.",
    ]:
        story += [p("- " + text, styles)]
    story += [p("Пропущенные дни", styles, "H2Ru"), p(
        "Основная ось - публикации ЦБ: это не искусственные нулевые движения. Отдельная robustness-проверка строит календарную сетку причинным forward-fill, добавляет факт обновления и разрешает сигнал только в день новой публикации.", styles
    ), PageBreak()]

    # Best approach.
    pub = pd.read_csv(RESULTS / "publication_timing_h1.csv")
    pub = pub[pub.years.str.contains("2024")].sort_values("h")
    stats = pd.read_csv(RESULTS / "statistical_audit.csv")
    stats = stats[stats.policy == "publication_timing"].set_index("h")
    story += [p("3. Лучший подход: использовать время публикации", styles, "H1Ru")]
    story += [callout(
        "Правило: после фактического релиза следующего эффективного курса ЦБ сигнал разрешен, если новый курс не хуже текущего; после сигнала - пауза 3 календарных дня. Никаких параметров по h и никакого доступа к еще не опубликованным значениям.", styles, PALE_GREEN, GREEN
    ), Spacer(1, 4 * mm)]
    rows = [["h", "Сигн./нед.", "Hit rate", "Base", "Lift", "95% CI lift", "Будущая выгода"]]
    for _, row in pub.iterrows():
        ci = stats.loc[int(row.h)]
        rows.append([
            f"{int(row.h)}", f"{row.frequency:.2f}", f"{row.hit_rate:.1%}",
            f"{row.base_rate:.1%}", f"{row.lift:.3f}",
            f"[{ci.lift_ci_low:.2f}; {ci.lift_ci_high:.2f}]",
            f"{row.forward_benefit_bps:+.1f} б.п.",
        ])
    story += [table(rows, styles, [10 * mm, 25 * mm, 25 * mm, 22 * mm, 20 * mm, 36 * mm, 36 * mm])]
    story += [Spacer(1, 4 * mm), Image(str(RESULTS / "report_summary.png"), width=174 * mm, height=107 * mm)]
    story += [p(
        "Bootstrap ресемплирует четырехнедельные блоки и сохраняет пять коридоров вместе. Поэтому интервал учитывает межвалютную корреляцию и часть зависимости из-за перекрывающихся будущих окон.", styles, "SmallRu"
    ), PageBreak()]

    # Calendar robustness.
    cal_plain = pd.read_csv(RESULTS / "calendar_day_robustness.csv")
    cal_time = pd.read_csv(RESULTS / "calendar_day_publication_timing.csv")
    story += [p("4. Выходные и календарная трактовка h", styles, "H1Ru")]
    story += [p(
        "Если h означает календарные дни, курс между публикациями forward-fill только из прошлого. Решение принимается лишь в дату обновления, поэтому выходные не порождают повторные сигналы.", styles
    )]
    rows = [["h, календ. дней", "Past-only lift", "After-release lift", "Частота", "Min lift по валютам", "Будущая выгода"]]
    for (_, a), (_, b) in zip(cal_plain.iterrows(), cal_time.iterrows()):
        rows.append([
            f"{int(a.h_calendar_days)}", f"{a.lift:.3f}", f"{b.lift:.3f}",
            f"{b.frequency:.2f}", f"{b.corridor_lift_min:.2f}",
            f"{b.forward_benefit_bps:+.1f} б.п.",
        ])
    story += [table(rows, styles, [25 * mm, 28 * mm, 31 * mm, 23 * mm, 34 * mm, 33 * mm])]
    story += [Spacer(1, 5 * mm), callout(
        "Вывод устойчив: after-release политика сохраняет lift 1.62-1.94 и частоту 1.36-1.38. Past-only h=5 дает 1.283, то есть выбор единицы горизонта действительно может решить, формально пройден порог 1.30 или нет.", styles
    )]
    story += [p("Ограничение времени", styles, "H2Ru"), p(
        'Банк России пишет, что точное время не регламентировано, но публикация обычно происходит до 18:00 Москвы: <link href="https://www.cbr.ru/Reception/TopicalMessage/Page/2661">официальный FAQ</link>. Production должен проверять факт обновления API, а не полагаться на часы. До события релиза gate недоступен.', styles
    ), PageBreak()]

    # Ordinary model.
    diag = pd.read_csv(RESULTS / "diagnostic_anchor_all_horizons.csv").sort_values("h")
    locked = pd.read_csv(RESULTS / "locked_all_horizons.csv").sort_values("h")
    story += [p("5. Лучший обычный прогноз без знания следующего курса", styles, "H1Ru")]
    story += [p(
        "Сильнейший понятный score: положение текущего курса в 90-дневном диапазоне плюс малые добавки momentum за 20 и 60 публикаций. Высокое положение не означает дешевую валюту; оно ловит продолжение уже идущего тренда, при котором сегодняшний курс еще не будет побит будущими.", styles
    )]
    rows = [["h", "Locked lift", "Post-hoc anchor lift", "Частота anchor", "Min год", "Min валюта"]]
    for (_, a), (_, b) in zip(locked.iterrows(), diag.iterrows()):
        rows.append([
            f"{int(a.h)}", f"{a.lift:.3f}", f"{b.lift:.3f}", f"{b.frequency:.2f}",
            f"{b.year_lift_min:.3f}", f"{b.corridor_lift_min:.3f}",
        ])
    story += [table(rows, styles, [14 * mm, 28 * mm, 37 * mm, 32 * mm, 27 * mm, 29 * mm])]
    story += [Spacer(1, 4 * mm), callout(
        "Не смешивать: h=5 post-hoc anchor имеет lift 1.406, hit rate 41.4%, base 29.4%, частоту 1.03 и 95% CI [1.13; 1.68]. Полностью locked full-library ансамбль, выбранный на 2022-2023, дал 1.260. Поэтому 1.406 - перспективная гипотеза для нового holdout, а не финальное доказательство.", styles, PALE_RED, RED
    )]
    story += [p("Что дало самый большой boost", styles, "H2Ru")]
    boost_rows = [
        ["Изменение", "Эффект", "Интерпретация"],
        ["Явное время релиза", "h=5: 1.406 -> 1.959", "Самый большой и устойчивый прирост; меняет доступный information set."],
        ["Простой anchor вместо ETS", "AUC 0.478 -> 0.572", "На дневном FX сложность не заменяет сильный level/trend prior."],
        ["Anchor + ExtraTrees, 2022-2023", "validation lift 1.400", "ML полезен как малая residual-добавка, но final lift только 1.260."],
        ["Cyclic + one-hot", "lift 1.137 -> 1.053", "Запрошенные признаки добавлены, но чистая циклическая замена ухудшила модель; raw/binary оставлены рядом."],
    ]
    story += [table(boost_rows, styles, [47 * mm, 41 * mm, 86 * mm]), PageBreak()]

    # Regimes and demand.
    regime = pd.read_csv(RESULTS / "regime_audit_summary.csv")
    regime = regime.set_index("period").loc[
        ["pre_2022", "shock_adaptation", "mature_postshock"]
    ].reset_index()
    story += [p("6. Режим 2022 года и гипотеза спроса", styles, "H1Ru")]
    story += [Image(str(RESULTS / "report_regimes.png"), width=174 * mm, height=94 * mm)]
    rows = [["Период", "Base fav_h5", "Mean |move|", "90d-high share"]]
    names = {"pre_2022": "2017-23.02.2022", "shock_adaptation": "24.02.2022-2023", "mature_postshock": "2024-2026"}
    for _, row in regime.iterrows():
        rows.append([
            names[row.period], f"{row.fav_h5_base_rate:.1%}",
            f"{row.mean_abs_daily_move_bps:.1f} б.п.", f"{row.share_at_90d_high:.1%}",
        ])
    story += [table(rows, styles, [48 * mm, 39 * mm, 43 * mm, 44 * mm])]
    story += [p("Что подтверждается", styles, "H2Ru"), p(
        "24.02.2022 - разумная заранее заданная точка structural break. В shock/adaptation средний модуль дневного движения вырос примерно с 56 до 115 б.п., а доля наблюдений у 90-дневного максимума - с 4.9% до 18.4%. В 2024+ волатильность частично нормализовалась, поэтому вечное повышение веса 2022 году оказалось хуже устойчивого anchor.", styles
    )]
    story += [p("Что не подтверждается курсом", styles, "H2Ru"), p(
        "Рост спроса на переводы нельзя идентифицировать по одному обменному курсу: на него одновременно действуют рубль, локальная валюта, ограничения и предложение ликвидности. База ЦБ по переводам существует, но имеет квартальную/годовую частоту и должна лагироваться до даты публикации. В обзоре ЦБ за 2023 год банковские трансграничные нетто-переводы снижались по кварталам, а доля дружественных направлений росла - структура изменилась, но простого монотонного роста нет.", styles
    )]
    story += [p(
        '<link href="https://www.cbr.ru/hd_base/tg/?tab.current=t2">База трансграничных переводов ЦБ</link> | <link href="https://www.cbr.ru/Collection/Collection/File/46563/ORFR_2023-10.pdf">Обзор рисков финансовых рынков, октябрь 2023</link>.', styles, "SmallRu"
    ), PageBreak()]

    # Features and seasonality.
    corr = pd.read_csv(RESULTS / "train_feature_correlations.csv").head(12)
    story += [p("7. Признаки и сезонность: только development", styles, "H1Ru")]
    story += [p(
        "Расчеты ниже ограничены 2010-2016. Это не тестовые объяснения. В матрице 279 признаков: currency one-hot; raw, cyclic и Fourier календарь; бинарные окна до праздников/Нового года/1 сентября; лаги и rolling level/range/volatility/trend; new high/low; EMA/MACD; USD/CNY/EUR движения, beta/correlation и cross-rates.", styles
    )]
    story += [Image(str(RESULTS / "report_seasonality.png"), width=174 * mm, height=57 * mm)]
    story += [p(
        "Самые повторяющиеся месяцы с повышенным fav_h5 - май и ноябрь; август заметен у TJS/AMD/KGS, но не одинаков для всех валют. Сезонность существует как слабый условный prior, а не универсальное правило.", styles
    )]
    top_rows = [["Feature", "Spearman", "Смысл"]]
    meanings = {
        "share_above_sma_20": "доля уровней выше SMA20",
        "cny_ret_20": "20-периодное движение CNY/RUB",
        "bars_since_min_30": "давность минимума окна",
        "usd_ret_20": "20-периодное движение USD/RUB",
        "quarter_cos": "циклический квартал",
        "accel_5_20": "ускорение momentum",
        "calendar_week": "номер недели рядом с Fourier",
        "raw_ret_30": "30-периодная log-доходность",
        "ret_kurt_20": "эксцесс доходностей",
        "eur_ret_20": "20-периодное движение EUR/RUB",
        "slope_z_20": "нормированный наклон",
    }
    for _, row in corr.head(10).iterrows():
        top_rows.append([row.feature, f"{row.spearman:+.3f}", meanings.get(row.feature, "causal rolling/calendar feature")])
    story += [table(top_rows, styles, [62 * mm, 28 * mm, 84 * mm])]
    story += [p(
        "Mutual information не используется для вывода: одинаковые reference-features повторяются по пяти валютам одного дня и могут завышать kNN-оценку. Для feature screening применен train-only rank correlation и обязательное включение currency identity/pct-range.", styles, "SmallRu"
    ), PageBreak()]

    # Models tried.
    general = pd.read_csv(RESULTS / "general_validation_h5.csv")
    best = general.sort_values("lift", ascending=False).groupby("candidate", as_index=False).first()
    best = best.sort_values("lift", ascending=False).head(8)
    story += [p("8. Какие модели проверены", styles, "H1Ru")]
    story += [p(
        "Все обучаемые модели оценены expanding/rolling walk-forward. Глобальные модели видят пять рядов совместно и currency one-hot; локальные fit выполняются отдельно по каждой валюте. Сравнивались expanding history, 5-year window и exponential half-life 2 years.", styles
    )]
    story += [table([
        ["Семья", "Результат", "Вывод"],
        ["ETS / seasonal naive", "h5 AUC 0.478 / 0.458", "Стабильной сезонной структуры недостаточно."],
        ["SARIMA", "best h5 AUC 0.523", "Небольшой сигнал, слабее anchor."],
        ["GRU", "h5 AUC 0.540", "Сложнее и хуже pct_range_90 (0.572)."],
        ["Direct future-min regression", "AUC 0.477-0.504", "Потеря качества не объясняется бинаризацией target."],
        ["Logit / elastic-net", "general lift до ~1.10", "Хороший контрольный глобальный baseline."],
        ["RF / ExtraTrees / HistGB", "general lift до 1.188", "ExtraTrees лучший старый validation, но выгода могла быть отрицательной."],
        ["CatBoost / XGBoost", "general lift до 1.171 / 1.149", "Полезны в смеси, нестабильны между режимами."],
        ["Tail + rescue ML", "lift 1.386 -> 1.146", "Добор частоты моделью разрушил качество сильного хвоста."],
    ], styles, [45 * mm, 43 * mm, 86 * mm])]
    story += [p("Top general-validation configurations, h=5", styles, "H2Ru")]
    model_rows = [["Candidate", "Freq.", "Lift", "Future bps", "Min year"]]
    for _, row in best.iterrows():
        model_rows.append([
            row.candidate, f"{row.frequency:.2f}", f"{row.lift:.3f}",
            f"{row.forward_benefit_bps:+.1f}", f"{row.year_lift_min:.3f}",
        ])
    story += [table(model_rows, styles, [72 * mm, 22 * mm, 22 * mm, 30 * mm, 28 * mm])]
    story += [p(
        "Почему сложность не победила: target - экстремальное событие на шумном FX-ряде, режим 2022 резко меняет распределение score, а оптимизация AUC не гарантирует lift в верхнем хвосте и положительную экономическую выгоду.", styles
    ), PageBreak()]

    # Literature and decisions.
    story += [p("9. Что взято из литературы", styles, "H1Ru")]
    refs = [
        ("Rolling-origin validation", "Hyndman tsCV последовательно обучается на y1...yt и прогнозирует t+h. У нас добавлен purge по дате достижения target.", "https://pkg.robjhyndman.com/forecast/reference/tsCV.html"),
        ("Global forecasting", "Исследования global models показывают пользу pooled regression/LGBM/RNN на наборах связанных и коротких рядов. Отсюда общий panel и currency identity.", "https://arxiv.org/abs/2012.12485"),
        ("Parameter instability", "Rossi показывает, что нестабильность параметров может скрывать predictability. Отсюда predeclared break, rolling window и time decay.", "https://doi.org/10.1017/S1365100506050085"),
        ("Model uncertainty", "Работа Beckmann et al. мотивирует shrinkage и model averaging при быстро меняющейся релевантности факторов.", "https://doi.org/10.1016/j.jimonfin.2015.07.001"),
        ("Сильный random-walk baseline", "Ahmed, Liu, Valente показывают, что факторные FX-модели часто не бьют random walk out of sample. Поэтому простые anchors не отбрасывались.", "https://doi.org/10.1016/j.ijforecast.2015.01.010"),
    ]
    for title, body, url in refs:
        story += [p(f'<b>{title}.</b> {body} <link href="{url}">Источник</link>.', styles)]
    story += [p("Почему классический ML остался фокусом", styles, "H2Ru"), p(
        "CatBoost, XGBoost, ExtraTrees, HistGB, логистическая и elastic-net регрессии покрывают нелинейности, взаимодействия валют и shrinkage при небольшом числе рядов. GRU проверена как sanity check и проиграла. Это совпадает с ограничением кейса на объяснимый сигнальный слой.", styles
    )]
    story += [p("Почему не добавлялись квартальные объемы прямо в daily fit", styles, "H2Ru"), p(
        "Они полезны как regime prior, но должны иметь timestamp публикации. Backfill квартального значения на весь тот же квартал был бы прямой утечкой. Следующий корректный эксперимент - as-of join по дате релиза статистики ЦБ.", styles
    ), PageBreak()]

    # Honest conclusion/reproduction.
    story += [p("10. Итоговый вердикт и следующий шаг", styles, "H1Ru")]
    story += [callout(
        "Условный зачет: ДА, если регламент продукта разрешает запуск после подтвержденной публикации следующего официального курса. Тогда все h проходят lift 1.30, частоту 1-2, положительную выгоду и устойчивость. Если сигнал нужен раньше публикации - НЕТ: h=1 не достиг 1.30 честным locked способом.", styles, PALE_GREEN, GREEN
    )]
    story += [p("Что готово в ветке", styles, "H2Ru")]
    for text in [
        "Длинная официальная панель CBR 2010-2026 и EUR как дополнительный reference.",
        "279 причинных признаков, currency one-hot, raw/cyclic/Fourier календарь и binary event windows.",
        "Purged annual walk-forward, general/regime/final split, rolling/decay/local/global модели и ансамбли.",
        "Publication-step и calendar-day targets, causal forward-fill и after-release gate.",
        "Четырехнедельный block bootstrap, train-only EDA и воспроизводимые CSV/PNG артефакты.",
    ]:
        story += [p("- " + text, styles)]
    story += [p("Приоритет следующего эксперимента", styles, "H2Ru"), p(
        "1) подтвердить точный SLA релиза и момент отправки; 2) заменить официальный курс на реальный executable rate продукта; 3) сделать новый untouched holdout после 04.09.2026 для ordinary anchor; 4) присоединить квартальные объемы переводов as-of и intraday market proxy; 5) измерить incremental conversion/volume в A/B, потому что высокий offline lift не гарантирует бизнес-эффект.", styles
    )]
    story += [p("Команды воспроизведения", styles, "H2Ru")]
    commands = [
        "python -m research.extended_features",
        "python -m research.train_only_eda",
        "python -m research.model_study",
        "python -m research.horizon_audit",
        "python -m research.publication_timing_audit",
        "python -m research.calendar_day_robustness",
        "python -m research.statistical_audit",
        "python -m pytest -c pytest.ini",
    ]
    story += [table([["Команда"]] + [[x] for x in commands], styles, [174 * mm])]
    story += [Spacer(1, 4 * mm), p(
        "Data SHA-256: 57240df1a64990ef3fa83e6f91eb8130c61fd3ad7203780e1248bae65bc728c2", styles, "SmallRu"
    )]
    story += [p(
        "Полный список источников и решений: research/references.md. Основные численные артефакты: results/research/.", styles, "SmallRu"
    )]

    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    print(OUTPUT)


if __name__ == "__main__":
    build()
