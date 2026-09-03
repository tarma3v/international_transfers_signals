"""Build the verified Russian PDF for the second deep-research round."""
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
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "research" / "round2"
PDF = ROOT / "output" / "pdf" / "ivan_deep_research_round2.pdf"

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


def clean(value) -> str:
    return (str(value).replace("—", "-").replace("–", "-")
            .replace("−", "-").replace("‑", "-"))


def fonts() -> None:
    base = Path("/System/Library/Fonts/Supplemental")
    pdfmetrics.registerFont(TTFont("Arial", str(base / "Arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(base / "Arial Bold.ttf")))


def styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("TitleR2", fontName="Arial-Bold", fontSize=25, leading=29,
                         textColor=NAVY, alignment=TA_LEFT, spaceAfter=10))
    s.add(ParagraphStyle("SubtitleR2", fontName="Arial", fontSize=11.5, leading=16,
                         textColor=GRAY, spaceAfter=14))
    s.add(ParagraphStyle("H1R2", fontName="Arial-Bold", fontSize=17, leading=20,
                         textColor=NAVY, spaceBefore=4, spaceAfter=8))
    s.add(ParagraphStyle("H2R2", fontName="Arial-Bold", fontSize=11.5, leading=14,
                         textColor=NAVY, spaceBefore=7, spaceAfter=4))
    s.add(ParagraphStyle("BodyR2", fontName="Arial", fontSize=9.2, leading=13,
                         textColor=NAVY, spaceAfter=6))
    s.add(ParagraphStyle("SmallR2", fontName="Arial", fontSize=7.4, leading=9.5,
                         textColor=GRAY, spaceAfter=3))
    s.add(ParagraphStyle("CalloutR2", fontName="Arial-Bold", fontSize=10.2, leading=14,
                         textColor=NAVY, leftIndent=4, rightIndent=4))
    s.add(ParagraphStyle("CellR2", fontName="Arial", fontSize=7.1, leading=9,
                         textColor=NAVY))
    s.add(ParagraphStyle("CellHeaderR2", fontName="Arial-Bold", fontSize=7.1, leading=9,
                         textColor=colors.white))
    s.add(ParagraphStyle("MetricR2", fontName="Arial-Bold", fontSize=18, leading=21,
                         alignment=TA_CENTER, textColor=BLUE))
    return s


def para(text, s, style="BodyR2"):
    return Paragraph(clean(text), s[style])


def tbl(data, s, widths=None, header=True):
    cooked = []
    for i, row in enumerate(data):
        style = "CellHeaderR2" if header and i == 0 else "CellR2"
        cooked.append([Paragraph(clean(cell), s[style]) for cell in row])
    table = Table(cooked, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), .35, LIGHT),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
                     ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    for i in range(1 if header else 0, len(cooked)):
        if i % 2 == 0: commands.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8fafc")))
    table.setStyle(TableStyle(commands))
    return table


def callout(text, s, fill=PALE_BLUE, border=BLUE):
    table = Table([[para(text, s, "CalloutR2")]], colWidths=[174 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill), ("BOX", (0, 0), (-1, -1), 1, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def footer(canvas, doc):
    canvas.saveState(); canvas.setStrokeColor(LIGHT)
    canvas.line(20 * mm, 14 * mm, 190 * mm, 14 * mm)
    canvas.setFont("Arial", 7); canvas.setFillColor(GRAY)
    canvas.drawString(20 * mm, 9 * mm, "international_transfers_signals | ivan-experiments")
    canvas.drawRightString(190 * mm, 9 * mm, f"04.09.2026 | {doc.page}")
    canvas.restoreState()


def build() -> Path:
    fonts(); s = styles(); PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=17*mm, bottomMargin=18*mm,
                            title="Глубокое исследование h=5 без утечек",
                            author="international_transfers_signals")
    story = []

    # Cover.
    story += [Spacer(1, 18*mm), para("Глубокое исследование h=5", s, "TitleR2"),
              para("Новые модели, режимы и ансамбли без заглядывания в будущее", s, "SubtitleR2")]
    metric = [
        [para("1.287", s, "MetricR2"), para("1.328*", s, "MetricR2"), para("76", s, "MetricR2")],
        [para("validation lift, 2022-23", s, "SmallR2"),
         para("retrospective lift, 2024-26", s, "SmallR2"),
         para("recorded configurations", s, "SmallR2")],
    ]
    m = Table(metric, colWidths=[58*mm]*3, rowHeights=[15*mm, 10*mm])
    m.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), PALE_BLUE),
                           ("BOX", (0,0), (-1,-1), 1, BLUE),
                           ("INNERGRID", (0,0), (-1,-1), .4, colors.HexColor("#bfdbfe")),
                           ("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
    story += [m, Spacer(1, 7*mm), callout(
        "Результат: лучший переносимый новый подход - простая мягкая смесь разных экспертов. "
        "Она дает lift 1.287 при 1.52 сигнала/коридор/неделю на 2022-2023 и 1.328 "
        "ретроспективно на 2024-2026. Это сильный кандидат, но не доказанное улучшение над "
        "ранее зафиксированным anchor 1.295.", s, PALE_GREEN, GREEN), Spacer(1, 5*mm)]
    story += [para("Почему не пишем \"мы пробили 1.40\"", s, "H2R2"), para(
        "Максимальный новый retrospective lift 1.383 получен динамическими глобальными весами, "
        "но один год имеет lift 0.913. После circular-shift max-поправки по 76 конфигурациям "
        "p=0.059 на 2024-2026 и p=0.305 на 2022-2023. Парные интервалы превосходства над anchor "
        "также пересекают ноль. Поэтому headline остается консервативным.", s)]
    story += [para("* 2024-2026 уже просматривались в первом исследовании; для новых идей это не holdout.", s, "SmallR2"),
              Spacer(1, 4*mm), para("Ветка ivan-experiments. Push не выполнялся.", s, "SmallR2"), PageBreak()]

    # Task and protocol.
    story += [para("1. Задача и target", s, "H1R2"), para(
        "На каждой публикации ЦБ и для каждого коридора TJS, UZS, KGS, AMD, KZT модель оценивает: "
        "fav_h5(t)=1, если текущий нормированный курс не выше каждого из следующих пяти опубликованных "
        "курсов. Меньший курс выгоднее отправителю рублей. h=5 - публикации, не календарные дни.", s)]
    story += [tbl([
        ["Показатель", "Определение / ограничение"],
        ["Lift", "hit rate среди сигналов / base rate на том же OOS-периоде"],
        ["Future-only benefit", "улучшение текущего курса относительно будущего окна, базисные пункты"],
        ["Рабочая частота", "1-2 сигнала на валютный коридор в неделю"],
        ["Главный горизонт", "h=5 публикаций ЦБ"],
        ["Данные", "официальные курсы ЦБ, 2010-01-01 - 2026-09-02; номинал нормирован построчно"],
    ], s, [42*mm, 132*mm])]
    story += [Spacer(1, 4*mm), callout(
        "Lift 1.40 означает \"в 1.4 раза чаще случайного дня\". При base rate 29.4% это примерно "
        "41.2% попаданий, а не +40 процентных пунктов.", s)]
    story += [para("Хронологический протокол", s, "H2R2"), tbl([
        ["Период", "Роль"], ["2010-2016", "development и train-only EDA"],
        ["2017-2020", "general validation новых семейств"],
        ["2021", "калибровка перед shock-блоком"],
        ["2022-2023", "shock/adaptation validation"],
        ["2024-2026", "ретроспективный аудит новых идей; не новый holdout"],
    ], s, [42*mm, 132*mm])]
    story += [para(
        "Для тестового года Y train заканчивается раньше Y-1 и очищается по фактической дате, до которой "
        "дотягивается h=5 target. Порог обучается только на Y-1. Признаки видят текущую и прошлые "
        "публикации. Тесты портят будущее и требуют точного совпадения прошлого.", s),
        para('<link href="https://talenttrack.aitalenthub.ru/hackathon/cases/455">Кейс</link> | '
             '<link href="https://www.cbr.ru/currency_base/dynamics/">источник курсов ЦБ</link>.', s, "SmallR2"), PageBreak()]

    # Data structure.
    story += [para("2. Структура данных и режимы", s, "H1R2"),
              Image(str(OUT / "report_data_structure.png"), width=174*mm, height=67*mm)]
    story += [para("Общий фактор", s, "H2R2"), para(
        "Первая главная компонента стандартизованных дневных движений объясняет 70% вариации в "
        "2017-2020, 84% в 2022-2023 и 92% в 2024-2026. Нагрузки имеют один знак. Это сильный аргумент "
        "в пользу global pooling и residual-слоя. При этом лаговые peer-движения слабы и нестабильны.", s)]
    story += [para("Сезонность", s, "H2R2"), para(
        "Порядок месяцев по hit rate почти не переносится между блоками: корреляции для отдельных валют "
        "около нуля и часто отрицательны. Cyclic sin/cos остаются допустимыми слабыми признаками, но "
        "отдельная seasonal policy не используется.", s)]
    story += [para("2022 и пропущенные дни", s, "H2R2"), para(
        "Оффлайн-скан подтверждает разрывы около 2021-2022 и конца 2022, но найденная с будущим дата не "
        "может быть feature. Причинные режимы задаются трендом, относительной волатильностью, положением "
        "в диапазоне и общим фактором. Выходные не считаются пропусками: основная ось - публикации ЦБ, "
        "а длина календарного разрыва входит отдельным gap_days.", s)]
    story += [para(
        'Стабильность валютной предсказуемости зависит от горизонта, sample и evaluation design: '
        '<link href="https://doi.org/10.1257/jel.51.4.1063">Rossi (2013)</link>. Structural breaks '
        'мотивируют rolling/EWMA и комбинации окон: <link href="https://www.ifo.de/en/cesifo/publications/2008/working-paper/forecasting-random-walks-under-drift-instability">Pesaran-Pick</link>.', s, "SmallR2"), PageBreak()]

    # Model comparison.
    story += [para("3. Новые семейства моделей", s, "H1R2"),
              Image(str(OUT / "report_model_comparison.png"), width=174*mm, height=75*mm)]
    story += [para("19 независимых базовых архитектур", s, "H2R2"), para(
        "Проверены local/global logistic и spline-logistic, shrinkage LDA, KNN-аналог траекторий, "
        "HistGradientBoosting, ExtraTrees, GMM regime mixture, discrete survival и quantile future-floor. "
        "Лучший отдельный эксперт - global ExtraTrees: lift 1.367 на 2022-2023 при частоте 1.00, затем "
        "1.254 ретроспективно.", s), Spacer(1, 3*mm)]
    story += [para("Local model -> global residual booster", s, "H2R2"), para(
        "Точная пользовательская идея реализована через локальный logit каждой валюты и глобальный XGB "
        "offset, обученный только на старых OOF-ошибках. Результат: 1.075 на 2022-2023 и 1.110 на "
        "2024-2026. Статическая глобальная поправка переносит устаревшие зависимости.", s),
        Spacer(1, 3*mm)]
    story += [para("Короткие окна", s, "H2R2"), para(
        "ExtraTrees window3 дает 1.355 на shock-блоке, но имеет финальный год 0.741. Смесь окон 2/3/5 "
        "дает 1.384, однако только 0.864 сигнала в неделю - ниже условия. На последнем блоке она "
        "стабильнее: 1.277 при 1.165 сигнала в неделю.", s)]
    story += [para(
        'Теория global pooling: <link href="https://doi.org/10.1016/j.ijforecast.2021.03.028">Montero-Manso и Hyndman</link>. '
        'Обзор комбинаций прогнозов: <link href="https://arxiv.org/abs/2205.04216">Wang et al.</link>.', s, "SmallR2"), PageBreak()]

    # Router.
    story += [para("4. Error-routed mixture of experts", s, "H1R2"), callout(
        "Идея: сохранять строго OOF-ошибки каждой модели, описывать текущий причинно наблюдаемый режим и "
        "мягко повышать вес модели, которая раньше была лучше в похожем состоянии.", s, PALE_GREEN, GREEN)]
    story += [para("Второй слой OOF", s, "H2R2"), para(
        "Для года Y gate обучается только на test-fold прогнозах годов раньше Y-1. Год Y-1 не попадает "
        "в gate train и остается калибровочным. Разные шкалы экспертов переводятся в покоридорные "
        "percentile ranks относительно предыдущего calibration fold.", s)]
    story += [tbl([
        ["Router", "2022-23", "2024-26*", "Слабое место"],
        ["Equal experts", "1.287", "1.328", "2025 lift 0.882"],
        ["Soft regime", "1.243", "1.325", "средний lift ниже; зато min year 1.161"],
        ["Global trailing weights", "1.200", "1.383", "min year 0.913; result retrospective"],
        ["Hard regime", "слабее 1.12", "не finalist", "ошибка выбора одного эксперта слишком дорогая"],
        ["Learned ExtraTrees/Ridge gate", "0.98-1.09", "не finalist", "мало независимых режимов для gate"],
    ], s, [51*mm, 27*mm, 28*mm, 68*mm])]
    story += [Spacer(1, 4*mm), Image(str(OUT / "report_year_stability.png"), width=174*mm, height=75*mm)]
    story += [para(
        "Практический вывод: данные подтверждают различие профилей ошибок, но пока поддерживают soft "
        "shrinkage, а не hard switch. Это инженерный аналог conditional predictive ability, а не "
        '<link href="https://doi.org/10.1111/j.1468-0262.2006.00718.x">формальный тест Giacomini-White</link>.', s, "SmallR2"), PageBreak()]

    # Failures and external data.
    story += [para("5. Что не сработало", s, "H1R2")]
    story += [tbl([
        ["Гипотеза", "Лучший наблюдаемый результат", "Решение"],
        ["Post-24.02.2022 weight x4", "Extra 1.089; Hist 1.180 на 2024-26*", "не использовать бинарный regime flag"],
        ["Brent + broad USD + RUONIA + key rate", "до 1.313 старый блок; <=1.032 shock", "не включать в headline"],
        ["XGB pairwise/NDCG ranker", "до 1.375 general; 0.718-1.073 shock", "сильное regime overfit"],
        ["Discrete survival", "1.080 general", "не улучшает прямой target"],
        ["Future-floor quantile", "1.143 shock; <0.90 final у q25", "регрессия минимума нестабильна"],
        ["Local -> global tower", "1.075 shock; 1.110 final*", "статический residual устаревает"],
    ], s, [58*mm, 58*mm, 58*mm])]
    story += [para("Внешние данные и время доступности", s, "H2R2"), para(
        "RUONIA присоединена по явной DateUpdate. Для latest-vintage FRED-файлов проверены лаги "
        "Brent/Dollar 2/7, 5/10 и 7/14 календарных дней. Поскольку исторические vintages отсутствуют, "
        "это только sensitivity, не доказательство real-time доступности.", s)]
    story += [para(
        '<link href="https://www.cbr.ru/hd_base/ruonia/dynamics/">ЦБ: RUONIA</link> | '
        '<link href="https://www.cbr.ru/hd_base/keyrate/">ЦБ: key rate</link> | '
        '<link href="https://fred.stlouisfed.org/series/DCOILBRENTEU">FRED/EIA: Brent</link> | '
        '<link href="https://fred.stlouisfed.org/series/DTWEXBGS">FRED/Federal Reserve: broad dollar</link>.', s, "SmallR2")]
    story += [callout(
        "Отрицательные результаты сохранены намеренно. Удалить их и показать только 1.383 означало бы "
        "скрыть самую большую угрозу качеству - выбор победителя из десятков нестабильных backtests.", s,
        PALE_ORANGE, ORANGE), PageBreak()]

    # Statistics.
    stats = pd.read_csv(OUT / "round2_block_bootstrap.csv")
    final = stats[stats.period == "retrospective_2024_2026"].set_index("policy")
    order = ["anchor_multiscale_locked", "router_equal", "router_regime_soft",
             "router_global_soft", "recency_window_short"]
    labels = ["Locked anchor", "Equal experts", "Soft regime", "Trailing weights", "Short-window mix"]
    rows = [["Политика", "Freq.", "Lift [95% CI]", "Future bps [95% CI]", "Diff vs anchor CI"]]
    for name, label in zip(order, labels):
        r = final.loc[name]
        rows.append([label, f"{r.frequency:.2f}", f"{r.lift:.3f} [{r.lift_ci_low:.3f}; {r.lift_ci_high:.3f}]",
                     f"{r.forward_benefit_bps:+.1f} [{r.benefit_ci_low:+.1f}; {r.benefit_ci_high:+.1f}]",
                     f"[{r.lift_diff_vs_anchor_ci_low:+.3f}; {r.lift_diff_vs_anchor_ci_high:+.3f}]"])
    story += [para("6. Неопределённость и множественный перебор", s, "H1R2"),
              tbl(rows, s, [36*mm, 18*mm, 45*mm, 46*mm, 29*mm])]
    story += [para("Block bootstrap", s, "H2R2"), para(
        "Ресемплируются четырёхнедельные календарные блоки; все пять валют одного дня остаются вместе. "
        "Так сохраняются перекрывающиеся h=5 outcomes и общий валютный фактор. Equal, regime и trailing "
        "ансамбли имеют положительные отдельные интервалы, но интервалы разницы с anchor содержат ноль.", s)]
    story += [para("Max-поправка", s, "H2R2"), para(
        "В circular-shift negative control target сдвигается общей датой для всех валют, сохраняя свою "
        "автокорреляцию. На каждом сдвиге берётся лучший lift из 76 записанных policy. Нулевая 95%-граница "
        "максимума равна 1.517 на 2022-2023 и 1.388 на 2024-2026. Наблюдаемые maxima 1.384 и 1.383; "
        "скорректированные p=0.305 и 0.059.", s)]
    story += [callout(
        "Статистически честная формулировка: новый ансамбль показывает сильный и практически полезный "
        "сигнал, но текущих независимых режимов недостаточно, чтобы доказать его превосходство над locked anchor.",
        s, PALE_RED, RED)]
    story += [para(
        'Методы data-snooping: <link href="https://doi.org/10.1111/1468-0262.00152">White Reality Check</link>, '
        '<link href="https://doi.org/10.1198/073500105000000063">Hansen SPA</link>, '
        '<link href="https://doi.org/10.3982/ECTA5771">Model Confidence Set</link>.', s, "SmallR2"), PageBreak()]

    # Recommendation.
    story += [para("7. Рекомендация", s, "H1R2"), callout(
        "Не заменять production-кандидата на максимальный retrospective score. Заморозить shrinkage-ансамбль, "
        "где equal experts - база, soft regime - ограниченная поправка, short-window ExtraTrees - отдельный "
        "адаптивный эксперт. Следующий результат считать только на данных после 04.09.2026.", s, PALE_GREEN, GREEN)]
    story += [para("Почему именно так", s, "H2R2")]
    for item in (
        "Equal mixture лучше всего прошла заранее выделенный shock-блок при корректной частоте.",
        "Soft regime router единственный новый finalist с min lift выше 1.16 и по годам, и по валютам на последнем блоке.",
        "Global trailing weights имеют лучший средний retrospective lift, но заметный провал года.",
        "Short-window mix адаптивна к break, но не проходит нижнюю границу частоты на validation.",
        "Сложный hard gate и post-2022 flag ухудшают переносимость.",
    ):
        story += [para("- " + item, s)]
    story += [para("Что заморозить", s, "H2R2"), tbl([
        ["Компонент", "Замороженное решение"],
        ["Information set", "только текущие и прошлые опубликованные курсы; следующий курс исключен"],
        ["Experts", "ExtraTrees, KNN path, local spline-logit, GMM-hist, local floor, survival-logit"],
        ["Gate", "soft weights с сильным shrinkage к equal; никаких дат будущего режима"],
        ["Порог", "по предыдущему году отдельно для каждой валюты"],
        ["Контроль", "1-2 alerts/corridor/week; future-only benefit >0; audit по году и валюте"],
        ["Подтверждение", "никакой перенастройки до накопления нового holdout"],
    ], s, [45*mm, 129*mm])]
    story += [para("Отдельный after-release сценарий", s, "H2R2"), para(
        "Если продукт проверяет факт публикации следующего эффективного курса ЦБ, отдельная политика даёт "
        "около 1.96. Это не обычный pre-publication forecast: первый шаг уже известен. ЦБ сообщает, что "
        "курс обычно публикуется до 18:00 Москвы, но точная минута не гарантирована. "
        '<link href="https://www.cbr.ru/Reception/TopicalMessage/Page/2661">Официальное разъяснение</link>.', s), PageBreak()]

    # Repro and references.
    story += [para("8. Воспроизводимость и источники", s, "H1R2"), para(
        "Все прогнозы, калибровочные индексы, тестовые индексы и итоговые таблицы сохранены локально. "
        "Канонический текст отчёта: results/research/round2/report-source.md. Реестр утверждений: "
        "claim-source-ledger.md. Checkpoint до второй волны: results/research/checkpoint_2026-09-04.md.", s)]
    story += [tbl([
        ["Модуль", "Назначение"],
        ["round2_eda", "общий фактор, сезонность, break scan, observed regimes"],
        ["round2_towers", "local/global OOF residual towers"],
        ["round2_diverse_models", "19 независимых классических архитектур"],
        ["round2_external_data/models", "release-aware external sensitivity"],
        ["round2_router", "OOF error profiles и hard/soft routing"],
        ["round2_recency", "2/3/5-year windows, decay, post-2022 weighting"],
        ["round2_rankers", "pairwise/NDCG direct ranking"],
        ["round2_statistical_audit", "block CI, breakdowns, circular max test"],
    ], s, [57*mm, 117*mm])]
    story += [para("Ключевые внешние источники", s, "H2R2")]
    refs = [
        ("Rossi, Exchange Rate Predictability", "https://doi.org/10.1257/jel.51.4.1063"),
        ("Montero-Manso & Hyndman, Global Forecasting Models", "https://doi.org/10.1016/j.ijforecast.2021.03.028"),
        ("Giacomini & White, Conditional Predictive Ability", "https://doi.org/10.1111/j.1468-0262.2006.00718.x"),
        ("White, Reality Check for Data Snooping", "https://doi.org/10.1111/1468-0262.00152"),
        ("Hansen, Superior Predictive Ability", "https://doi.org/10.1198/073500105000000063"),
        ("Hansen, Lunde & Nason, Model Confidence Set", "https://doi.org/10.3982/ECTA5771"),
        ("Pesaran & Pick, Forecasting Under Drift Instability", "https://www.ifo.de/en/cesifo/publications/2008/working-paper/forecasting-random-walks-under-drift-instability"),
        ("Bank of Russia official rates and publication timing", "https://www.cbr.ru/Reception/TopicalMessage/Page/2661"),
    ]
    for title, url in refs:
        story += [para(f'- <link href="{url}">{title}</link>', s, "SmallR2")]
    story += [Spacer(1, 5*mm), callout(
        "Итог: новая сильная гипотеза найдена - мягкая смесь экспертов с причинным regime routing. "
        "Её преимущество должно быть подтверждено следующим, действительно новым временным отрезком.",
        s, PALE_BLUE, BLUE)]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(PDF)
    return PDF


if __name__ == "__main__":
    build()
