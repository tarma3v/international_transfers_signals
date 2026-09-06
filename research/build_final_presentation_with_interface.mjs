import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import {
  FileBlob,
  PresentationFile,
} from "/Users/jeck5iv/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const ROOT = "/Users/jeck5iv/Documents/ChatGPT/itmo/international_transfers_signals";
const SOURCE = path.join(ROOT, "submission/prezentaciya-finalnaya.pptx");
const FINAL = path.join(ROOT, "output/presentation/international_transfers_final_with_interface_2026-09-06_v2.pptx");
const SKILL_DIR = "/Users/jeck5iv/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations";
const RUNTIME_PYTHON = "/Users/jeck5iv/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const SOURCE_SHA256 = "3b939c09b0344b23c4ae497dc7ea9d2a3506a1ce00bcbb60594b25ae75c5b815";

process.env.RUNTIME_NODE = "/Users/jeck5iv/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node";
process.env.RUNTIME_NODE_MODULES = "/Users/jeck5iv/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
process.env.RUNTIME_BIN_DIR = "/Users/jeck5iv/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override";

const C = {
  navy: "#142331",
  navy2: "#1D3142",
  orange: "#B5650B",
  pale: "#EFF2F4",
  pale2: "#F6F7F8",
  white: "#FFFFFF",
  gray: "#586777",
  green: "#136B55",
  greenPale: "#E8F4EF",
  red: "#A33C33",
  redPale: "#F8EDEA",
  amberPale: "#FCF3E8",
};

const presentation = await PresentationFile.importPptx(await FileBlob.load(SOURCE));

function clearSlide(slide) {
  for (const element of [...slide.elements.items]) slide.elements.deleteById(element.id);
}

function addText(slide, name, value, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    fontFamily: "Calibri",
    fontSize: 20,
    color: C.navy,
    ...style,
  };
  return shape;
}

function addRect(slide, name, position, fill, line = "none", radius = 0) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    name,
    position,
    fill,
    line: { style: "solid", fill: line, width: line === "none" ? 0 : 1 },
    borderRadius: radius || undefined,
  });
}

function addFooter(slide, number, dark = false) {
  addText(
    slide,
    `footer-${number}`,
    `international_transfers_signals  ·  06.09.2026  ·  ${number}`,
    { left: 68, top: 671, width: 1142, height: 24 },
    { fontSize: 11, color: dark ? "#BED0DE" : C.gray },
  );
}

function startStandard(slide, number, title, subtitle = "") {
  clearSlide(slide);
  slide.background.fill = C.white;
  slide.shapes.add({
    geometry: "ellipse",
    name: `number-badge-${number}`,
    position: { left: 67, top: 50, width: 46, height: 41 },
    fill: C.orange,
    line: { style: "solid", fill: C.orange, width: 0 },
  });
  addText(slide, `number-label-${number}`, String(number), { left: 67, top: 56, width: 46, height: 24 }, { fontSize: number >= 10 ? 13 : 18, bold: true, color: C.white, alignment: "center" });
  addText(slide, `title-${number}`, title, { left: 123, top: 42, width: 1087, height: 59 }, { fontFamily: "Cambria", fontSize: 30, bold: true });
  if (subtitle) addText(slide, `subtitle-${number}`, subtitle, { left: 68, top: 112, width: 1142, height: 40 }, { fontSize: 18, color: C.gray });
  addFooter(slide, number, false);
}

function addMetricCard(slide, name, left, top, width, value, label, color = C.orange) {
  const box = addRect(slide, `${name}-box`, { left, top, width, height: 142 }, C.pale2, "#DDE3E7", 12);
  addText(slide, `${name}-value`, value, { left: left + 18, top: top + 17, width: width - 36, height: 50 }, { fontSize: 34, bold: true, color });
  addText(slide, `${name}-label`, label, { left: left + 18, top: top + 74, width: width - 36, height: 54 }, { fontSize: 16, color: C.gray });
  box.sendToBack();
}

function addTable(slide, name, left, top, width, height, values, widths, fontSize = 15) {
  const table = slide.tables.add({ rows: values.length, columns: values[0].length, left, top, width, height, values, columnWidths: widths });
  table.name = name;
  table.borders.assign({ style: "solid", fill: C.white, width: 1 });
  for (let r = 0; r < values.length; r += 1) {
    for (let c = 0; c < values[0].length; c += 1) {
      const cell = table.getCell(r, c);
      cell.fill = r === 0 ? C.navy : (r % 2 === 1 ? C.pale : C.white);
      cell.text.style = { fontFamily: "Calibri", fontSize: r === 0 ? fontSize - 1 : fontSize, bold: r === 0, color: r === 0 ? C.white : C.navy };
    }
  }
  return table;
}

function note(slide, value) {
  slide.speakerNotes.textFrame.setText(value);
}

// Slide 1: refreshed cover. Product and interface slides from the source deck remain native.
{
  const s = presentation.slides.getItem(0);
  clearSlide(s);
  s.background.fill = C.navy;
  addText(s, "cover-date", "ФИНАЛЬНОЕ РЕШЕНИЕ · 06.09.2026", { left: 68, top: 143, width: 1142, height: 34 }, { fontSize: 17, bold: true, color: C.orange });
  addText(s, "cover-title", "Как поймать выгодный момент\nдля перевода за рубеж", { left: 68, top: 203, width: 1142, height: 170 }, { fontFamily: "Cambria", fontSize: 50, bold: true, color: C.white });
  addText(s, "cover-subtitle", "Температура в любой момент · ожидаемая выгода · редкий push\nAMD · KGS · KZT · TJS · UZS", { left: 68, top: 412, width: 1100, height: 95 }, { fontSize: 22, color: "#D8E4EC" });
  addText(s, "cover-repo", "github.com/tarma3v/international_transfers_signals", { left: 68, top: 625, width: 800, height: 30 }, { fontSize: 14, color: "#BED0DE" });
  note(s, "Главный результат на сегодня — единый причинный any-time контур. Он выдаёт температуру, отдельную оценку размера future-only выгоды и отдельный редкий push.");
}

// Slide 2: value proposition with current outcomes.
{
  const s = presentation.slides.getItem(1);
  startStandard(s, 1, "Ценность: клиент перестаёт выбирать момент вслепую", "Один экран отвечает всегда; уведомление приходит только в сильный момент");
  addMetricCard(s, "value-temp", 68, 166, 350, "0–100", "непрерывная температура по выбранному h", C.green);
  addMetricCard(s, "value-lift", 465, 166, 350, "2,43–2,53", "adjusted lift редкого AP37 push", C.orange);
  addMetricCard(s, "value-rate", 862, 166, 348, "≤ 2 / нед.", "лимит уведомлений на валюту", C.navy);
  addRect(s, "value-body", { left: 68, top: 343, width: 1142, height: 251 }, C.pale2, "#DDE3E7", 12);
  addText(s, "value-body-title", "Три независимых ответа вместо одного магического прогноза", { left: 94, top: 368, width: 1060, height: 38 }, { fontFamily: "Cambria", fontSize: 24, bold: true });
  addText(s, "value-body-text", "1 · Температура: насколько похожие ситуации были удачными.\n2 · Ожидаемые б.п.: насколько велико возможное преимущество текущего курса ЦБ.\n3 · Push: достаточно ли момент сильный и редкий, чтобы отвлечь клиента.\n\nСумма к получению должна рассчитываться по исполнимой котировке банка, а не по модельному курсу ЦБ.", { left: 94, top: 423, width: 1060, height: 151 }, { fontSize: 18 });
  note(s, "Температура, expected bps и push — разные модельные выходы и разные метрики. Это сохраняет понятность продукта и не выдаёт вероятность за обещание банковского курса.");
}

// Slide 3 stays unchanged: the complete client journey.
note(presentation.slides.getItem(2), "Сохраняем исходный сильный продуктовый слайд: он показывает путь клиента от открытия приложения до перевода, истории и обратной связи.");

// Slide 4: outputs and metrics.
{
  const s = presentation.slides.getItem(3);
  startStandard(s, 3, "Один продукт — три модельных выхода", "Каждый выход имеет свою честную метрику");
  addTable(s, "output-metrics", 68, 165, 1142, 330, [
    ["Выход", "Что отвечает", "Как проверяем"],
    ["Температура 0–100", "P(действующий ЦБ не хуже min следующих h публикаций)", "Brier ↓ · AUC ↑ · калибровка"],
    ["Ожидаемая выгода", "future-only разница относительно действующего ЦБ", "MAE базисных пунктов ↓"],
    ["Редкий push", "выбор сильных моментов при max2/неделю", "adjusted lift ↑ · rate · б.п."],
    ["Техническая метрика ТЗ", "симметричное окно ±h", "считаем отдельно, не смешиваем с продуктом"],
  ], [255, 585, 302], 15);
  addRect(s, "metric-warning", { left: 68, top: 522, width: 1142, height: 105 }, C.amberPale, "#E8C69F", 10);
  addText(s, "metric-warning-text", "Температура 73 означает: примерно 73 из 100 похожих прошлых ситуаций были удачными. Это не 73% прибыли и не гарантия цены сделки.", { left: 94, top: 545, width: 1090, height: 62 }, { fontSize: 19, bold: true });
  note(s, "Не сравниваем lift и AUC как одну шкалу. Lift оценивает политику редких решений, Brier/AUC — вероятность, MAE — размер будущей разницы.");
}

// Slide 5: phase router.
{
  const s = presentation.slides.getItem(4);
  startStandard(s, 4, "Как прогноз работает в течение дня", "Роутер использует только источник, уже завершившийся к as_of");
  addTable(s, "phase-router", 68, 147, 1142, 476, [
    ["Москва", "Что известно", "Что делает система"],
    ["до 09:00", "история ЦБ и календарь", "history-only · limited confidence"],
    ["09:00–10:30", "CNYRUBF · первая CNY spot-свеча", "сильный ранний market update"],
    ["10:30–15:30", "накопленные завершённые spot-prefix", "температура уточняется по сессии"],
    ["15:30–receipt", "доступные 16:30/17:30 свечи", "bridge без курса ЦБ на завтра"],
    ["после receipt", "завтрашний ЦБ уже получен", "AP50 probability · AP51 benefit"],
    ["19:00–20:00", "новая spot-свеча", "T7B обновляет температуру"],
    ["20:00–23:00", "CNYRUBF + USDRUBF", "T16 уточняет benefit h3/h5"],
    ["ночь / выходной", "нового источника нет", "hold · age · stale"],
  ], [186, 466, 490], 14);
  note(s, "Receipt должен быть реальным событием, а не жёстким временем. Пока историческая реализация использует календарное приближение; это ключевая production-доработка.");
}

// Slide 6 stays unchanged: pilot design.
note(presentation.slides.getItem(5), "Сохраняем исходный слайд пилота: различаем техническую метрику и бизнес-эффект и заранее фиксируем правила эксперимента.");

// Slide 7 stays unchanged: appendix divider.
note(presentation.slides.getItem(6), "Переход к техническому приложению.");

// Slide 8: leakage protection.
{
  const s = presentation.slides.getItem(7);
  startStandard(s, 5, "Подглядывания в будущее нет — это проверяется", "Прогноз воспроизводится только из информационного мира момента as_of");
  addTable(s, "causal-controls", 68, 159, 1142, 362, [
    ["Защита", "Что запрещает", "Проверка"],
    ["mature-only train", "обучаться на ещё не созревшем будущем", "дата метки должна быть известна"],
    ["embargo 2 дня", "подтягивать соседнее будущее через границу", "жёсткий временной зазор"],
    ["source_at ≤ as_of", "читать незавершённую или будущую свечу", "provenance в каждом ответе"],
    ["future-prefix corruption", "скрыто зависеть от будущих строк", "прошлые ответы обязаны совпасть"],
    ["frozen screen rules", "переобучать выбор на test", "screen-2024 отдельно от opened-2025–2026"],
  ], [285, 502, 355], 14);
  addRect(s, "causal-note", { left: 68, top: 545, width: 1142, height: 84 }, C.greenPale, "#B7DCCD", 10);
  addText(s, "causal-note-text", "Более свежий источник не подключается автоматически: новая голова обязана заранее пройти ворота качества. Иначе роутер оставляет сильный контроль.", { left: 94, top: 565, width: 1090, height: 50 }, { fontSize: 18, bold: true, color: C.green });
  note(s, "2025–2026 многократно открывался в исследовании, поэтому это диагностика переноса, не новый независимый holdout. Следующий уровень — prospective frozen shadow.");
}

// Slide 9: probability results.
{
  const s = presentation.slides.getItem(8);
  startStandard(s, 6, "Температура: рынок сильнее всего помогает к 15:30", "Открытая диагностика 2025–2026; AUC выше, Brier ниже — лучше");
  addTable(s, "probability-results", 68, 154, 1142, 384, [
    ["Фаза", "AUC", "Brier", "Комментарий"],
    ["premarket h1/h3/h5", "0,594 / 0,604 / 0,596", "—", "осторожный history-only fallback"],
    ["09:00 perpetual h1", "0,753", "0,2024", "контроль: 0,587 / 0,2450"],
    ["09:00 perpetual h3", "0,734", "0,1898", "контроль: 0,586 / 0,2173"],
    ["10:30 · mean", "0,757", "0,1562", "рынок уже хорошо различает дни"],
    ["15:30 · mean", "0,785", "0,1477", "лучший pre-publication срез"],
    ["19:00 T7B · mean", "0,716", "0,1492", "небольшой плюс вечернего spot"],
    ["after-publication h3/h5/h10/h20", "—", "0,165 / 0,163 / 0,164 / 0,133", "новый ЦБ уже известен"],
  ], [298, 224, 265, 355], 13);
  addText(s, "probability-foot", "После 20:00 новые probability-кандидаты не прошли screen-ворота, поэтому температура честно остаётся на T7B.", { left: 68, top: 565, width: 1142, height: 64 }, { fontSize: 18, bold: true, color: C.red });
  note(s, "Это намеренный отказ от лишнего обновления. Физически свежая свеча не означает лучшую калибровку вероятности.");
}

// Slide 10: benefit and push.
{
  const s = presentation.slides.getItem(9);
  startStandard(s, 7, "Ожидаемая выгода и push решают разные задачи", "Вечер уточняет размер эффекта; AP37 выбирает редкие моменты");
  addTable(s, "benefit-push", 68, 151, 1142, 354, [
    ["Контур", "Результат", "Контроль", "Вывод"],
    ["15:30 benefit h1/h3/h5/h10", "54,50 / 81,13 / 98,13 / 126,06 MAE", "61,71 / 87,70 / 102,65 / 131,77", "выигрыш 5–15 б.п."],
    ["23:00 T15 benefit h3/h5", "53,14 / 80,85 MAE", "58,08 / 87,88", "dual Ridge принят"],
    ["AP37 push h3/h5/h10/h20", "lift 2,429 / 2,509 / 2,479 / 2,525", "случайная частота", "max2/неделю"],
    ["AP37 h5", "695 решений · 133,73 future-only б.п.", "—", "≈ 1 сигнал/неделю"],
  ], [284, 360, 273, 225], 13);
  addMetricCard(s, "push-lift", 68, 530, 350, "2,525", "лучший adjusted lift: h20", C.orange);
  addMetricCard(s, "benefit-mae", 465, 530, 350, "80,85", "MAE T15 в 23:00 · h5", C.green);
  addMetricCard(s, "snapshot-count", 862, 530, 348, "60 370", "итоговых any-time снимков", C.navy);
  note(s, "MAE измеряется в базисных пунктах официального курса ЦБ. Lift оценивает редкую политику решений и не доказывает точность температуры.");
}

// Slide 11: what T15/T16 added.
{
  const s = presentation.slides.getItem(10);
  startStandard(s, 8, "T15 → T16: вечером обновляем только доказанную часть", "Probability и magnitude могут иметь разные источники и время");
  addTable(s, "evening-router", 68, 155, 1142, 344, [
    ["Проверка", "Результат", "Решение"],
    ["CNYRUBF + USDRUBF", "100% coverage в 2024, 2025, 2026", "можно честно проверять 20/21/22/23"],
    ["Probability после 20:00", "logit, dual logit и HGB не прошли оба CI", "оставить T7B"],
    ["Benefit после 20:00", "dual Ridge улучшил 7 комбинаций h3/h5", "обновлять expected bps"],
    ["h1 и длинные горизонты", "h1 уже известен; h10/h20 без выигрыша", "точный h1, контроль h10/h20"],
    ["Push", "все AP37 решения совпали", "уведомления не менять"],
  ], [324, 502, 316], 14);
  addRect(s, "provenance-box", { left: 68, top: 527, width: 1142, height: 101 }, C.amberPale, "#E8C69F", 10);
  addText(s, "provenance-text", "Пример 21:15: температура может быть от spot 20:00, а expected bps — от perpetual 20:59:59. Интерфейс показывает оба timestamp, возраст и freshness.", { left: 94, top: 550, width: 1090, height: 61 }, { fontSize: 19, bold: true });
  note(s, "T16 маршрутизирует головы раздельно. Это не рассинхронизация, а честное признание: свежий perpetual доказал пользу для размера эффекта, но не для вероятности.");
}

// Slide 12 stays unchanged: seven native interface screens from the source deck.
note(presentation.slides.getItem(11), "Сохраняем семь экранов существующего пользовательского flow. В текущем payload к ним добавляются phase, horizon, source_at, age, freshness и отдельный provenance expected bps.");

// Slide 13: refreshed push copy without the layout collision in the source slide.
{
  const s = presentation.slides.getItem(12);
  startStandard(s, 9, "Push: коротко, честно и только в сильный момент", "Текст объясняет факт; AP37 отдельно решает, стоит ли отвлекать клиента");
  addTable(s, "push-copy", 68, 154, 1142, 330, [
    ["Ситуация", "Пример текста", "Почему честно"],
    ["Сильный момент h5", "Сейчас курс выглядит сильнее обычного. Если перевод планировался — проверьте сумму.", "нет обещания будущей цены"],
    ["Клиент открыл позже", "Оценка обновлена в 20:00. Сейчас температура 73 из 100, источник свежий.", "видны время и freshness"],
    ["Данные устарели", "Рынок закрыт. Показываем последнюю оценку, ей 9 часов.", "hold не маскируется под live"],
    ["Фиксированная дата", "Если перевод обязателен сегодня, сравните сумму и комиссию банка.", "модель не советует нарушать срок"],
  ], [268, 560, 314], 14);
  addRect(s, "push-rule", { left: 68, top: 516, width: 1142, height: 112 }, C.navy, C.navy, 10);
  addText(s, "push-rule-title", "Правило отправки", { left: 94, top: 536, width: 260, height: 33 }, { fontSize: 22, bold: true, color: C.orange });
  addText(s, "push-rule-body", "Высокая температура не равна push. Нужны сильный AP37 rank, cooldown и лимит max2 на валюту в ISO-неделю.", { left: 94, top: 574, width: 1084, height: 40 }, { fontSize: 18, color: C.white });
  note(s, "Тексты — продуктовый слой над AP37. Они сообщают наблюдаемый факт и предлагают проверить исполнимую сумму, но не обещают курс и не заставляют откладывать обязательный платёж.");
}

// Slide 14: refreshed team contributions without stale branch metrics.
{
  const s = presentation.slides.getItem(13);
  startStandard(s, 10, "Команда и текущий вклад", "Исследование, причинность и продукт собираются совместно");
  const people = [
    ["Александр Тармаев", "Product Engineer", "продуктовая постановка · пользовательский flow · тексты · дизайн пилота"],
    ["Даниил Недайборщ", "AI Engineer · разработка", "данные и признаки · загрузчики · инфраструктура причинности · воспроизводимость"],
    ["Иван Калинин", "AI Engineer · ML", "фазовый роутер · OOS-модели · T15/T16 · аудиты · сводка результатов"],
  ];
  people.forEach((p, idx) => {
    const top = 153 + idx * 151;
    addRect(s, `person-${idx}`, { left: 68, top, width: 1142, height: 130 }, idx === 2 ? C.amberPale : C.pale2, idx === 2 ? "#E8C69F" : "#DDE3E7", 10);
    addText(s, `person-name-${idx}`, p[0], { left: 94, top: top + 18, width: 310, height: 34 }, { fontFamily: "Cambria", fontSize: 22, bold: true });
    addText(s, `person-role-${idx}`, p[1], { left: 94, top: top + 56, width: 310, height: 28 }, { fontSize: 16, color: C.gray });
    addText(s, `person-work-${idx}`, p[2], { left: 430, top: top + 30, width: 730, height: 68 }, { fontSize: 18 });
  });
  addText(s, "team-foot", "Репозиторий, frozen-регламенты и артефакты позволяют воспроизвести каждое принятое решение.", { left: 68, top: 620, width: 1142, height: 32 }, { fontSize: 17, color: C.gray });
  note(s, "Проценты вклада не заявляем: работа совместная. Слайд отражает зоны ответственности и убирает устаревшее сравнение ранних веток.");
}

// Slide 15: refreshed close.
{
  const s = presentation.slides.getItem(14);
  clearSlide(s);
  s.background.fill = C.navy;
  addText(s, "final-eyebrow", "ЧТО МЫ РЕШИЛИ", { left: 68, top: 78, width: 1142, height: 32 }, { fontSize: 17, bold: true, color: C.orange });
  addText(s, "final-title", "Момент ловится.\nТеперь — в любое время дня.", { left: 68, top: 136, width: 1142, height: 140 }, { fontFamily: "Cambria", fontSize: 46, bold: true, color: C.white });
  addText(s, "final-body", "Температура объясняет обстановку. Expected bps оценивает размер эффекта.\nAP37 отправляет редкий push. Роутер меняет модель только после появления источника и доказанного улучшения.", { left: 68, top: 320, width: 1090, height: 130 }, { fontSize: 23, color: "#D8E4EC" });
  addRect(s, "final-next", { left: 68, top: 502, width: 1142, height: 102 }, C.navy2, "#40596E", 12);
  addText(s, "final-next-text", "До production: реальный receipt ЦБ · котировки банка и комиссия · frozen shadow · экономический uplift", { left: 96, top: 530, width: 1086, height: 48 }, { fontSize: 21, bold: true, color: C.white });
  addText(s, "final-repo", "github.com/tarma3v/international_transfers_signals", { left: 68, top: 646, width: 900, height: 28 }, { fontSize: 13, color: "#BED0DE" });
  note(s, "Финальный вывод: причинный any-time прототип уже закрывает пользовательский экран и редкий push; бизнес-экономика требует банковской исполнимой цены и prospective проверки.");
}

const requirements = {
  explicitTotalSlideCount: 15,
  requiredNativeTableOwnerSlides: [4, 5, 8, 9, 10, 11],
  requiredNativeChartOwnerSlides: [],
};
const fontPolicy = {
  basis: "reference",
  families: ["Calibri", "Cambria"],
  referencePath: SOURCE,
  referenceSha256: SOURCE_SHA256,
};

const { finalizePresentation } = await import(pathToFileURL(
  path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs"),
).href);
const stagingDir = path.join(ROOT, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
await fs.mkdir(path.dirname(FINAL), { recursive: true });
const candidatePath = path.join(stagingDir, "final-with-interface-candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const result = await finalizePresentation({
  ...requirements,
  workspaceDir: ROOT,
  candidatePath,
  finalPath: FINAL,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    ...requirements.requiredNativeTableOwnerSlides.flatMap((n) => ["--require-native-table-slide", String(n)]),
  ],
  requiredNativeTableOwnerSlides: requirements.requiredNativeTableOwnerSlides,
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "international_transfers_final_with_interface_2026-09-06_v2.pptx.validation.json"),
});

console.log(JSON.stringify({ final: FINAL, result }, null, 2));
