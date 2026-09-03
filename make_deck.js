const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";           // 13.3 x 7.5
p.author = "Команда проекта";
p.title  = "Триггерная модель для трансграничных переводов";

const INK="1A2530", PAPER="F4F6F8", W="FFFFFF",
      AMBER="9C5B12", GREEN="16674F", RED="9E3226", GREY="6E7A88", MUTED="4E5966";
const HEAD="Cambria", BODY="Calibri";
const FIG = "submission/figures/";

// ——— помощники ———
function titleSlide(s, kicker, title, sub) {
  s.background = { color: INK };
  s.addText(kicker, { x:0.7, y:2.05, w:11.9, h:0.3, isTextBox:true, margin:0,
    fontFace:BODY, fontSize:13, color:AMBER, charSpacing:2, bold:true });
  s.addText(title, { x:0.7, y:2.45, w:11.9, h:1.9, isTextBox:true, margin:0,
    fontFace:HEAD, fontSize:40, color:W, bold:true, lineSpacing:46 });
  if (sub) s.addText(sub, { x:0.7, y:4.45, w:11.0, h:0.9, isTextBox:true, margin:0,
    fontFace:BODY, fontSize:15, color:"C7CFD8", lineSpacing:24 });
}
function head(s, n, title) {
  s.background = { color: W };
  s.addShape(p.ShapeType.ellipse, { x:0.7, y:0.52, w:0.42, h:0.42, fill:{color:AMBER} });
  s.addText(String(n), { x:0.7, y:0.52, w:0.42, h:0.42, isTextBox:true, margin:0,
    fontFace:BODY, fontSize:14, color:W, bold:true, align:"center", valign:"middle" });
  s.addText(title, { x:1.28, y:0.44, w:11.3, h:0.62, isTextBox:true, margin:0,
    fontFace:HEAD, fontSize:27, color:INK, bold:true, valign:"middle" });
}
function note(s, t) {
  s.addText(t, { x:0.7, y:6.86, w:11.9, h:0.34, isTextBox:true, margin:0,
    fontFace:BODY, fontSize:10.5, color:GREY, italic:true });
}
function stat(s, x, y, w, big, label, col) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h:1.72, fill:{color:PAPER}, rectRadius:0.06 });
  s.addText(big, { x:x+0.02, y:y+0.16, w:w-0.04, h:0.82, isTextBox:true, margin:0,
    fontFace:HEAD, fontSize:37, color:col, bold:true, align:"center" });
  s.addText(label, { x:x+0.22, y:y+1.00, w:w-0.44, h:0.62, isTextBox:true, margin:0,
    fontFace:BODY, fontSize:12, color:MUTED, align:"center", lineSpacing:15 });
}
function card(s, x, y, w, h, ttl, body, col) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h, fill:{color:PAPER}, rectRadius:0.06 });
  s.addText(ttl, { x:x+0.28, y:y+0.2, w:w-0.56, h:0.46, isTextBox:true, margin:0,
    fontFace:HEAD, fontSize:15.5, color:col||INK, bold:true });
  s.addText(body, { x:x+0.28, y:y+0.68, w:w-0.56, h:h-0.9, isTextBox:true, margin:0,
    fontFace:BODY, fontSize:12.5, color:MUTED, lineSpacing:17 });
}

// ——— 1. Титул ———
let s = p.addSlide();
titleSlide(s, "ПРОМЕЖУТОЧНАЯ ЗАЩИТА · 04.09.2026",
  "Как поймать выгодный момент\nдля перевода за рубеж",
  "Триггерная модель для коридоров Россия → Таджикистан, Узбекистан, Киргизия, Армения, Казахстан");
s.addText("github.com/tarma3v/international_transfers_signals", { x:0.7, y:6.5, w:8, h:0.35,
  isTextBox:true, margin:0, fontFace:BODY, fontSize:12, color:AMBER });
s.addNotes("Промежуточная сдача. Главный результат — не найденный сигнал, а понимание того, что измеряет метрика кейса.");

// ——— 2. Проблематика ———
s = p.addSlide(); head(s, 1, "Клиент выбирает момент вслепую");
s.addText("Разброс курса внутри месяца — 600–660 базисных пунктов. На переводе 30 000 ₽ это около 1 800–2 000 ₽ разницы между лучшим и худшим днём. Но забрать её целиком нельзя: клиент привязан к дате зарплаты.",
  { x:0.7, y:1.28, w:11.9, h:0.8, isTextBox:true, margin:0, fontFace:BODY, fontSize:14.5, color:MUTED, lineSpacing:21 });
stat(s, 0.7,  2.30, 3.75, "36–40 %", "месячного размаха доступно\nклиенту, привязанному к зарплате", AMBER);
stat(s, 4.78, 2.30, 3.75, "22–30 б. п.", "честный потолок выгоды\nв реальном окне клиента", AMBER);
stat(s, 8.86, 2.30, 3.75, "0", "массовых сервисов переводов\nшлют push «сейчас выгодно»", AMBER);
card(s, 0.7, 4.35, 5.87, 2.28, "Что делает ближайший аналог",
  "Google Flights показывает метку на экране — «цена ниже обычного» — а не рассылает уведомления. Мы взяли этот принцип за основу: индикатор в момент, когда клиент и так пришёл переводить.", INK);
card(s, 6.73, 4.35, 5.87, 2.28, "Два сегмента ведут себя по-разному",
  "Пересылают зарплату — привязаны к дате, ждать не могут, видят 36–40 % возможности. Переводят на траты — могут ждать. Вся измеримая ценность продукта сосредоточена во втором сегменте.", INK);
s.addNotes("Три цифры измерены нами на данных ЦБ 2019–2026. Потолок 22–30 бп — это то, что клиент реально может забрать в своём окне.");

// ——— 3. Постановка ———
s = p.addSlide(); head(s, 2, "Постановка задачи и наше уточнение");
s.addText("Фокус — триггерные коммуникации, а не ценообразование. Фиксация курса и скидки в спреде исключены из MVP.",
  { x:0.7, y:1.25, w:11.9, h:0.4, isTextBox:true, margin:0, fontFace:BODY, fontSize:13.5, color:MUTED });
s.addTable([
  [{text:"Метрика заказчика", options:{bold:true, color:W, fill:{color:INK}}},
   {text:"Определение", options:{bold:true, color:W, fill:{color:INK}}},
   {text:"Наша позиция", options:{bold:true, color:W, fill:{color:INK}}}],
  ["«Сейчас выгодно»", "курс не станет ниже за h публикаций", "считаем как задано; показали, чем опасна"],
  ["«Окно закрывается»", "через h публикаций курс выше", "считаем как задано"],
  [{text:"«Выгода момента»", options:{bold:true}}, "сравнение с окном ±h",
   {text:"раскладываем на достижимую и недостижимую половины", options:{bold:true, color:AMBER}}],
], { x:0.7, y:1.82, w:11.9, colW:[2.7,3.9,5.3], fontFace:BODY, fontSize:12.5,
     color:INK, border:{type:"solid", color:"DDE3E9", pt:1}, rowH:0.46, valign:"middle" });
card(s, 0.7, 4.05, 11.9, 2.55, "Уточнение, которое меняет постановку",
  "Метрика «выгода момента» симметрична: она сравнивает день с окном ±h, половина которого — прошлое.\n\nПравило, срабатывающее ПОСЛЕ падения курса, набирает по ней высокий балл, ничего не предсказав. Клиент может забрать только форвардную половину — прошлое ему недоступно.\n\nМы считаем обе половины и в качестве продуктовой используем достижимую.", AMBER);
s.addNotes("Это уточнение — основа всего дальнейшего. Горизонты h = 1,3,5,10,20, все пять.");

// ——— 4. Главная находка ———
s = p.addSlide();
titleSlide(s, "ГЛАВНЫЙ ПРОМЕЖУТОЧНЫЙ РЕЗУЛЬТАТ",
  "Метрику кейса максимизирует\nодна строка кода, а не модель",
  "Мы обучили модель под метрику заказчика и сравнили с однострочным правилом при равной частоте срабатываний.\nОни неразличимы: lift 1,30 против 1,30. Лучший результат по метрике даёт фиксированный порог — 1,42.\nВыгода для семьи при этом +18 против +17 базисных пунктов при потолке +143 и интервалах, пересекающих ноль.");
s.addNotes("Метрика кейса монотонна по одному признаку. ML к ней не добавляет ничего — это факт о метрике, а не о моделях.");

// ——— 5. Конфликт метрик ———
s = p.addSlide(); head(s, 3, "Две метрики указывают в разные стороны");
s.addImage({ path: FIG+"02-konflikt-metrik.png", x:0.7, y:1.28, w:11.9, h:4.475 });
s.addText([
  {text:"Слева — что чувствует клиент. ", options:{bold:true}},
  {text:"Справа — что показывает метрика кейса. Попадание растёт монотонно от дешёвых дней к дорогим: 24 % → 38 %. Ровно против выгоды."},
], { x:0.7, y:5.95, w:11.9, h:0.78, isTextBox:true, margin:0, fontFace:BODY, fontSize:13, color:MUTED, lineSpacing:19 });
note(s, "Данные ЦБ 2019–2026, пять коридоров, h = 5. Доверительные интервалы краёв не пересекают ноль.");

// ——— 6. Разложение выгоды ———
s = p.addSlide(); head(s, 4, "Заявленная выгода индикаторов ТЗ — это прошлое");
s.addImage({ path: FIG+"03-razlozhenie-vygody.png", x:0.7, y:1.20, w:7.5, h:3.79 });
card(s, 8.35, 1.20, 4.25, 3.79, "Что произошло",
  "Индикатор «моментум» показывает +92 б. п. по метрике заказчика.\n\n91 % этого — падение, которое уже случилось. Клиент его забрать не может.\n\nОстаётся −21 б. п., и доверительный интервал [−36; −7] целиком ниже нуля.", RED);
stat(s, 0.7, 5.22, 3.856, "−21 б. п.", "достижимая выгода\nиндикатора «моментум»", RED);
stat(s, 4.737, 5.22, 3.856, "−3 б. п.", "достижимая выгода\nиндикатора «уровень»", RED);
stat(s, 8.774, 5.22, 3.856, "+28 б. п.", "достижимая выгода\nмодели на h = 5", GREEN);
s.addNotes("Ни одно правило ТЗ не даёт одновременно lift выше 1 и положительную достижимую выгоду. Модель даёт.");

// ——— 7. Схема решения ———
s = p.addSlide(); head(s, 5, "Схема решения");
const steps = [
  ["Данные", "Курсы ЦБ 2019–2026\nПострочная нормировка\nноминала\nРяд публикаций"],
  ["Признаки", "79 признаков\nтолько по прошлому\nМоментум, уровень,\nразворот, календарь"],
  ["Ворота", "Доказательство\nотсутствия утечки\nЭксперимент не стартует\nбез прохождения"],
  ["Модели", "Walk-forward с очисткой\nЛогрегрессия, лес,\nGBM, CatBoost, XGBoost"],
  ["Продукт", "Индикатор на экране\nОповещение об уровне\nПравило отбора"],
];
steps.forEach((st, i) => {
  const x = 0.7 + i*2.42;
  const isLast = i === steps.length-1;
  s.addShape(p.ShapeType.roundRect, { x, y:1.55, w:2.16, h:2.75,
    fill:{color: isLast ? INK : PAPER}, rectRadius:0.06 });
  s.addText(st[0], { x:x+0.16, y:1.72, w:1.84, h:0.4, isTextBox:true, margin:0,
    fontFace:HEAD, fontSize:15, color: isLast ? W : AMBER, bold:true, align:"center" });
  s.addText(st[1], { x:x+0.16, y:2.16, w:1.84, h:2.0, isTextBox:true, margin:0,
    fontFace:BODY, fontSize:10.5, color: isLast ? "C7CFD8" : MUTED, align:"center", lineSpacing:14 });
  if (!isLast) s.addText("→", { x:x+2.16, y:2.72, w:0.26, h:0.4, isTextBox:true, margin:0,
    fontFace:BODY, fontSize:16, color:GREY, align:"center" });
});
card(s, 0.7, 4.62, 5.87, 2.0, "Что уже работает",
  "Загрузчик, 79 признаков, доказательство честности, 16 автотестов, walk-forward с очисткой, пять моделей, отбор признаков, испытания результата.", GREEN);
card(s, 6.73, 4.62, 5.87, 2.0, "Что осталось до финала",
  "Публичный signals_as_of(T), signals.csv, правило комбинирования и добор до полосы частоты, тексты уведомлений, прототип экрана, дизайн пилота.", AMBER);

// ——— 8. Честность ———
s = p.addSlide(); head(s, 6, "Заглядывания в будущее нет — и это доказано");
s.addText("Требование дисквалифицирующее, поэтому гарантия не опирается на аккуратность разработчика. Она обеспечена конструкцией.",
  { x:0.7, y:1.24, w:11.9, h:0.42, isTextBox:true, margin:0, fontFace:BODY, fontSize:13.5, color:MUTED });
card(s, 0.7, 1.82, 3.85, 2.35, "Единственный срез",
  "Ряд режется по времени в одном месте — past_slice. Функции признаков получают срез, а не весь ряд: будущего у них физически нет.", INK);
card(s, 4.73, 1.82, 3.85, 2.35, "Порча будущего",
  "Все значения после даты среза умножаются на 3. Признаки в прошлом обязаны совпасть побитово. Ноль расхождений из 79 на двух срезах.", INK);
card(s, 8.75, 1.82, 3.85, 2.35, "Тест на сам тест",
  "В срез подставляется реальная ошибка — сдвиг на 5 дней вперёд. Проверка обязана её поймать.", AMBER);
s.addShape(p.ShapeType.roundRect, { x:0.7, y:4.42, w:11.9, h:2.2, fill:{color:INK}, rectRadius:0.06 });
s.addText("Эпизод, который стоит рассказать", { x:1.0, y:4.62, w:11.3, h:0.4, isTextBox:true, margin:0,
  fontFace:HEAD, fontSize:16, color:AMBER, bold:true });
s.addText("Первая версия проверки не работала: подставная утечка читала массив, захваченный до порчи, и проверка «проходила» всегда. Мы это обнаружили и исправили.\n\nПроверка, не умеющая находить проблему, создаёт ложную уверенность — она хуже отсутствия проверки.",
  { x:1.0, y:5.06, w:11.3, h:1.4, isTextBox:true, margin:0, fontFace:BODY, fontSize:13, color:"C7CFD8", lineSpacing:19 });

// ——— 9. Результаты ———
s = p.addSlide(); head(s, 7, "Первые результаты");
s.addText("Цель «сейчас выгодно», h = 5, out-of-sample 2021–2026, базовая ставка 29,5 %. Порог модели зафиксирован на обучении — частоты у строк разные.",
  { x:0.7, y:1.24, w:11.9, h:0.4, isTextBox:true, margin:0, fontFace:BODY, fontSize:13, color:MUTED });
s.addTable([
  [{text:"Правило / модель", options:{bold:true,color:W,fill:{color:INK}}},
   {text:"lift", options:{bold:true,color:W,fill:{color:INK},align:"center"}},
   {text:"Выгода ±h", options:{bold:true,color:W,fill:{color:INK},align:"center"}},
   {text:"ДОСТИЖИМАЯ", options:{bold:true,color:W,fill:{color:INK},align:"center"}},
   {text:"95 % ДИ", options:{bold:true,color:W,fill:{color:INK},align:"center"}}],
  ["ТЗ: моментум (падение 3 дн)", {text:"0,81",align:"center"}, {text:"+92 б. п.",align:"center"},
   {text:"−21 б. п.",options:{align:"center",bold:true,color:RED}}, {text:"[−36; −7]",options:{align:"center"}}],
  ["ТЗ: уровень (нижний дециль)", {text:"0,83",align:"center"}, {text:"+58 б. п.",align:"center"},
   {text:"−3 б. п.",options:{align:"center",bold:true,color:RED}}, {text:"[−16; +12]",options:{align:"center"}}],
  ["ТЗ: разворот от минимума", {text:"0,90",align:"center"}, {text:"+1 б. п.",align:"center"},
   {text:"+13 б. п.",options:{align:"center"}}, {text:"[−7; +34]",options:{align:"center"}}],
  ["ТЗ: сезонность (до праздника)", {text:"1,01",align:"center"}, {text:"−10 б. п.",align:"center"},
   {text:"−13 б. п.",options:{align:"center",color:RED}}, {text:"[−27; +1]",options:{align:"center"}}],
  [{text:"контрпример: верхние 5 % диапазона",options:{italic:true,color:GREY}},
   {text:"1,42",options:{align:"center",italic:true,color:GREY}}, {text:"−60 б. п.",options:{align:"center",italic:true,color:GREY}},
   {text:"+63 б. п.",options:{align:"center",italic:true,color:GREY}}, {text:"[+47; +80]",options:{align:"center",italic:true,color:GREY}}],
  [{text:"Модель, выбранная ДО теста",options:{bold:true,fill:{color:"E8F0EC"}}},
   {text:"1,21",options:{align:"center",bold:true,fill:{color:"E8F0EC"}}},
   {text:"+28 б. п.",options:{align:"center",bold:true,fill:{color:"E8F0EC"}}},
   {text:"+28 б. п.",options:{align:"center",bold:true,color:GREEN,fill:{color:"E8F0EC"}}},
   {text:"[+19; +36]",options:{align:"center",bold:true,fill:{color:"E8F0EC"}}}],
], { x:0.7, y:1.78, w:11.9, colW:[4.3,1.3,2.0,2.3,2.0], fontFace:BODY, fontSize:12,
     color:INK, border:{type:"solid", color:"DDE3E9", pt:1}, rowH:0.42, valign:"middle" });
card(s, 0.7, 5.28, 11.9, 1.32, "Uplift к лучшему индикатору ТЗ — только на h = 5",
  "+0,20 по lift и +15 б. п. по достижимой выгоде. На горизонтах 1 / 3 / 10 / 20 прирост равен −17 / +3 / −4 / +20 б. п.: он колеблется вокруг нуля и меняет знак. Показываем все пять, а не лучший.", AMBER);
note(s, "Целевой порог lift ≥ 1,3 не достигнут. Он достижим правилом, которое отправляет клиента переводить на пике — мы показываем и то, и другое.");

// ——— 10. Две метрики, две модели ———
s = p.addSlide(); head(s, 8, "Две метрики, две модели");
s.addImage({ path: FIG+"05-dve-modeli.png", x:0.7, y:1.22, w:11.9, h:4.27 });
card(s, 0.7, 5.68, 5.87, 1.5, "Модель воспроизводит правило",
  "При равной частоте срабатывания: 1,30 у модели против 1,30 у правила pct ≥ 85, и +18 против +17 бп для семьи. Различий нет.", AMBER);
card(s, 6.73, 5.68, 5.87, 1.5, "Добавление признаков только вредит",
  "80 признаков дают 1,14, один с ограничением монотонности — 1,30. Метрика монотонна по уровню, и учить тут нечего.", GREEN);
s.addNotes("Ни одна политика не даёт значимого плюса по клиентской метрике: интервалы пересекают ноль, захвачено 2-8 % потолка.");

// ——— 11. Устойчивость ———
s = p.addSlide(); head(s, 9, "Цена выбора модели по тесту");
s.addImage({ path: FIG+"04-ustoychivost.png", x:0.7, y:1.26, w:11.9, h:4.257 });
s.addText([
  {text:"Зелёные столбцы — конфигурация, выигравшая конкурс из десяти на тестовых данных: +57 б. п. в среднем за год. ", options:{}},
  {text:"Оранжевые — та, что зафиксирована ДО теста: −9 б. п. и два года из шести в плюсе. ", options:{bold:true}},
  {text:"Поэтому конфигурация фиксируется до теста, а максимум по тесту мы показываем отдельно и результатом не считаем."},
], { x:0.7, y:5.68, w:11.9, h:1.05, isTextBox:true, margin:0, fontFace:BODY, fontSize:13, color:MUTED, lineSpacing:19 });
s.addNotes("Устойчивость, измеренная на тех же годах, по которым модель и выбиралась, ничего не значит. Разрыв между двумя сериями — 66 б. п. — и есть цена выбора по тесту.");

// ——— 11. Продукт ———
s = p.addSlide(); head(s, 10, "Продукт: три компонента, ни одному не нужен прогноз");
card(s, 0.7, 1.35, 3.85, 2.75, "1 · Индикатор на экране",
  "Три состояния по положению курса в квартальном диапазоне: дёшево — нейтрально — дорого.\n\nЭто факт, а не прогноз. Считается одной формулой, работает с первого дня.", GREEN);
card(s, 4.73, 1.35, 3.85, 2.75, "2 · Оповещение об уровне",
  "Порог задаёт сам клиент.\n\nПревращает честное «подождите» из повода уйти из приложения в повод остаться. И даёт данные о намерении, которых у нас нет.", GREEN);
card(s, 8.75, 1.35, 3.85, 2.75, "3 · Правило отбора",
  "Кому и когда написать: близость к дате выплаты, достаточный баланс, календарь праздников коридора.\n\nРанжируем по готовности переводить, а не по качеству сигнала.", GREEN);
s.addShape(p.ShapeType.roundRect, { x:0.7, y:4.32, w:11.9, h:2.3, fill:{color:PAPER}, rectRadius:0.06 });
s.addText("Калибровка индикатора", { x:1.0, y:4.5, w:11.3, h:0.38, isTextBox:true, margin:0,
  fontFace:HEAD, fontSize:15.5, color:INK, bold:true });
s.addText([
  {text:"Нижний дециль: +52 б. п.", options:{bold:true, color:GREEN, breakLine:true}},
  {text:"Середина, 65 % дней: выгода ≈ 0 — честный серый", options:{color:MUTED, breakLine:true}},
  {text:"Верхний дециль: −55 б. п.", options:{bold:true, color:RED, breakLine:true}},
  {text:"Три состояния, а не пять: данные различают только края. Серый на две трети дней — причина доверять зелёному.", options:{color:MUTED}},
], { x:1.0, y:4.92, w:11.3, h:1.5, isTextBox:true, margin:0, fontFace:BODY, fontSize:12.5, lineSpacing:19 });

// ——— 12. Риски и план ———
s = p.addSlide(); head(s, 11, "Основные риски и план до финала");
s.addText("Ограничения", { x:0.7, y:1.28, w:5.87, h:0.36, isTextBox:true, margin:0,
  fontFace:HEAD, fontSize:16, color:RED, bold:true });
s.addText([
  {text:"Масштаб эффекта — десятки базисных пунктов: около 84 ₽ на переводе 30 000 ₽", options:{bullet:true, breakLine:true}},
  {text:"Модель ошибается в двух случаях из трёх — свойство задачи, а не дефект", options:{bullet:true, breakLine:true}},
  {text:"Сезонность — самый сильный и самый хрупкий признак: истории всего семь лет", options:{bullet:true, breakLine:true}},
  {text:"2022 год — 13 % наблюдений и 58–64 % всей дисперсии", options:{bullet:true, breakLine:true}},
  {text:"Нет данных клиентов: эластичность и доля «гибких» не измеримы до пилота", options:{bullet:true, breakLine:true}},
  {text:"Курс приложения ≠ курс ЦБ, работаем на прокси", options:{bullet:true}},
], { x:0.7, y:1.72, w:5.87, h:2.72, isTextBox:true, margin:0, valign:"top", fontFace:BODY, fontSize:11.5,
     color:MUTED, lineSpacing:16, paraSpaceAfter:4 });
s.addText("План до 07.09", { x:6.73, y:1.28, w:5.87, h:0.36, isTextBox:true, margin:0,
  fontFace:HEAD, fontSize:16, color:GREEN, bold:true });
s.addText([
  {text:"Пт: публичный signals_as_of(T), signals.csv, правило комбинирования, экономика в рублях", options:{bullet:true, breakLine:true}},
  {text:"Сб: калибровка светофора по коридорам, тексты уведомлений, прототип экрана, дизайн пилота", options:{bullet:true, breakLine:true}},
  {text:"Вс: сборка презентации, два прогона с таймером, финальный журнал допущений", options:{bullet:true}},
], { x:6.73, y:1.72, w:5.87, h:2.72, isTextBox:true, margin:0, valign:"top", fontFace:BODY, fontSize:11.5,
     color:MUTED, lineSpacing:16, paraSpaceAfter:4 });
s.addShape(p.ShapeType.roundRect, { x:0.7, y:4.62, w:11.9, h:2.0, fill:{color:INK}, rectRadius:0.06 });
s.addText("Открытые вопросы, от которых зависит размер эффекта", { x:1.0, y:4.78, w:11.3, h:0.38,
  isTextBox:true, margin:0, fontFace:HEAD, fontSize:15.5, color:AMBER, bold:true });
s.addText("Какова доля клиентов, способных отложить перевод? Если гибких 10 %, пилот на всей базе покажет ноль из-за разбавления.\n\nКлиент фиксирует сумму отправки или получения? Если получения, лучший курс означает меньше отправленных рублей — знак влияния на выручку меняется.",
  { x:1.0, y:5.20, w:11.3, h:1.3, isTextBox:true, margin:0, fontFace:BODY, fontSize:12, color:"C7CFD8", lineSpacing:17 });

// ——— 13. Команда ———
s = p.addSlide(); head(s, 12, "Команда и распределение задач");
const team = [
  ["Александр Тармаев", "Product Engineer", "Продуктовое видение, распределение работы, MVP",
   "Постановка продуктовой задачи и границы MVP · разбор девяти продуктовых гипотез · предложение клиентской метрики и её обоснование · журнал допущений и развилок · дизайн пилота · распределение работ"],
  ["Даниил Недайборщ", "AI Engineer · разработка", "Данные, признаки, инфраструктура честности",
   "Загрузчик ЦБ с построчной нормировкой номинала · 79 признаков только по прошлому · доказательство отсутствия заглядывания в будущее и 16 автотестов · walk-forward с очисткой · воспроизводимость репозитория"],
  ["Иван Калинин", "AI Engineer · машинное обучение", "Модели, отбор признаков, валидация результата",
   "Пять семейств моделей, включая CatBoost и XGBoost · честный отбор признаков · подбор конфигураций и диагностика провала на 80 признаках · две модели под две метрики · испытания результата"],
];
team.forEach((t, i) => {
  const y = 1.30 + i*1.75;
  s.addShape(p.ShapeType.roundRect, { x:0.7, y, w:11.9, h:1.58, fill:{color:PAPER}, rectRadius:0.06 });
  s.addText(t[0], { x:0.98, y:y+0.17, w:2.7, h:0.36, isTextBox:true, margin:0,
    fontFace:HEAD, fontSize:15, color:INK, bold:true });
  s.addText(t[1], { x:0.98, y:y+0.56, w:2.7, h:0.32, isTextBox:true, margin:0,
    fontFace:BODY, fontSize:11.5, color:AMBER, bold:true });
  s.addText(t[2], { x:0.98, y:y+0.90, w:2.7, h:0.54, isTextBox:true, margin:0,
    fontFace:BODY, fontSize:10, color:GREY, lineSpacing:12.5 });
  s.addText(t[3], { x:3.85, y:y+0.22, w:7.35, h:1.18, isTextBox:true, margin:0,
    fontFace:BODY, fontSize:11, color:MUTED, lineSpacing:15 });
  s.addText("33 %", { x:11.35, y:y+0.60, w:1.05, h:0.38, isTextBox:true, margin:0,
    fontFace:BODY, fontSize:14, color:AMBER, bold:true, align:"center" });
});
note(s, "Работа велась совместно, роли разделены по зонам ответственности; вклад участников сопоставим.");

p.writeFile({ fileName: "submission/prezentaciya-promezhutochnaya.pptx" })
 .then(f => console.log("готово:", f));
