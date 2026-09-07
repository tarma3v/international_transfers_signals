/** Final defence deck. Edit the latest main reference in place, preserving its design.
 * Requires @oai/artifact-tool. Source reference is pinned, never checks out main.
 * Output is a draft under tmp; finalize/export PDF separately.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';

const ROOT=path.resolve(new URL('..',import.meta.url).pathname);
const BUILD=path.join(ROOT,'tmp/final_presentation');
const RUNTIME=process.env.ITMO_RUNTIME || '/Users/jeck5iv/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const {FileBlob,PresentationFile}=await import(pathToFileURL(path.join(RUNTIME,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')));
const SOURCE_REF='3263473';
const SOURCE=path.join(BUILD,'reference/submission/prezentaciya-finalnaya.pptx');
await fs.mkdir(path.dirname(SOURCE),{recursive:true});
await fs.writeFile(SOURCE,execFileSync('git',['show',`${SOURCE_REF}:submission/prezentaciya-finalnaya.pptx`],{cwd:ROOT,maxBuffer:30*1024*1024}));
const p=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const original=[...p.slides.items];
const snap=await p.inspect({kind:'slide,textbox,shape,image,table,layout',maxChars:1000000});
const inventory=snap.ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x));
await fs.writeFile(path.join(BUILD,'source-inspect.ndjson'),snap.ndjson);
const C={ink:'#1A2530',paper:'#F4F6F8',white:'#FFFFFF',amber:'#9C5B12',green:'#16674F',red:'#9E3226',grey:'#6E7A88',muted:'#4E5966',lightAmber:'#E0A75E',lightGreen:'#6FBFA1',lightRed:'#E89484'};
const I=96;
const box=(x,y,w,h)=>({left:x*I,top:y*I,width:w*I,height:h*I});
const tz='https://talenttrack.aitalenthub.ru/hackathon/cases/455';
const repo='https://github.com/tarma3v/international_transfers_signals/tree/ivan-experiments/';
const sourceRepo='https://github.com/tarma3v/international_transfers_signals/blob/'+SOURCE_REF+'/';
const common='Кейс 455: '+tz+'\nQ&A 04.09.2026 и 05.09.2026: главная метрика lift по hit rate, выгода ±h отдельно. Только факты о прошлом и настоящем в клиентском тексте. Данные, доступные на дату T, включая уже опубликованный курс на завтра, допустимы. Курс ЦБ не является курсом исполнения.\n';
const evidence='Источник: '+repo+'results/research/after_publication/ap37_effective/retrospective_all_horizons.csv\nЗафиксированный кандидат ap26_core_mature_precision_calendar_fallback_cap2. База: действующий сегодня курс ЦБ, будущие публикации. h5: 2024-01-09..2026-08-25, 3260 строк, 695 сигналов. Параметры причинные, но семейство выбрано после многих опытов на уже открывавшейся истории. Это ретроспектива, не новый независимый holdout. Исторический receipt предполагается по календарю (18:30 МСК), фактический журнал получения не сертифицирован. В рабочем контракте обязателен verified_receipt_at.\n';
const edits=[];
function edit(n,prefix,value){
 const hits=inventory.filter(r=>r.slide===n&&r.kind==='textbox'&&r.text.startsWith(prefix));
 if(hits.length!==1)throw Error(`Expected unique text ${n}: ${prefix}, got ${hits.length}`);
 const target=p.resolve(hits[0].id);
 if(hits[0].text.includes('\n')){
  target.text=value;
  target.text.style={typeface:'Calibri',fontSize:12.5*4/3,color:[7,8,18,24].includes(n)?'#C7CFD8':C.muted,autoFit:'none',insets:{top:0,bottom:0,left:0,right:0}};
 }else target.text.replace(hits[0].text,value);
 edits.push({id:hits[0].id,value});
}
function notes(s,body){s.speakerNotes.textFrame.setText(common+body);}
function text(s,value,x,y,w,h,size=13.5,color=C.muted,opts={}){
 const a=s.shapes.add({geometry:'textbox',position:box(x,y,w,h),fill:'none',line:{fill:'none',width:0}});
 a.text=value;
 a.text.style={typeface:'Calibri',fontSize:size*4/3,color,autoFit:'none',verticalAlignment:'top',insets:{top:0,bottom:0,left:0,right:0},...opts};
 return a;
}
function rect(s,x,y,w,h,fill=C.paper){return s.shapes.add({geometry:'roundRect',position:box(x,y,w,h),fill,line:{fill:'none',width:0},borderRadius:5.76});}
function card(s,x,y,w,h,ttl,body,color=C.ink){
 rect(s,x,y,w,h);text(s,ttl,x+.28,y+.2,w-.56,.56,15.5,color,{typeface:'Cambria',bold:true});
 text(s,body,x+.28,y+.83,w-.56,h-1.0,12.5,C.muted);
}
function band(s,y,h,ttl,body){
 rect(s,.7,y,11.9,h,C.ink);text(s,ttl,1,y+.17,11.3,.36,15.5,C.lightAmber,{typeface:'Cambria',bold:true});
 text(s,body,1,y+.63,11.3,h-.83,12.5,'#C7CFD8');
}
function footer(s,value){text(s,value,.7,6.9,11.9,.36,10.5,C.grey,{italic:true});}
function factualPushCopy(s,x,y,w,h,size){
 // Keep the source screenshot. Correct only the notification body with editable text.
 s.shapes.add({geometry:'rect',position:box(x,y,w,h),fill:'#F2F2F2',line:{fill:'none',width:0}});
 text(s,'За 30 000 ₽ — 3 198 сомони.\nЛучше было 8 дней из 90.\nКурс ЦБ на 4 сентября.',x+.015,y+.008,w-.03,h-.016,size,'#30353B');
}
function clearBody(n,title,label){
 const s=original[n-1];
 for(const r of inventory.filter(r=>r.slide===n&&r.bbox&&r.bbox[1]>=105)){
   try{s.elements.deleteById(p.resolve(r.id).id);}catch{}
 }
 // Header and numbered circle remain native and unchanged in position.
 edit(n,inventory.find(r=>r.slide===n&&r.kind==='textbox'&&r.bbox?.[0]>120&&r.bbox?.[1]<50).text,title);
 if(label)edit(n,inventory.find(r=>r.slide===n&&r.kind==='textbox'&&r.bbox?.[0]<100&&r.bbox?.[1]<70).text,label);
 return s;
}
function newSlide(title,label){
 const s=p.slides.add();s.setLayout(p.layouts.items[0]);s.background.fill=C.white;
 s.shapes.add({geometry:'ellipse',position:box(.7,.52,.42,.42),fill:label.startsWith('П')?C.grey:C.amber,line:{fill:'none',width:0}});
 text(s,label,.7,.54,.42,.38,label.startsWith('П')?10:13,C.white,{alignment:'center',bold:true});
 text(s,title,1.28,.44,11.3,.62,27,C.ink,{typeface:'Cambria',bold:true});return s;
}
function table(s,values,x,y,w,h,widths,size=12){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x*I,top:y*I,width:w*I,height:h*I,values,columnWidths:widths.map(v=>v*I)});
 t.borders.assign({style:'solid',fill:'#DCE2E8',width:.65});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?C.paper:C.white;
  cell.text.style={typeface:'Calibri',fontSize:size*4/3,color:C.muted,bold:r===0};
 }
 return t;
}
function chart(s,title,categories,series,x,y,w,h){
 const ts={typeface:'Calibri',fontSize:16,fill:C.muted};
 return s.charts.add('bar',{position:box(x,y,w,h),title,titleTextStyle:{typeface:'Cambria',fontSize:21,fill:C.ink,bold:true},categories,
 series:series.map((q,i)=>({...q,values:q.values.map(v=>Number(v.toFixed(12))),fill:i===0?C.green:'#A6AFB8',valuesFormatCode:'0.0%'})),
 barOptions:{direction:'column',grouping:'clustered',gapWidth:100},hasLegend:true,legend:{position:'bottom',textStyle:ts},
 xAxis:{textStyle:ts,line:{fill:'#BFC7CE',width:.6},majorGridlines:null},
 yAxis:{min:0,max:1,majorUnit:.2,numberFormatCode:'0%',textStyle:ts,majorGridlines:{fill:'#E8ECEF',width:.6}},
 dataLabels:{showValue:true,position:'outEnd',textStyle:{...ts,bold:true,fontSize:16}},chartFill:C.white,plotAreaFill:C.white,
 chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}});
}
function parseCSV(raw){const lines=raw.trim().split(/\r?\n/),heads=lines.shift().split(',');return lines.map(l=>Object.fromEntries(l.split(',').map((v,i)=>[heads[i],v])));}
const all=parseCSV(await fs.readFile(path.join(ROOT,'results/research/after_publication/ap37_effective/retrospective_all_horizons.csv'),'utf8'));
const m=all.filter(r=>r.candidate==='ap26_core_mature_precision_calendar_fallback_cap2');
const h5=m.find(r=>r.h==='5');
if(Math.abs(+h5.hit_rate/+h5.base_rate-(+h5.pooled_lift))>1e-12)throw Error('Lift arithmetic');

// 1. Original cover, typography and placements preserved.
edit(1,'Сигнальный слой над','Сигнальный слой над существующим переводом: RUB → TJS, UZS, KGS, AMD, KZT. Модель выбирает момент, а клиент видит понятный факт о текущем курсе и истории.');
notes(original[0],'Финальный кандидат final-temperature-v1-2026-09-07. '+repo+'model/final_temperature_model_v1.json\nМакеты команды из main, исходник '+SOURCE_REF+'.');

// 2. Keep all original six card positions. Replace unsupported money promises.
edit(2,'Клиент не видит разницы','Клиент видит, сколько получит семья и как текущий курс выглядит на фоне истории. Банк выбирает редкий повод напомнить о переводе.');
edit(2,'+63 б. п.','2,492 lift');
edit(2,'столько выигрывает клиент','Попадание: 73,38 % против 29,45 % у случайного дня. Горизонт: следующие пять публикаций ЦБ.');
edit(2,'2 дня из 3','+74,55 б. п.');
edit(2,'ничем не выделяются','Выгода момента в окне ±5 публикаций. Это результат по публичному курсу ЦБ, не обещанная экономия клиента.');
edit(2,'0 обещаний','1,04 / нед.');
edit(2,'правило срабатывает','Столько сигналов приходится на валюту в среднем. Не больше двух за календарную неделю.');
edit(2,'Показывает, сколько получит','Сумма получателя и сравнение с историей. Температура модели остаётся внутренней оценкой.');
edit(2,'Кому и когда написать','В пилоте банк добавит согласие клиента и общий лимит сообщений. Клиентских данных на хакатоне нет.');
edit(2,'Все три компонента','AP37, h=5, 09.01.2024–25.08.2026. Открытая ретроспектива после получения нового курса ЦБ.');
notes(original[1],evidence+'Lift на слайде pooled=0.7338129496402878/0.294478527607362=2.491906474820144. Adjusted по коридору/периоду=2.5091882251629154, это другая агрегация. Выгода ±h=74.55139335041945 б.п. Клиентскую выгоду/выручку ещё нужно измерить в пилоте.');

// 3. Triggers still use the original customer journey styling.
edit(3,'Первые два шага','Первые два шага клиент не видит. Модель обновляется по мере поступления данных: история ЦБ, завершённые биржевые свечи, затем новый опубликованный курс.');
edit(3,'Уровень клиента','Модель момента');
edit(3,'Порог, который он','Оценка удачности перевода на ближайшие пять публикаций. Отдельный фильтр решает, нужен ли пуш.');
edit(3,'Редкость момента','Факт для клиента');
edit(3,'Курс в верхней части','Уровень относительно прошлых 90 дней, изменение за неделю или достигнутый порог клиента. Только проверяемый факт.');
edit(3,'Календарь коридора','Свой уровень');
edit(3,'Навруз, Рамазан','Клиент сам называет нужную сумму получателю. Это дополнительная ветка продукта, отдельно от качества модели.');
edit(3,'Развилка кейса:','Модель выбирает момент, текст объясняет настоящее');
edit(3,'Три источника не','Температура 0–100 используется внутри системы. В пуш и виджет не выводим вероятность будущего курса. Если для сильного сигнала нет подходящего факта, коммуникацию пропускаем.');
edit(3,'Схема пути целиком','Частоту после клиентских фильтров нужно отдельно измерить в пилоте. Метрики модели считаются до них.');
notes(original[2],repo+'model/final_temperature_model_v1.json\n'+repo+'model/final_temperature_model_v1.json\nФильтр клиентского текста, местного времени и общего лимита банка — продуктовый контракт, его влияние не включено в AP37 scorecard.');

// 4. Preserve cards and clarify what is actually implemented.
edit(4,'Cooldown против','Прореживание серий');
edit(4,'Сигналы приходят','AP37 ограничивает поток двумя сигналами в неделю на валюту. Сильное ядро дополняется резервным сигналом при долгой тишине.');
edit(4,'Блэкауты по','Календарные ограничения');
edit(4,'24 апреля','Продуктовое правило: учитываем значимые даты коридора. Например, не используем поздравительный повод 24 апреля для Армении.');
edit(4,'Окно задаётся','Отправка по местному времени клиента. Если вечерняя публикация попала в ночь, утром заново проверяем сигнал и курс.');
edit(4,'Порог редкости','Достоверность и свежесть');
edit(4,'Ниже порога','Не получен источник, нет сильного сигнала или безопасного текста: молчим. Слабый сигнал не отправляем ради плана.');
edit(4,'Что здесь честно','Граница прототипа и пилота');
edit(4,'Cooldown и добор','Фильтр AP37 уже реализован: в среднем 1,04 сигнала на валюту в неделю. Общий лимит на клиента, время отправки и разрешения должен подключить банк.');
notes(original[3],evidence+repo+'research/after_publication_ap37_effective_models.py\nКалендарная неделя ISO, cap2 не означает автоматический cap2 на любом скользящем семидневном окне. Для клиента общий лимит банка суммарный.');

// 5–8. Preserve phone assets. Replace one indirect forecast with a historical fact.
factualPushCopy(original[4],1.058,4.0,1.665,.345,5.4);
edit(5,'Не курс и не проценты.','На первом месте сумма получателя. В макете она иллюстративная, по ЦБ. В пилоте нужна актуальная котировка банка.');
edit(5,'Значимо одно состояние','Температура остаётся внутри');
edit(5,'И клиенту оно выглядит','Пользователь видит сумму, дату и факт о прошлом. Не показываем ему «80 % шанса» или утверждение «курс вырастет».');
edit(5,'Сто вариантов текста','Макет команды сохранён. Числа в экранах иллюстративные, по ЦБ, а не реальная банковская котировка.');
notes(original[4],sourceRepo+'design/pushi-100-variantov.md\nUI-изображения main '+SOURCE_REF+'. В нижнем пуше фраза «ждать чаще не помогало» заменена редактируемым фактом редкости без намёка на будущее. Для пилота исторические шаблоны проверяются на фактах и используют банковскую котировку.');
edit(6,'За одну публикацию','При открытии заново получаем котировку банка и внутреннюю оценку момента. Сохраняем контекст пуша, но показываем актуальные условия.');
edit(6,'Сигнал считается по','Бэктест выполнен по ЦБ. Суммы в макете примерные. Перед подтверждением клиент должен увидеть реальный курс, комиссию и итоговую сумму.');
notes(original[5],sourceRepo+'submission/12-dizayn-interfeysa.md\nQ&A 05.09, с.7: курс приложения в реальном времени и связан с поставщиками ликвидности.');
edit(7,'Три равноправных исхода.','Три равноправных исхода. Перевести, задать свой уровень или ничего не делать. Клиент сам принимает решение.');
edit(7,'Без ответа честность','Если клиенту не подходит сумма, он может задать свой уровень. Оповещение сохраняет его намерение без давления.\n\nШкала на экране сравнивает текущую сумму с историей. Это не внутренняя температура модели и не обещание будущего.');
notes(original[6],sourceRepo+'submission/12-dizayn-interfeysa.md\nВажно различать положение суммы в историческом диапазоне и вероятность future-only события. Визуальный индикатор из main менять для температуры не требуется.');
edit(8,'Две трети дней','В большинство дней модель не создаёт пуш. Исторический индикатор на экране и оповещение об уровне продолжают работать.\n\nОтключение модельных подсказок не должно удалять уровень, который клиент задал сам. Это отдельные настройки.');
notes(original[7],evidence+'30 дней жизни заявки — продуктовая гипотеза команды, не условие ТЗ.');

// 9. Current headline results, native evidence rather than an old ranking screenshot.
{
 const s=clearBody(9,'Сигнал: 73,38 % удачных дней вместо 29,45 %');
 text(s,'После получения нового курса ЦБ. Оцениваем относительно курса, действующего сегодня.',.7,1.2,11.9,.46);
 chart(s,'Попадание на следующих пяти публикациях',['h = 5'],[
  {name:'Сигналы AP37',values:[+h5.hit_rate]},{name:'Случайный день',values:[+h5.base_rate]}],.7,1.9,6.4,4.55);
 card(s,7.4,1.92,5.2,1.76,'Что добавил отбор сигналов','Простое правило «завтра не дешевле»: lift 1,962. Итоговый AP37: 2,492.',C.green);
 card(s,7.4,3.88,5.2,2.22,'Сравнение по одной базе','Один период, пять валют, действующий курс ЦБ. У AP37 695 сигналов и 1,04 сигнала в неделю. У простого правила 1,43.',C.ink);
 footer(s,'09.01.2024–25.08.2026. Открытая ретроспектива. Это не банковская доходность и не новый независимый тест.');
 notes(s,evidence+'Простой контроль known_next_not_lower_cd3: hit 0.5778474399164054, pooled lift 1.9622735980494599, frequency 1.4298826040554962. Разная частота оговорена, это не изолированная абляция при одинаковом бюджете.');
}

// New approved temperature example, retained byte-for-byte and labelled as selected.
{
 const s=newSlide('Температура и курс: пример для узбекского сума','3');
 s.images.add({blob:new Uint8Array(await fs.readFile(path.join(ROOT,'output/defense_visuals/temperature_example.png'))),contentType:'image/png',alt:'Выбранный пользователем пример UZS: курс, температура и сигналы, 18 мая – 23 июня 2026',fit:'contain',position:box(.7,1.23,11.9,5.26)});
 footer(s,'Выбранный удачный фрагмент открытой ретроспективы, 18.05–23.06.2026. Температура внутренняя. Общие метрики — на всём периоде.');
 notes(s,evidence+repo+'output/defense_visuals/chart_evidence.json\nИсходное изображение пользователя сохраняется без изменений. Это иллюстрация AP50 temperature и AP37 push после публикации. 6/6 попаданий на фрагменте не означает 100% точности модели. Нельзя причинно интерпретировать наложение осей разного масштаба.');
 s.moveTo(9);
}

// Original timing slide becomes a native, editable routing scheme.
{
 const s=clearBody(10,'Модель в течение дня: какой источник уже доступен');
 text(s,'Часы задают срез данных. Переключение после ЦБ запускает факт получения, а не наступление 18:00.',.7,1.2,11.9,.52);
 card(s,.7,1.95,2.85,3.25,'До рынка','История ЦБ.\n\nБустинг по трендам, уровням, волатильности и календарю.\n\nОграниченная уверенность.');
 card(s,3.72,1.95,2.85,3.25,'10:30–15:30','Завершённые свечи CNY/RUB начиная с 10:00.\n\nСредняя цена к текущему ЦБ, ранг по прошлым дням и калибровка.',C.green);
 card(s,6.74,1.95,2.85,3.25,'15:30–получение','Удерживаем последний доступный дневной прогноз.\n\nБез новых данных не изображаем обновление.',C.amber);
 card(s,9.76,1.95,2.85,3.25,'После получения','Уже известное изменение ЦБ на завтра и исторические модели.\n\nCatBoost оценивает вероятность, AP37 отдельно отбирает пуш.',C.green);
 band(s,5.46,1.26,'Когда отправляем','Сильный сигнал после получения ЦБ, затем проверка местного времени и условий банка. При переносе на утро считаем заново.');
 footer(s,'Выходной или сбой источника: история и последнее доступное состояние. Нет новых данных — нет фиктивного обновления.');
 notes(s,repo+'model/final_temperature_model_v1.json\nT4: HistGradientBoosting на 53 признаках. T5: causally ranked CNY/RUB basis + поквартальная логистическая калибровка по валюте/горизонту. T14 добавляет ранний CNY perpetual только h1/h3, если завершённая свеча доступна. T16 вечером улучшает только оценку размера эффекта h3/h5, не заменяет температуру и не переносит автоматически AP37 lift на каждый час. 15:30 — срез выбранной модели, не конец всей торговой сессии.');
}

// 11. Stale notification: no unsupported guarantee of OS replacement or 1-day life.
edit(11,'Одно живое уведомление','Обновляем контекст при открытии');
edit(11,'Следующая публикация','Не рассчитываем, что старый пуш обязательно заменится на устройстве. По нажатию всегда загружаем свежие условия и показываем изменение суммы.');
edit(11,'Снять просроченный','Дата и источник в сообщении');
edit(11,'Ни одна платформа','Пуш остаётся историческим сообщением. Дата курса позволяет понять, к какому моменту относилась сумма.');
edit(11,'Срок жизни рекомендации','Свежая оценка вместо срока «сутки»');
edit(11,'Отличить сигнал','Температуру пересчитываем из доступных данных. Если источник устарел, не показываем старую оценку как новую и не обещаем прежние условия.');
edit(11,'Слева – тот же','Макеты сценария «момент изменился». Суммы по ЦБ иллюстративные. В пилоте используем актуальные условия банка.');
notes(original[10],repo+'model/final_temperature_model_v1.json\nЭкран 03 исходного прототипа показывает желаемое обновление пуша. Оно не является гарантией доставки или замены на iOS/Android. Реализация транспорта вне скоупа.');

// 12. Keep the exact table layout while remove unverified delivery promises.
edit(12,'Три правила, которые','Для каждого сообщения проверяем факт, дату и источник. Примеры ниже иллюстративные, с расчётом по ЦБ.');
edit(12,'Деньги первыми','Понятная сумма');
edit(12,'Сумма получателя в первой','В пилоте сумма из котировки банка. В прототипе расчёт по ЦБ явно назван ориентиром.');
edit(12,'Сумма в каждом сообщении','Только проверяемые факты');
edit(12,'Сумма стоит в любом','Историческое сравнение не заменяем прогнозом. Если подходящего факта нет, модельный пуш пропускаем.');
const pushTable=original[11].tables.items[0];
const pushRows=[
 ['Уровень достигнут','Дошло до вашего уровня','По ЦБ за 30 000 ₽: 4 402 000 сум. Ваш ориентир: 4 400 000. Курс на 4 сентября.'],
 ['Редкий момент','Такой курс бывает нечасто','По ЦБ за 30 000 ₽: 3 198 сомони. Лучше было 8 дней из 90. Курс на 4 сентября.'],
 ['Изменение за неделю','Рубль укрепился к сому','За неделю на 2,1 % по ЦБ. Условия перевода по курсу банка: в приложении.'],
 ['Праздник','Навруз муборак!','Пусть близкие будут рядом. Условия перевода: в приложении.'],
 ['Свой уровень','Сколько должно дойти домой?','Задайте нужную сумму получателю. Сообщим о достижении вашего уровня.'],
];
pushRows.forEach((row,r)=>row.forEach((value,c)=>pushTable.cells.set(r+1,c,value)));
edit(12,'Ни одного обещания','Не используем «курс вырастет», «успейте, пока не подорожало», «гарантируем лучший курс». Прогноз остаётся внутри модели, клиент получает факт.');
notes(original[11],sourceRepo+'design/pushi-100-variantov.md\nЧисла и даты примеров в этой таблице демонстрационные. В пилоте обязательны актуальный источник, проверка истины каждого факта и согласование банка.');

// 13–15. Current project scope and pilot, preserving all template shapes.
edit(13,'Основное ·','Основное: пилот сигнального слоя');
edit(13,'Порог называет сам клиент, поэтому','Финальный кандидат объединяет доступные источники и отбирает редкие сигналы. Главный горизонт для объяснения: пять публикаций.\n\nСначала сопоставляем сигнал с реальной котировкой банка. Затем проверяем на клиентах дополнительный объём переводов против контрольной группы.\n\nКод работает на исторических данных. Подключение боевых источников и отправки ещё требуется.');
edit(13,'Дополнительно ·','Дополнительно: уровень и интерфейс');
edit(13,'Сигнал правила.','Клиент задаёт нужную сумму получателю. При открытии пуша видит актуальные условия, историческое сравнение и возможность оставить оповещение.\n\nТемпература помогает системе оценить состояние в любой момент. В интерфейсе сохраняются факты, а не вероятность будущего.\n\nКалендарные поводы проверяем отдельно, не прибавляем их к измеренной точности модели.');
const pilotT=original[12].tables.items[0];
[['Публичный курс и курс перевода могут расходиться','Нужна проверка на котировках банка, комиссиях и сроке действия'],['Историческая связь может измениться','Новый период и пилот без перенастройки на результатах'],['Пуш может сдвинуть уже запланированный перевод','Меряем дополнительный объём и каннибализацию за полный период']].forEach((r,i)=>r.forEach((v,j)=>pilotT.cells.set(i,j,v)));
notes(original[12],evidence+'Реальные объёмы, маржа, согласия, адресаты и комиссии не предоставлены. Нет обещания срока окупаемости или статистической мощности пилота без этих данных.');
edit(14,'1 · Отложенный перевод','1 · Новый период');
edit(14,'Логичное продолжение','Зафиксировать модель и собирать новые результаты без подбора по ним. Хранить точное время получения каждого курса и свечи.');
edit(14,'2 · Cooldown и частота','2 · Общий лимит клиента');
edit(14,'Дописать то, что','Подключить разрешения и общую коммуникационную политику банка. Проверить частоту после текстового фильтра и переноса по часовым поясам.');
edit(14,'Померить деградацию','Измерить сигнал на исполнимой котировке, с комиссией и сроком действия. Не переносить базисные пункты ЦБ в обещания клиенту.');
edit(14,'Две руки против','Сравнить модельные пуши, оповещения об уровне и контроль без новых сообщений. Главная бизнес-метрика: дополнительный net-объём переводов.');
edit(14,'Порядок отражает','Пилоту нужны банковские котировки, согласия и контрольная группа. Размер и срок определяем по трафику банка.');
notes(original[13],common+'Исследования с клиентами и боевые интеграции в скоуп хакатона не входят.');
{
 const s=clearBody(15,'Температура внутри, понятные факты на экране');
 card(s,.7,1.45,5.87,3.36,'Что рассчитывает система','Температура от 0 до 100: оценка вероятности, что на выбранном горизонте более дешёвого дня не будет.\n\nОтдельно оцениваем размер изменения и свежесть источника. Для пуша используем собственный строгий фильтр.',C.green);
 card(s,6.73,1.45,5.87,3.36,'Что видит клиент','Сколько получит близкий человек. Как текущий курс выглядит на фоне истории. Достигнут ли его собственный уровень.\n\nШкала в макете сравнивает с прошлыми днями. Температуру, вероятность и обещание будущего не выводим.',C.ink);
 band(s,5.12,1.50,'Результат для кейса','Сигнальный слой, редкий поток событий и готовый сценарий открытия пуша. Доказательство на публичных курсах есть, проверка денежного эффекта остаётся за пилотом.');
 footer(s,'AP37: lift 2,492, выгода ±5 +74,55 б. п., частота 1,04 на валюту в неделю. Открытая ретроспектива.');
 notes(s,evidence+'Температура=100×калиброванная вероятность. Это внутренняя модельная оценка, а не новая пользовательская метрика. Полный интерфейс наследуем из main.');
}
edit(16,'Дальше – материалы','Дальше: проверка доступности данных, определения метрик, результаты по горизонтам и валютам, состав моделей и интерфейс.');
edit(17,'Доказательство честности','Доступность данных, определения метрик, результаты по горизонтам, сравнение подходов команды, устройство модели и интерфейс.');

// Appendix P1. No unqualified claim that all sources have certified historical timestamps.
edit(18,'Заглядывания в будущее','Как мы контролируем доступность данных');
edit(18,'Требование дисквалифицирующее','Каждый прогноз должен использовать только то, что уже доступно на момент запроса. Проверяем границы источников и созревание ответов.');
edit(18,'Один временной срез','Срез на произвольный момент');
edit(18,'past_slice(values','Завершение свечи и получение курса должны быть не позже запроса. Новый курс ЦБ не включается автоматически в 18:00.');
edit(18,'Порча будущего двумя','Обучение только на прошлом');
edit(18,'ml/leakage.py','Квартальное обучение. Только ответы, для которых весь горизонт уже завершён. Дополнительный зазор в два дня.');
edit(18,'Подсаженная утечка','Проверки неизменности');
edit(18,'В детектор специально','Меняем будущие данные и убеждаемся, что прогноз для более раннего среза не изменился. Отдельно проверяем защиту нового курса ЦБ.');
edit(18,'Ту же процедуру','Что ещё нельзя считать доказанным');
edit(18,'Порча всех наблюдений','Оценки 2024–2026 относятся к открытой ретроспективе: период уже использовался в исследованиях. Новый независимый тест ещё нужен.\n\nИсторическое время получения курса в бэктесте моделируется по календарю. Для пилота нужен журнал реальных получений, котировки банка и проверка полной цепочки отправки.');
edit(18,'49 тестов','Финальный кандидат: model/final_temperature_model_v1.json. Тесты причинности не заменяют новый независимый период.');
notes(original[17],evidence+repo+'tests/test_final_temperature_model.py\n'+repo+'model/final_temperature_model_v1.json');

// P2: exact distinction of the required metrics.
{
 const s=clearBody(19,'Какие метрики считаем и что они означают');
 table(s,[['Метрика','Определение','Итог при h = 5'],
 ['Hit rate','Сегодняшний курс не выше любого из следующих h курсов','73,3813 %'],
 ['Lift по точности','Hit rate сигнала / hit rate случайного дня','2,491906'],
 ['Выгода момента ±h','Средний курс вокруг дня сигнала, 95 % ДИ: +59,38…+92,46 б. п.','+74,5514 б. п.'],
 ['Выгода только вперёд','Сравнение со средним следующих h курсов','+133,7337 б. п.'],
 ['Частота','Сигналов на валюту в календарную неделю','1,03842']],.7,1.45,11.9,3.05,[2.65,6.0,3.25],13);
 band(s,4.82,1.76,'Почему в отчётах встречается ещё 2,509','2,492 получается из общих долей 73,38 % и 29,45 %. 2,509 использует базу с поправкой на коридор и период. Это две агрегации одних сигналов, не два разных результата модели.');
 footer(s,'Q&A: lift по попаданию в будущее, выгода ±h отдельно. h считаем в публикациях. Курс в рублях за единицу валюты.');
 notes(s,evidence+'Q&A 05.09 с.3–5. Симметричная разметка локального минимума ±h присутствует в исследовании отдельно от headline lift. Для сообщения «окно закрывается» применяется другой таргет. Полная матрица: '+repo+'results/research/final_case_metric_matrix.csv');
}

// P3. Native chart and all required horizons.
{
 const s=clearBody(20,'Попадания сохраняются на разных горизонтах');
 const rows=m.filter(r=>r.h!=='1');
 chart(s,'Доля удачных дней',rows.map(r=>r.h+(r.h==='3'?' публикации':' публикаций')),[{name:'Сигналы AP37',values:rows.map(r=>+r.hit_rate)},{name:'Случайный день',values:rows.map(r=>+r.base_rate)}],.7,1.25,11.9,4.82);
 text(s,'h=1: 100 % у сигналов, 51,62 % у случайного дня. Следующий курс уже известен, поэтому это не доказательство прогноза.',.7,6.15,11.9,.5,12.5);
 footer(s,'2024–2026, открытая ретроспектива. Для h=3/5/10/20 остаётся неизвестная часть будущего. Разные h имеют разную дату конца.');
 notes(s,evidence+'Все пять горизонтов: '+JSON.stringify(m));
}

// P4. Preserve colleagues' attribution and historical table, but not a post-hoc claim.
edit(21,'Особенность: полная','Независимая реализация');
edit(21,'Ценность: внешняя','Что проверили');
edit(21,'Наше правило, прогнанное','Сверили признаки и расчёт показателей. Совпадение близких чисел помогает проверить код, но не создаёт независимый тест данных.');
edit(21,'Вывод: обучение добавляет','Эти цифры используют верхние 15 % тестовых оценок. Это ретроспективное ранжирование, не порог для реальной отправки.');
text(original[20],'С нашим 2,492 напрямую не сравниваем: отличаются время сигнала, опорный курс, период и правило отбора.',.7,6.26,11.9,.46,12.5,C.red);
notes(original[20],'Источник: main '+SOURCE_REF+' и '+repo+'research/version_b_honest_audit.py\nЧисла таблицы сохранены как исторический результат version_b (Даниил Недайборщ). Порог top15% вычислен по тестовым оценкам, поэтому это post-hoc политика, даже если сами признаки причинные. Честный фиксированный online порог может дать иной результат.');

// P5. Actual intraday algorithm, keeping the same title style.
{
 const s=clearBody(22,'Днём: юань как датчик движения рубля');
 card(s,.7,1.42,5.87,2.18,'1. Берём завершённые свечи','CNY/RUB с 10:00 до текущего среза, максимум до 15:30. Считаем среднюю доступную цену. Незавершённую свечу не используем.',C.green);
 card(s,6.73,1.42,5.87,2.18,'2. Сравниваем с официальным курсом','Отношение биржевой средней к доступному ЦБ CNY показывает, куда уже сдвинулся рынок относительно дневного ориентира.',C.green);
 card(s,.7,3.78,5.87,2.18,'3. Переводим в вероятность','Ранг среди прошлых значений и логистическая калибровка отдельно по валюте и горизонту. Параметры обновляем поквартально.');
 card(s,6.73,3.78,5.87,2.18,'4. Не усложняем без выигрыша','Прямые AMD/RUB и KZT/RUB проверили. Вероятностная модель с KZT ухудшилась в 2025 году, поэтому в итоговую температуру её не включили.');
 footer(s,'15:30, h=5, 2025–2026: ошибка вероятности Brier 0,1572 против 0,1999 у базовой частоты. Меньше — лучше.');
 notes(s,repo+'research/temperature_t5_market_grid.py\n'+repo+'results/research/temperature/t5_market_grid/calibration_metrics.csv\n'+repo+'research/temperature_t46_local_pair_fallback_report.md\nCNY/RUB не создаёт арбитраж и не задаёт один-к-одному остальные валюты. Это ликвидный рублёвый индикатор относительно медленного публичного ориентира. Дневной Brier не следует сравнивать с вечерним без одинаковой выборки.');
}

// P6 screenshots kept exactly. P7 roles are not turned into invented percentages.
factualPushCopy(original[22],2.449,3.295,1.133,.229,3.6);
edit(23,'Новых шагов не добавляется','Семь исходных экранов команды. Шкала показывает положение суммы в историческом диапазоне, а не температуру. Числа иллюстративные по ЦБ, в пилоте нужны реальные условия банка.');
notes(original[22],'Источник: main '+SOURCE_REF+', submission/figures/06-makety-interfeysa.png. Изображение сохранено, поверх тела нижнего пуша добавлен редактируемый факт редкости вместо «ждать чаще не помогало».');
edit(24,'Product Engineer.','AI Product. Продуктовая постановка, клиентский путь, макеты, тексты, базовые индикаторы и комплект материалов.');
edit(24,'AI Engineer. Данные,','AI Engineer. Данные и признаки, независимая реализация version_b, модели и проверка расчётов.');
edit(24,'AI Engineer. Модели,','AI Engineer. Внутридневные и вечерние модели, эксперименты, контроль доступности данных, температура и редкие сигналы.');
edit(24,'Что дало параллельное','Вклад команды в итоговое решение');
edit(24,'Ядро подтвердилось','Продуктовая часть сохраняет существующий путь перевода и безопасные формулировки. Независимые реализации помогли проверить признаки и обнаружить различия в оценке.\n\nФинальное исследование добавило работу по доступности данных, оценку момента в течение дня и управляемый поток сигналов.');
notes(original[23],'Роли и выполненные зоны работы по источникам main и ivan-experiments. Численные доли участия не указаны: инструкция ZAGRUZKA.md называет 33/33/33 заглушкой, фактические значения должны подтвердить участники.');

// P8. Native chart of all five currencies from the verified chart evidence.
{
 const evidenceJson=JSON.parse(await fs.readFile(path.join(ROOT,'output/defense_visuals/chart_evidence.json'),'utf8'));
 // Exact per-currency rows are independently exported in chart_evidence.json.
 const s=newSlide('Результат на всех пяти валютах','П8');
 const currencies=evidenceJson.currency_robustness.map(r=>r.currency);
 const hits=evidenceJson.currency_robustness.map(r=>r.hit_rate);
 const bases=evidenceJson.currency_robustness.map(r=>r.base_rate);
 chart(s,'Попадание на следующих пяти публикациях',currencies,[{name:'Сигналы AP37',values:hits},{name:'Случайный день',values:bases}],.7,1.4,11.9,4.9);
 footer(s,'2024–2026, открытая ретроспектива после получения ЦБ. На каждой валюте около 1,02–1,06 сигнала в неделю.');
 notes(s,evidence+repo+'output/defense_visuals/chart_evidence.json\n'+repo+'results/research/after_publication/ap37_effective/diagnostic_breakdown.csv\nКоридоры связаны общим движением рубля. Пять валют не являются пятью независимыми рынками.');
}

// P9. Explain the genuinely different probability and sparse-push models.
{
 const s=newSlide('После публикации: температура и отбор пушей','П9');
 card(s,.7,1.45,5.87,3.83,'Температура: CatBoost + калибровка','Входы: известное изменение ЦБ на завтра, ранги трёх исторических моделей и их согласие, недавняя частота сигналов, календарь и валюта.\n\nОтдельные модели на 3, 5, 10 и 20 публикаций. Калибровка превращает оценку в вероятность.\n\nПризнак «курс завтра» включается только после получения.',C.green);
 card(s,6.73,1.45,5.87,3.83,'Пуш: AP37, отдельное правило','Сильное ядро из исторических экспертов. Резервный сигнал при низкой частоте или длительной тишине.\n\nКачество резерва оцениваем по уже завершённым прошлым случаям, с учётом валюты. Не более двух событий в неделю.\n\nПуш не равен условию «температура выше 70».',C.ink);
 band(s,5.5,1.17,'Почему помогает новый курс','Одна часть ближайшего будущего уже стала известным фактом. Модель оценивает оставшиеся дни, а базой сравнения остаётся действующий сегодня курс.');
 footer(s,'Температура AP49/AP50. Пуш AP37. Модельные метрики ЦБ не доказывают возможность перевода по старой цене банка.');
 notes(s,evidence+repo+'research/after_publication_ap49_effective.py\n'+repo+'research/after_publication_ap50_temperature_models.py\n'+repo+'research/after_publication_ap37_effective_models.py\nЭксперты AP37: ap26_y20, ridge_survival, distributional_cat. Качество резервной группы: созревшие исходы h20 до даты T минус 2 дня, сглаживание глобальной и локальной частоты. Резерв после разогрева84д, при trailing rate<1 и отсутствии ядра, при тишине>=10д или конце недели с достаточным качеством.');
}

const finalSnap=await p.inspect({kind:'slide,textbox,table,chart,image,notes',maxChars:1000000});
const editedRows=finalSnap.ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x));
for(const e of edits){const row=editedRows.find(r=>r.id===e.id&&r.kind==='textbox');if(row&&row.text!==e.value)throw Error('Text replacement did not persist: '+e.id);}
await fs.writeFile(path.join(BUILD,'final-inspect.ndjson'),finalSnap.ndjson);
await (await PresentationFile.exportPptx(p)).save(path.join(BUILD,'candidate.pptx'));
await fs.writeFile(path.join(BUILD,'build-manifest.json'),JSON.stringify({sourceRef:SOURCE_REF,sourcePath:'submission/prezentaciya-finalnaya.pptx',slides:p.slides.items.length,metrics:h5,originalScreensPreserved:true,temperatureImage:'output/defense_visuals/temperature_example.png',nativeChartSlides:[9,21,26],nativeTableSlides:[13,14,20,22]},null,2));
console.log('Draft created:',p.slides.items.length,'slides');
