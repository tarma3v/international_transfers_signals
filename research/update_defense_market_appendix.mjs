/** Focused revision of the final deck: replace historical version_b appendix.
 * The source deck remains intact; all other slides and notes are preserved.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';

const ROOT=path.resolve(new URL('..',import.meta.url).pathname);
const BUILD=path.join(ROOT,'tmp/market_appendix');
const RT='/Users/jeck5iv/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const SKILL='/Users/jeck5iv/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const source=path.join(ROOT,'submission/prezentaciya-itogovaya-2026-09-07.pptx');
const final=path.join(ROOT,'submission/prezentaciya-itogovaya-2026-09-07-birzha.pptx');
const {FileBlob,PresentationFile}=await import(pathToFileURL(path.join(RT,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')));
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')));
await fs.mkdir(BUILD,{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=await p.inspect({kind:'slide,textbox,shape,image,table,chart,notes,layout',maxChars:1000000});
await fs.writeFile(path.join(BUILD,'before.ndjson'),snap.ndjson);
const records=snap.ndjson.split('\n').filter(Boolean).map(v=>JSON.parse(v));
const anchor=records.find(r=>r.kind==='slide'&&r.slide===22);
if(!anchor)throw Error('Missing appendix slide');
const slide=p.resolve(anchor.id);
await fs.writeFile(path.join(BUILD,'template-structure.json'),JSON.stringify({
  masters:p.masters.items.map(m=>({id:m.id,name:m.name})),
  layouts:p.layouts.items.map(l=>({id:l.id,name:l.name,placeholders:l.placeholders.summary()})),
},null,2));
await fs.writeFile(path.join(BUILD,'before.layout.json'),await (await slide.export({format:'layout'})).text());

const title=records.find(r=>r.slide===22&&r.kind==='textbox'&&r.text.startsWith('Решение Даниила:'));
if(!title)throw Error('Source slide differs from expected reference');
p.resolve(title.id).text.replace(title.text,'Дневной резерв: биржа без завтрашнего ЦБ');
const removed=new Set();
for(const r of records.filter(r=>r.slide===22&&r.bbox&&r.bbox[1]>=105)){
  const e=p.resolve(r.id);
  if(!removed.has(e.id)){slide.elements.deleteById(e.id);removed.add(e.id);}
}

function csv(raw){const [h,...rows]=raw.trim().split(/\r?\n/);const keys=h.split(',');return rows.map(r=>Object.fromEntries(r.split(',').map((v,i)=>[keys[i],v])));}
const metricsPath='results/research/round7/audit/breakdown_h5.csv';
const m=csv(await fs.readFile(path.join(ROOT,metricsPath),'utf8')).find(r=>r.candidate==='incumbent'&&r.kind==='all'&&r.group==='all');
const horizons=csv(await fs.readFile(path.join(ROOT,'results/research/round7/direct_pairs/later_by_horizon.csv'),'utf8'));
const h5=horizons.find(r=>r.candidate==='incumbent'&&r.period==='2025-2026'&&r.horizon==='5');
if(+m.n_signals!==506||Math.abs(+m.hit_rate/+m.base_rate-(+m.pooled_lift))>1e-12)throw Error('Metric mismatch');
const fmt=(value,n)=>Number(value).toFixed(n).replace('.',',');
const C={ink:'#1A2530',paper:'#F4F6F8',green:'#16674F',muted:'#4E5966',grey:'#6E7A88',lightAmber:'#E0A75E'};
const box=(x,y,w,h)=>({left:x*96,top:y*96,width:w*96,height:h*96});
function text(value,x,y,w,h,size=13.5,color=C.muted,opts={}){
 const t=slide.shapes.add({geometry:'textbox',position:box(x,y,w,h),fill:'none',line:{fill:'none',width:0}});
 t.text=value;t.text.style={typeface:'Calibri',fontSize:size*4/3,color,autoFit:'none',verticalAlignment:'top',insets:{top:0,bottom:0,left:0,right:0},...opts};return t;
}
function panel(x,y,w,h,fill=C.paper){slide.shapes.add({geometry:'roundRect',position:box(x,y,w,h),fill,line:{fill:'none',width:0},borderRadius:5.76});}
text('Срез 15:30. Горизонт: пять следующих публикаций. Пять валют, 10.01.2025–26.08.2026.',.7,1.2,11.9,.5);
panel(.7,1.85,5.87,3.22);
text(fmt(m.pooled_lift,3)+' lift',.98,2.1,5.31,.67,29,C.green,{typeface:'Cambria',bold:true});
text(fmt(+m.hit_rate*100,2)+' % попаданий против '+fmt(+m.base_rate*100,2)+' %\nу случайного дня.',.98,3.04,5.31,.76,17,C.ink);
text(m.n_signals+' сигналов, '+fmt(m.frequency,2)+' на валюту в неделю.',.98,4.25,5.31,.52,14);
panel(6.73,1.85,5.87,3.22);
text('Биржа задаёт дневной сигнал',7.01,2.09,5.31,.54,16,C.green,{typeface:'Cambria',bold:true});
text('Средняя CNY/RUB с 10:00 до 15:30 относительно действующего курса ЦБ.\n\nРанг по прошлым дням выделяет сильные сигналы. Если свечей нет, включается резерв на истории.',7.01,2.94,5.31,1.8,14);
panel(.7,5.38,11.9,1.27,C.ink);
text('Как читаем этот результат',1,5.55,11.3,.36,15.5,C.lightAmber,{typeface:'Cambria',bold:true});
text('Курс ЦБ на завтра не используем. Попадания считаем от сегодняшнего ЦБ, а не от цены перевода в банке.',1,6.02,11.3,.45,12.5,'#C7CFD8');
text('Открытая ретроспектива. Отдельно проверенный дневной вариант. Это не качество любого часа и не новый закрытый тест.',.7,6.9,11.9,.36,10.5,C.grey,{italic:true});
slide.speakerNotes.textFrame.setText([
 'Источник: https://github.com/tarma3v/international_transfers_signals/blob/ivan-experiments/'+metricsPath,
 'Границы периода: results/research/round7/audit/period.json. 2025-01-10..2026-08-26, 2020 строк валюта-дата. Коридоры AMD, KGS, KZT, TJS, UZS.',
 'Сохранённая политика incumbent, CNY availability-router. h=5 следующих публикаций. Hit = текущий действующий курс ЦБ не выше минимума следующих пяти курсов. Pooled lift = '+m.hit_rate+' / '+m.base_rate+' = '+m.pooled_lift+'. 279 попаданий / 506 сигналов. Частота '+m.frequency+'.',
 'Lift с поправкой на состав валют и периодов '+h5.case_lift+' отличается от pooled lift способом агрегации. Симметричная выгода '+h5.symmetric_benefit_bps+' б.п., future-only '+h5.future_benefit_bps+' б.п. Обе оценки от действующего ЦБ, не банковская экономия.',
 'Это не модель совсем без ЦБ: исторический и действующий ЦБ остаются входами. Исключён следующий, ещё не опубликованный курс. Завершённые свечи CNY/RUB, геометрическая средняя к CBR и ранг на предыдущих 250 наблюдениях. Причинный порог по предыдущим 20 оценкам, верхние 22%. При отсутствии сессии noon fallback. Общая цифра включает дни с биржей и резервные дни.',
 'Политика редких дневных сигналов отличается от калибровки вероятностей T5 на следующем слайде. Основной итоговый поток AP37 после получения ЦБ не изменён. Дневной вариант показан как проверенный резерв, не как заново объединённая политика отправки.',
 'Период неоднократно использовался в исследованиях, новый независимый тест нужен. Публикационная задержка бесплатного биржевого источника не сертифицирована, в пилоте нужен фактический as-of журнал.',
 'Прежний слайд о version_b заменён для ясности финальной защиты. Исследование и авторство Даниила сохранены в research/version_b_honest_audit.py и предыдущем файле презентации.',
].join('\n'));
await fs.writeFile(path.join(BUILD,'after.layout.json'),await (await slide.export({format:'layout'})).text());
const after=await p.inspect({kind:'slide,textbox,shape,image,table,chart,notes,layout',maxChars:1000000});
await fs.writeFile(path.join(BUILD,'after.ndjson'),after.ndjson);
await (await PresentationFile.exportPptx(p)).save(path.join(BUILD,'candidate.pptx'));
// Import/export drops source chart workbook relationships in unrelated slides.
// Preserve original workbooks and every unchanged OOXML part, never resnapshot.
console.log(execFileSync(path.join(RT,'python/bin/python3'),[
 path.join(ROOT,'research/preserve_unchanged_deck_parts.py'),source,
 path.join(BUILD,'candidate.pptx'),path.join(BUILD,'preserved.pptx'),
],{encoding:'utf8'}));
process.env.RUNTIME_NODE=path.join(RT,'node/bin/node');
process.env.RUNTIME_NODE_MODULES=path.join(RT,'node/node_modules');
process.env.RUNTIME_BIN_DIR=path.join(RT,'bin/override');
console.log(await finalizePresentation({
 workspaceDir:ROOT,candidatePath:path.join(BUILD,'preserved.pptx'),finalPath:final,
 pythonExecutable:path.join(RT,'python/bin/python3'),
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 explicitTotalSlideCount:27,requiredNativeChartOwnerSlides:[9,21,26],requiredNativeTableOwnerSlides:[13,14,20],
 sourceTemplatePath:source,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[13,14,20].flatMap(n=>['--require-native-table-slide',String(n)])],
 fontPolicy:{basis:'reference',families:['Cambria','Calibri'],referencePath:source,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex')},
 verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation.json'),
}));
