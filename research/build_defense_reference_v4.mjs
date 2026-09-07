/** Current results in the exact style of the user's final (4) reference. */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
const ROOT=path.resolve(new URL('..',import.meta.url).pathname);
const BUILD=path.join(ROOT,'tmp/reference_v4/build');
const RT='/Users/jeck5iv/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const SKILL='/Users/jeck5iv/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const source=path.join(ROOT,'submission/prezentaciya-itogovaya-2026-09-07-birzha.pptx');
const final=process.env.ITMO_FINAL_PPTX||path.join(ROOT,'submission/prezentaciya-finalnaya-2026-09-07-obnovlennaya.pptx');
await fs.mkdir(BUILD,{recursive:true});
const referencePdf=execFileSync('git',['show','3263473:submission/prezentaciya-finalnaya.pdf'],{cwd:ROOT,maxBuffer:30*1024*1024});
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
if(sha(referencePdf)!=='98a7184c008faec3ac97d92640b112f614082bbafc7450cf3d063fd65151e26f')throw Error('Reference PDF mismatch');
const reference=path.join(BUILD,'reference.pptx');
await fs.writeFile(reference,execFileSync('git',['show','3263473:submission/prezentaciya-finalnaya.pptx'],{cwd:ROOT,maxBuffer:30*1024*1024}));
const {FileBlob,PresentationFile}=await import(pathToFileURL(path.join(RT,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')));
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')));
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snapshot=await p.inspect({kind:'slide,textbox,shape,table,chart,image,notes,layout',maxChars:1000000});
await fs.writeFile(path.join(BUILD,'before.ndjson'),snapshot.ndjson);
const rows=snapshot.ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x));
const cover=p.resolve(rows.find(r=>r.kind==='slide'&&r.slide===1).id);
const C={ink:'#1A2530',white:'#FFFFFF',paper:'#F4F6F8',green:'#16674F',amber:'#9C5B12',lightGreen:'#6FBFA1',lightAmber:'#E0A75E',muted:'#4E5966',grey:'#6E7A88'};
const box=(x,y,w,h)=>({left:x*96,top:y*96,width:w*96,height:h*96});
function text(s,value,x,y,w,h,size=13.5,color=C.muted,opts={}){
 const t=s.shapes.add({geometry:'textbox',position:box(x,y,w,h),fill:'none',line:{fill:'none',width:0}});
 t.text=value;t.text.style={typeface:'Calibri',fontSize:size*4/3,color,autoFit:'none',verticalAlignment:'top',insets:{top:0,bottom:0,left:0,right:0},...opts};return t;
}
function panel(s,x,y,w,h,fill=C.paper){s.shapes.add({geometry:'roundRect',position:box(x,y,w,h),fill,line:{fill:'none',width:0},borderRadius:5.76});}
function csv(raw){const [h,...rs]=raw.trim().split(/\r?\n/);const keys=h.split(',');return rs.map(r=>Object.fromEntries(r.split(',').map((v,i)=>[keys[i],v])));}
const readCSV=async q=>csv(await fs.readFile(path.join(ROOT,q),'utf8'));
const ap=(await readCSV('results/research/after_publication/ap37_effective/retrospective_all_horizons.csv')).find(r=>r.candidate==='ap26_core_mature_precision_calendar_fallback_cap2'&&r.h==='5');
const fmt=(n,d)=>Number(n).toFixed(d).replace('.',',');
if(Math.abs(+ap.hit_rate/+ap.base_rate-(+ap.pooled_lift))>1e-12)throw Error('AP37 ratio mismatch');
const coverMetrics=[
 [fmt(ap.pooled_lift,3)+' lift','73,38 % удачных сигналов',C.lightGreen],
 ['+'+fmt(ap.symmetric_bps,2)+' б.п.','Выгода момента в окне ±5',C.white],
 [fmt(ap.frequency,2)+' / нед.','Сигналов на валюту, в среднем',C.lightAmber],
];
for(let i=0;i<coverMetrics.length;i++){
 const [value,label,color]=coverMetrics[i],x=.7+i*4.08;
 text(cover,value,x,5.53,3.75,.61,28,color,{typeface:'Cambria',bold:true});
 text(cover,label,x,6.19,3.75,.5,12.5,'#C7CFD8');
}
text(cover,'h=5. 09.01.2024–25.08.2026. После получения ЦБ, от действующего курса. Открытая ретроспектива, не банковская экономия.',.7,6.94,11.9,.3,10,'#AAB6C3');
cover.speakerNotes.textFrame.setText('Итоговая оценка AP37: '+JSON.stringify(ap)+'\nИсточник: results/research/after_publication/ap37_effective/retrospective_all_horizons.csv. Пять валют, 695 сигналов. Первый следующий курс уже опубликован. Известный первый шаг помогает оценке. Это метрика относительно действующего сегодня официального курса ЦБ, не доказанная экономия на банковской котировке. Параметры причинные, но открытая история многократно использовалась в исследовании.');

// Appendix: concrete like-for-like experiment evidence, separate from evening AP37.
const s=p.slides.add();s.setLayout(p.layouts.items[0]);s.background.fill=C.white;
s.shapes.add({geometry:'ellipse',position:box(.7,.52,.42,.42),fill:C.grey,line:{fill:'none',width:0}});
text(s,'П10',.7,.65,.42,.22,9,C.white,{alignment:'center',bold:true});
text(s,'Эксперименты: что оставили в итоговом решении',1.28,.44,11.3,.62,27,C.ink,{typeface:'Cambria',bold:true});
text(s,'Дневной срез 15:30, h=5. Одна опора: действующий курс ЦБ. Открытая ретроспектива 2025–2026.',.7,1.2,11.9,.48);
const direct=await readCSV('results/research/round7/direct_pairs/later_by_horizon.csv');
const residual=await readCSV('results/research/round7/residual_floor/later_by_horizon.csv');
const pick=(rs,name)=>{const r=rs.find(r=>r.candidate===name&&r.period==='2025-2026'&&r.horizon==='5');if(!r)throw Error('Missing experiment '+name);return fmt(r.case_lift,3);};
text(s,'Проверка прямых валютных пар',.7,1.93,7.0,.48,17,C.green,{typeface:'Cambria',bold:true});
const values=[['Вариант','Lift*'],
 ['CNY/RUB с резервом на истории',pick(direct,'incumbent')],
 ['Добавка прямой пары 25 %',pick(direct,'last_basis_soft_w025')],
 ['Общая модель поправки к CNY',pick(residual,'global_quantile_w10')],
 ['Индивидуальные веса валют',pick(direct,'per_currency_weights')]];
const t=s.tables.add({rows:values.length,columns:2,left:.7*96,top:2.56*96,width:7.05*96,height:2.34*96,values,columnWidths:[5.5*96,1.55*96]});
t.borders.assign({style:'solid',fill:'#DCE2E8',width:.65});
for(let r=0;r<values.length;r++)for(let c=0;c<2;c++){const cell=t.getCell(r,c);cell.fill=r===0?C.paper:C.white;cell.text.style={typeface:'Calibri',fontSize:14*4/3,color:r===1?C.green:C.muted,bold:r<=1};}
panel(s,8.08,1.9,4.52,3.1);
text(s,'Что ещё проверили',8.36,2.1,3.96,.48,16,C.ink,{typeface:'Cambria',bold:true});
text(s,'ETS, SARIMA и нейросети.\n\nМодели по одной валюте, общий бустинг поправок и смеси по режимам.\n\nПроверяли перенос между годами и частоту сигналов.',8.36,2.9,3.96,1.85,13);
panel(s,.7,5.34,11.9,1.3,C.ink);
text(s,'Что вошло в итог',1,5.52,11.3,.34,15.5,C.lightAmber,{typeface:'Cambria',bold:true});
text(s,'Днём: доступные биржевые данные и резерв. После получения ЦБ: модели оставшегося будущего. Температура и редкие пуши решают разные задачи.',1,5.98,11.3,.48,12.5,'#C7CFD8');
text(s,'* Lift с поправкой на состав валют и периодов. У CNY 2,053 здесь и 2,059 на слайде 22: две агрегации одних сигналов.',.7,6.92,11.9,.34,10.5,C.grey,{italic:true});
s.speakerNotes.textFrame.setText('Таблица: results/research/round7/direct_pairs/later_by_horizon.csv и results/research/round7/residual_floor/later_by_horizon.csv. '+JSON.stringify(values)+'\nПолитики выбраны на очищенном 2024 и оценены на уже изученной ретроспективе 2025–2026. Первый вариант incumbent, второй last_basis_soft_w025, третий global_quantile_w10, четвёртый per_currency_weights. Поправка к CNY обучает квантиль будущего минимума глобально по валютам, добавляется с весом 10%.\nСписок других семейств: EXPERIMENTS_SUMMARY.md, разделы «Что не дало устойчивого выигрыша» и AP34-E/AP36-E. Их цифры не смешиваются с этой таблицей из-за разных периодов и целей. Таблица не доказывает, что все ансамбли хуже: она показывает конкретную проверку прямых пар. Свежие источники, калибровка и отбор по созревшим ответам входят в итоговый контракт model/final_temperature_model_v1.json. Все оценки ЦБ являются исследовательскими, не исполнимой котировкой.');

await fs.writeFile(path.join(BUILD,'after.ndjson'),(await p.inspect({kind:'slide,textbox,table,chart,image,notes',maxChars:1000000})).ndjson);
await (await PresentationFile.exportPptx(p)).save(path.join(BUILD,'candidate.pptx'));
console.log(execFileSync(path.join(RT,'python/bin/python3'),[path.join(ROOT,'research/preserve_unchanged_deck_parts.py'),source,path.join(BUILD,'candidate.pptx'),path.join(BUILD,'preserved.pptx'),'--replace-slides','1','--append-slides','28','--table-style-reference',reference],{encoding:'utf8'}));
process.env.RUNTIME_NODE=path.join(RT,'node/bin/node');process.env.RUNTIME_NODE_MODULES=path.join(RT,'node/node_modules');process.env.RUNTIME_BIN_DIR=path.join(RT,'bin/override');
console.log(await finalizePresentation({
 workspaceDir:ROOT,candidatePath:path.join(BUILD,'preserved.pptx'),finalPath:final,
 pythonExecutable:path.join(RT,'python/bin/python3'),
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 explicitTotalSlideCount:28,requiredNativeChartOwnerSlides:[9,21,26],
 requiredNativeTableOwnerSlides:[13,14,20,28],sourceTemplatePath:reference,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[13,14,20,28].flatMap(n=>['--require-native-table-slide',String(n)])],
 fontPolicy:{basis:'reference',families:['Cambria','Calibri'],referencePath:reference,referenceSha256:sha(await fs.readFile(reference))},
 verifyArtifactToolImport:true,receiptPath:path.join(BUILD,path.basename(final)+'.validation.json'),
}));
