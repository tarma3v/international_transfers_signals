/** Focused update of the approved deck: understandable names and real UI. */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';

const ROOT=path.resolve(new URL('..',import.meta.url).pathname);
const RT=process.env.ITMO_RUNTIME||'/Users/jeck5iv/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const SKILL='/Users/jeck5iv/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const BUILD=path.join(ROOT,'tmp/model_interface');
const FINAL=process.env.ITMO_FINAL_PPTX||path.join(ROOT,'submission/prezentaciya-finalnaya-2026-09-07-model-i-interfeys.pptx');
await fs.mkdir(BUILD,{recursive:true});
await fs.mkdir(path.dirname(FINAL),{recursive:true});
const source=path.join(BUILD,'source.pptx');
await fs.writeFile(source,execFileSync('git',['show','66cd033:submission/prezentaciya-finalnaya-2026-09-07-reshenie.pptx'],{cwd:ROOT,maxBuffer:20*1024*1024}));
const {FileBlob,PresentationFile}=await import(pathToFileURL(path.join(RT,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')));
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')));
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const before=await p.inspect({kind:'slide,shape,textbox,image,chart,table',maxChars:1000000});
await fs.writeFile(path.join(BUILD,'before.ndjson'),before.ndjson);
const rows=before.ndjson.split('\n').filter(Boolean).map(JSON.parse),changes={};
function objs(n){return rows.filter(r=>r.slide===n&&['shape','textbox'].includes(r.kind));}
function row(id){const r=rows.find(r=>r.id===id);if(!r)throw Error(id);return r;}
function mark(id){const r=row(id),i=objs(r.slide).findIndex(x=>x.id===id);if(i<0)throw Error(id);(changes[r.slide]??=[]).push(i);return p.resolve(id);}
function text(id,value){mark(id).text=value;}
function pos(id,patch){const s=mark(id);s.position={...s.position,...patch};}
function slide(n){return p.resolve(rows.find(r=>r.kind==='slide'&&r.slide===n).id);}

// Match the badge to the journey navigation, not the physical PDF page.
for(const [n,label] of [[3,1],[4,2],[5,3],[6,4],...[7,8,9,10,11,12,13].map(n=>[n,5]),[14,6],[15,7],[16,8],[17,9],[18,10]]){
 const badge=objs(n).find(r=>r.kind==='textbox'&&r.bbox[0]<100&&r.bbox[1]<100);
 text(badge.id,String(label));
}
// Explain the name without changing the frozen model or any metric.
for(const r of rows.filter(r=>r.kind==='textbox'&&r.text?.includes('AP37'))){
 text(r.id,r.text.replaceAll('Итоговый AP37','Итоговый ансамбль').replaceAll('У AP37','У ансамбля').replaceAll('Фильтр AP37','Отбор сигналов').replaceAll('AP37','Ансамбль'));
}
text('sh/gv29g7q5','Ансамбль ограничивает поток двумя сигналами за календарную неделю на валюту. Основной отбор дополняется резервом при долгой тишине.');
text('sh/xknyx4ja','В ретроспективе — 1,04 сигнала на валюту в неделю. Общий лимит на клиента, время отправки и разрешения подключает банк в пилоте.');
text('sh/atcjidg7','Когда отправляем пуш');
text('sh/vu5kbihc','В пилоте: после получения нового курса ЦБ, если ансамбль дал сигнал и пройдены клиентские фильтры. Не автоматически в 18:00.');
text('sh/zaxs3yhc','Известное изменение ЦБ на завтра и исторические модели.\n\nCatBoost оценивает температуру. Ансамбль отдельно отбирает пуши.');
text('sh/tsnip0ny','После получения курса: завтра не дешевле, есть сигнал ансамбля, пройдены фильтры банка. При переносе на утро считаем заново.');
text('sh/lofadsni','Решение: вечерний ансамбль и отбор сигналов');
text('sh/utobax4f','Пуш: ансамбль с отбором сигналов');
text('sh/vuxsj250','Бустинги, включая CatBoost, и линейная модель риска. Основной отбор дополняется резервом при долгой тишине.\n\nПрошлые завершённые случаи помогают оценить качество резерва по валюте. Максимум два события за календарную неделю.\n\nПуш — не просто «температура выше 70».');
text('sh/kvyxo7qp','Температура и отбор пушей — разные части системы. Метрики по ЦБ не доказывают возможность перевода по старой цене банка.');
text('sh/hcvy14ne','Ансамбль с отбором сигналов, h=5, 09.01.2024–25.08.2026. Открытая ретроспектива после получения нового курса ЦБ.');

// Reuse the team's original screenshot, without redrawing or changing pixels.
pos('sh/zutgvm94',{width:686.4});
pos('sh/kv2h4rqp',{width:632.64});
pos('sh/lwbyxwra',{width:632.64});
mark('sh/ip4zel83').fill='none';
text('sh/3qdg7qpo','Что видит клиент');
pos('sh/3qdg7qpo',{left:817.92,top:166.08,width:350.4,height:38.4});
text('sh/fm1gzq5o','Сумма и сравнение с историей');
pos('sh/fm1gzq5o',{left:807.36,top:632.64,width:390.72,height:25.92});
pos('sh/e1sf65o3',{width:686.4});
text('sh/1ojy10ne','Понятный факт вместо прогноза');
pos('sh/1ojy10ne',{width:630.72});
text('sh/gnax8v6t','«Лучшая за три месяца была 56 дней из девяноста».\nНа экране — сумма получателя и сравнение с прошлым. Температуру и обещание будущего не показываем.');
pos('sh/gnax8v6t',{width:630.72,height:71.04});
text('sh/nqlg3a5k','Макет команды: числа иллюстративные, по ЦБ. В пилоте используем актуальную котировку банка.');
const phone=execFileSync(path.join(RT,'python/bin/python3'),['-c',"from zipfile import ZipFile; import sys; sys.stdout.buffer.write(ZipFile(sys.argv[1]).read('ppt/media/image3.png'))",source]);
slide(11).images.add({blob:new Uint8Array(phone),contentType:'image/png',alt:'Исходный макет команды: сумма перевода и сравнение с прошлыми 90 днями',fit:'contain',position:{left:870.72,top:205.44,width:207.36,height:423.02}});

// Expand Ivan's role while retaining the colleagues' original text and font.
text('sh/vy1sj694','AI Engineer. Разработка итоговой модели: дневной и вечерний прогноз, температура.\n\n100+ вариантов моделей, признаков и правил отбора. Эксперименты по валютам и периодам, контроль утечек.\n\nОтбор пушей, итоговые метрики и финализация презентации.');
for(const id of ['sh/29wby1wf','sh/fm5snmx4','sh/h0jalgra'])pos(id,{height:292.8});
pos('sh/vy1sj694',{height:198.72});
pos('sh/8ba98bqt',{top:456,height:192});
pos('sh/9cjahgry',{top:472.32});
pos('sh/m98r618n',{top:511.68,height:120.96});

const chartLabels=[];
for(const r of rows.filter(r=>r.kind==='chart')){
 const chart=p.resolve(r.id);
 for(let i=0;i<chart.series.items.length;i++){
  const s=chart.series.getItemAt(i);
  if(s.name.includes('AP37')){chartLabels.push({slide:r.slide,series:i,old:s.name,new:s.name.replaceAll('AP37','ансамбля')});s.name=s.name.replaceAll('AP37','ансамбля');}
 }
}
const candidate=path.join(BUILD,'candidate.pptx'),preserved=path.join(BUILD,'preserved.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
const manifest={shapeIndices:Object.fromEntries(Object.entries(changes).map(([n,v])=>[n,[...new Set(v)]])),newImage:{slide:11,sourcePart:'ppt/media/image3.png'},chartLabels};
await fs.writeFile(path.join(BUILD,'edits.json'),JSON.stringify(manifest,null,2));
console.log(execFileSync(path.join(RT,'python/bin/python3'),[path.join(ROOT,'research/preserve_model_interface.py'),source,candidate,preserved,path.join(BUILD,'edits.json')],{encoding:'utf8'}));
await fs.writeFile(path.join(BUILD,'after.ndjson'),(await p.inspect({kind:'slide,shape,textbox,image,chart,table',maxChars:1000000})).ndjson);
process.env.RUNTIME_NODE=path.join(RT,'node/bin/node');process.env.RUNTIME_NODE_MODULES=path.join(RT,'node/node_modules');process.env.RUNTIME_BIN_DIR=path.join(RT,'bin/override');
const tables=[16,17,22,28],charts=[13,23,27];
console.log(await finalizePresentation({workspaceDir:ROOT,candidatePath:preserved,finalPath:FINAL,pythonExecutable:path.join(RT,'python/bin/python3'),integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),explicitTotalSlideCount:28,requiredNativeChartOwnerSlides:charts,requiredNativeTableOwnerSlides:tables,requiredEmbeddedWorkbookChartOwnerSlides:charts,sourceTemplatePath:source,layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],fontPolicy:{basis:'reference',families:['Cambria','Calibri'],referencePath:source,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,path.basename(FINAL)+'.validation.json')}));
