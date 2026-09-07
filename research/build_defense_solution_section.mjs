/** Group the existing model slides into step 5, preserving the approved deck. */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';

const ROOT=path.resolve(new URL('..',import.meta.url).pathname);
const RT=process.env.ITMO_RUNTIME||'/Users/jeck5iv/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const SKILL='/Users/jeck5iv/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const BUILD=path.join(ROOT,'tmp/solution_section');
const FINAL=process.env.ITMO_FINAL_PPTX||path.join(ROOT,'submission/prezentaciya-finalnaya-2026-09-07-reshenie.pptx');
await fs.mkdir(BUILD,{recursive:true});
const source=path.join(BUILD,'source.pptx');
await fs.writeFile(source,execFileSync('git',['show','a29adb1:submission/prezentaciya-finalnaya-2026-09-07-obnovlennaya.pptx'],{cwd:ROOT,maxBuffer:20*1024*1024}));
const {FileBlob,PresentationFile}=await import(pathToFileURL(path.join(RT,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')));
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')));
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const before=await p.inspect({kind:'slide,shape,textbox,image,chart,table,layout',maxChars:1000000});
await fs.writeFile(path.join(BUILD,'before.ndjson'),before.ndjson);
const rows=before.ndjson.split('\n').filter(Boolean).map(JSON.parse);
const solutions=[11,23,27,16,10,9];
const order=[1,2,3,4,5,6,7,...solutions,8,12,13,14,15,17,18,19,20,21,22,24,25,26,28];
if(order.length!==28||new Set(order).size!==28)throw Error('Invalid slide order');
const changes={};
function objects(n){return rows.filter(r=>r.slide===n&&['shape','textbox'].includes(r.kind));}
function mark(n,r){const i=objects(n).findIndex(x=>x.id===r.id);if(i<0)throw Error('Shape not found');(changes[n]??=[]).push(i);return p.resolve(r.id);}
function update(n,r,text){mark(n,r).text=text;}
function position(n,r,patch){const s=mark(n,r);s.position={...s.position,...patch};}
function header(n,title,label='2',appendix=false){
 const rs=objects(n), badge=rs.find(r=>r.kind==='textbox'&&r.bbox[0]<100&&r.bbox[1]<100);
 const shape=rs.find(r=>r.kind==='shape'&&r.bbox[0]<100&&r.bbox[1]<100);
 const head=rs.find(r=>r.kind==='textbox'&&r.bbox[0]>110&&r.bbox[1]<100);
 const b=mark(n,badge);b.text=label;
 b.position={left:67.2,top:49.92,width:40.32,height:40.32};
 b.text.style={typeface:'Calibri',fontSize:(appendix?10.5:14)*4/3,bold:true,color:'#FFFFFF',alignment:'center',verticalAlignment:'middle',autoFit:'none',insets:{top:0,bottom:0,left:0,right:0}};
 mark(n,shape).fill=appendix?'#6E7A88':'#9C5B12';
 if(title){const h=mark(n,head);h.text=title;h.text.style={typeface:'Cambria',fontSize:36,bold:true,color:'#1A2530',verticalAlignment:'middle',autoFit:'none',insets:{top:0,bottom:0,left:0,right:0}};}
}
// Exact geometry, colours and type from the approved step-5 navigation on slide 7.
function nav(n){
 const slide=p.resolve(rows.find(r=>r.kind==='slide'&&r.slide===n).id);
 const labels=['1 · Триггер','2 · Фильтр','3 · Пуш','4 · Экран перевода','5 · Решение','6 · Удержание'];
 for(let i=0;i<6;i++){
  const left=124.8+i*183.36, pos={left,top:103.68,width:168,height:30.72};
  slide.shapes.add({name:`solution-nav-bg-${i+1}`,geometry:'roundRect',position:pos,fill:i===4?'#9C5B12':i<4?'#D8DEE4':'#F4F6F8',line:{fill:'none',width:0},adjustmentList:[{name:'adj',formula:'val 31250'}]});
  const t=slide.shapes.add({name:`solution-nav-text-${i+1}`,geometry:'textbox',position:pos,fill:'none',line:{fill:'none',width:0}});
  t.text=labels[i];t.text.style={typeface:'Calibri',fontSize:14,bold:i===4,color:i===4?'#FFFFFF':i<4?'#5A6672':'#8996A3',alignment:'center',verticalAlignment:'middle',autoFit:'none',insets:{top:0,bottom:0,left:0,right:0}};
  if(i<5){const a=slide.shapes.add({name:`solution-nav-arrow-${i+1}`,geometry:'textbox',position:{left:left+168,top:103.68,width:15.36,height:30.72},fill:'none',line:{fill:'none',width:0}});a.text='›';a.text.style={typeface:'Calibri',fontSize:16,color:'#B6BEC7',alignment:'center',verticalAlignment:'middle',autoFit:'none',insets:{top:0,bottom:0,left:0,right:0}};}
 }
}
const titles={11:'Решение: модель в течение дня',23:'Решение: дневная модель на биржевых данных',27:'Решение: модель после публикации ЦБ',16:'Решение: температура внутри, факты на экране',10:'Решение: температура и курс узбекского сума',9:'Решение: 73,38 % удачных дней вместо 29,45 %'};
for(const n of solutions){header(n,titles[n]);nav(n);}

// Make room below the navigation without reducing the original body font sizes.
for(const n of [9,11]){
 const r=objects(n).find(r=>r.kind==='textbox'&&r.bbox[1]===115.2);
 position(n,r,{top:150.72,height:30.72});
}
for(const r of objects(11)){
 if(r.bbox[1]===187.2)position(11,r,{top:211.2,height:r.bbox[3]-24});
 if(r.bbox[1]===206.4)position(11,r,{top:230.4});
 if(r.bbox[1]===266.88)position(11,r,{top:290.88,height:r.bbox[3]-24});
}
for(const n of [16,27])for(const r of objects(n)){
 if(r.bbox[1]===139.2)position(n,r,{top:177.6,height:r.bbox[3]-38.4});
 if(r.bbox[1]===158.4)position(n,r,{top:196.8});
 if(r.bbox[1]===218.88)position(n,r,{top:257.28,height:r.bbox[3]-38.4});
}
for(const r of objects(23))if(r.bbox[1]>=136&&r.bbox[1]<600)position(23,r,{top:r.bbox[1]+38.4});
const im=rows.find(r=>r.slide===10&&r.kind==='image');
const picture=p.resolve(im.id), height=480, width=im.bbox[2]*height/im.bbox[3];
picture.frame={...picture.frame,left:(1280-width)/2,top:158.4,width,height};

// Renumber sections and appendix badges after moving six slides into step 5.
for(const [n,label] of [[12,'3'],[13,'4'],[14,'5'],[15,'6']])header(n,null,label);
for(const [n,label] of [[24,'П5'],[25,'П6'],[26,'П7'],[28,'П8']])header(n,null,label,true);
const note=objects(28).find(r=>r.text?.includes('на слайде 22'));
update(28,note,note.text.replace('на слайде 22','на слайде 24'));
const q=objects(17).find(r=>r.text?.startsWith('Дальше:'));
update(17,q,'Дальше: доступность данных, определения метрик, результаты по горизонтам и валютам, дневной резерв, интерфейс и эксперименты.');
const divider=objects(18).find(r=>r.text?.startsWith('Доступность данных,'));
update(18,divider,'Доступность данных, определения метрик, результаты по горизонтам и валютам, дневной резерв, интерфейс и эксперименты.');

const candidate=path.join(BUILD,'candidate.pptx'),preserved=path.join(BUILD,'preserved.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
const manifest={order,solutions,shapeIndices:Object.fromEntries(Object.entries(changes).map(([n,ids])=>[n,[...new Set(ids)]])),imageFrameSlides:[10]};
await fs.writeFile(path.join(BUILD,'edits.json'),JSON.stringify(manifest,null,2));
console.log(execFileSync(path.join(RT,'python/bin/python3'),[path.join(ROOT,'research/preserve_solution_section.py'),source,candidate,preserved,path.join(BUILD,'edits.json')],{encoding:'utf8'}));
await fs.writeFile(path.join(BUILD,'after.ndjson'),(await p.inspect({kind:'slide,shape,textbox,image,chart,table',maxChars:1000000})).ndjson);
process.env.RUNTIME_NODE=path.join(RT,'node/bin/node');process.env.RUNTIME_NODE_MODULES=path.join(RT,'node/node_modules');process.env.RUNTIME_BIN_DIR=path.join(RT,'bin/override');
const tables=[16,17,22,28],charts=[13,23,27];
console.log(await finalizePresentation({workspaceDir:ROOT,candidatePath:preserved,finalPath:FINAL,pythonExecutable:path.join(RT,'python/bin/python3'),integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),explicitTotalSlideCount:28,requiredNativeChartOwnerSlides:charts,requiredNativeTableOwnerSlides:tables,requiredEmbeddedWorkbookChartOwnerSlides:charts,sourceTemplatePath:source,layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],fontPolicy:{basis:'reference',families:['Cambria','Calibri'],referencePath:source,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,path.basename(FINAL)+'.validation.json')}));
