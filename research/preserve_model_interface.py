"""Transplant selected Artifact Tool edits without resaving original evidence.

Logical slide order is resolved through relationships; the source was reordered.
Only selected shapes, one reused picture, and series *names* may change.
"""
import copy
import io
import json
import posixpath
import sys
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E

P='http://schemas.openxmlformats.org/presentationml/2006/main'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG='http://schemas.openxmlformats.org/package/2006/relationships'
C='http://schemas.openxmlformats.org/drawingml/2006/chart'
NS={'p':P,'a':A,'r':R,'c':C}

def xml(b): return E.fromstring(b)
def out(x): return E.tostring(x,encoding='utf-8',xml_declaration=True)
def relpart(p): return posixpath.join(posixpath.dirname(p),'_rels',posixpath.basename(p)+'.rels')
def target(p,t): return t.lstrip('/') if t.startswith('/') else posixpath.normpath(posixpath.join(posixpath.dirname(p),t))
def order(z):
    rels={r.get('Id'):target('ppt/presentation.xml',r.get('Target')) for r in xml(z.read('ppt/_rels/presentation.xml.rels'))}
    return [rels[e.get('{'+R+'}id')] for e in xml(z.read('ppt/presentation.xml')).findall('p:sldIdLst/p:sldId',NS)]
def shapes(x): return x.findall('p:cSld/p:spTree/p:sp',NS)

def main():
    source,candidate,output,manifest=map(Path,sys.argv[1:])
    assert output.resolve() not in {source.resolve(),candidate.resolve()}
    edits=json.loads(manifest.read_text()); updates={}
    with ZipFile(source) as src,ZipFile(candidate) as draft:
        olds,news=order(src),order(draft)
        assert len(olds)==len(news)==28
        for raw_n,indices in edits['shapeIndices'].items():
            n=int(raw_n); part=olds[n-1]
            original=xml(src.read(part)); edited=xml(draft.read(news[n-1]))
            os,es=shapes(original),shapes(edited)
            assert len(os)==len(es),(n,len(os),len(es))
            for i in indices:
                replacement=copy.deepcopy(es[i])
                assert not any(k.startswith('{'+R+'}') for e in replacement.iter() for k in e.attrib)
                replacement.find('p:nvSpPr/p:cNvPr',NS).set('id',os[i].find('p:nvSpPr/p:cNvPr',NS).get('id'))
                os[i].getparent().replace(os[i],replacement)
            updates[part]=out(original)
        n=edits['newImage']['slide']; part=olds[n-1]
        original=xml(updates.get(part,src.read(part))); edited=xml(draft.read(news[n-1]))
        pics=edited.findall('p:cSld/p:spTree/p:pic',NS)
        assert len(pics)==1 and not original.findall('p:cSld/p:spTree/p:pic',NS)
        pic=copy.deepcopy(pics[0]); rid='rIdTemperatureClientExample'
        old_embed=pic.find('p:blipFill/a:blip',NS).get('{'+R+'}embed')
        draft_rels={r.get('Id'):target(news[n-1],r.get('Target')) for r in xml(draft.read(relpart(news[n-1])))}
        assert draft.read(draft_rels[old_embed])==src.read(edits['newImage']['sourcePart'])
        pic.find('p:blipFill/a:blip',NS).set('{'+R+'}embed',rid)
        nextid=max(int(e.get('id')) for e in original.findall('.//p:cNvPr',NS))+1
        pic.find('p:nvPicPr/p:cNvPr',NS).set('id',str(nextid))
        original.find('p:cSld/p:spTree',NS).append(pic)
        rels=xml(src.read(relpart(part))); assert all(r.get('Id')!=rid for r in rels)
        E.SubElement(rels,'{'+PKG+'}Relationship',Id=rid,Type=R+'/image',Target='/'+edits['newImage']['sourcePart'])
        updates[part]=out(original);updates[relpart(part)]=out(rels)

        # Names only: retain all chart values, formulas, formatting and workbook data.
        label_map={e['old']:e['new'] for e in edits['chartLabels']}
        changed_charts=0
        for part in src.namelist():
            if '/charts/' in part and part.endswith('.xml') and '/_rels/' not in part:
                x=xml(src.read(part)); changed=False
                before=x.xpath('//c:numCache//c:v/text()',namespaces=NS)
                for v in x.xpath('//c:ser/c:tx//c:v',namespaces=NS):
                    if v.text in label_map: v.text=label_map[v.text];changed=True
                if changed:
                    assert before==x.xpath('//c:numCache//c:v/text()',namespaces=NS)
                    updates[part]=out(x);changed_charts+=1
            if part.startswith('ppt/embeddings/') and part.endswith('.xlsx'):
                buf=io.BytesIO(); changed=False
                with ZipFile(io.BytesIO(src.read(part))) as oldbook,ZipFile(buf,'w') as book:
                    for item in oldbook.infolist():
                        b=oldbook.read(item.filename)
                        if item.filename.endswith('.xml'):
                            x=xml(b)
                            for t in x.xpath('//*[local-name()="t"]'):
                                if t.text in label_map:t.text=label_map[t.text];changed=True
                            b=out(x) if changed else b
                        book.writestr(item,b)
                if changed:updates[part]=buf.getvalue()
        assert changed_charts==len(edits['chartLabels'])==3
        with ZipFile(output,'w') as dst:
            for entry in src.infolist():dst.writestr(entry,updates.get(entry.filename,src.read(entry.filename)))
    with ZipFile(source) as src,ZipFile(output) as dst:
        assert set(src.namelist())==set(dst.namelist()) and dst.testzip() is None
        assert order(src)==order(dst)
        for f in src.namelist():
            if '/media/' in f or f not in updates:assert src.read(f)==dst.read(f),f
        changed=[f for f in src.namelist() if src.read(f)!=dst.read(f)]
    print(json.dumps({'changed_parts':changed,'slides':28,'original_images_preserved':True,'chart_numbers_preserved':True},ensure_ascii=False))

if __name__=='__main__':main()
