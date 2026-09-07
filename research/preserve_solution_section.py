"""Install selected Artifact Tool edits and reorder slides, preserving evidence.

No chart, workbook, image data, source relationships or unselected shape changes.
The approved navigation is copied from slide 7 so its original formatting stays
exactly intact instead of depending on the importer's rendering of small labels.
"""
import copy
import json
import sys
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as ET

P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NS = {'p': P, 'a': A}


def tree(xml):
    return xml.find('p:cSld/p:spTree', NS)


def shapes(xml):
    return tree(xml).findall('p:sp', NS)


def serialize(xml):
    return ET.tostring(xml, encoding='utf-8', xml_declaration=True)


def main():
    source, candidate, output, manifest = map(Path, sys.argv[1:])
    if output.resolve() in {source.resolve(), candidate.resolve()}:
        raise ValueError('Output must not overwrite either source')
    edits = json.loads(manifest.read_text())
    updates = {}
    with ZipFile(source) as src, ZipFile(candidate) as draft:
        nav_source = ET.fromstring(src.read('ppt/slides/slide7.xml'))
        navigation = shapes(nav_source)[3:20]
        assert len(navigation) == 17
        for raw_n, indices in edits['shapeIndices'].items():
            n = int(raw_n)
            part = f'ppt/slides/slide{n}.xml'
            original = ET.fromstring(src.read(part))
            edited = ET.fromstring(draft.read(part))
            old_shapes, new_shapes = shapes(original), shapes(edited)
            extra = 17 if n in edits['solutions'] else 0
            assert len(new_shapes) == len(old_shapes) + extra, (n, len(old_shapes), len(new_shapes))
            for i in indices:
                replacement = copy.deepcopy(new_shapes[i])
                assert not any(k.startswith('{' + R + '}') for e in replacement.iter() for k in e.attrib)
                # Existing IDs are retained for any references from the original package.
                new_properties = replacement.find('p:nvSpPr/p:cNvPr', NS)
                old_properties = old_shapes[i].find('p:nvSpPr/p:cNvPr', NS)
                new_properties.set('id', old_properties.get('id'))
                tree(original).replace(old_shapes[i], replacement)
            if n in edits['imageFrameSlides']:
                old_pics = tree(original).findall('p:pic', NS)
                new_pics = tree(edited).findall('p:pic', NS)
                assert len(old_pics) == len(new_pics) == 1
                old_xfrm = old_pics[0].find('p:spPr/a:xfrm', NS)
                new_xfrm = new_pics[0].find('p:spPr/a:xfrm', NS)
                old_xfrm.getparent().replace(old_xfrm, copy.deepcopy(new_xfrm))
            if n in edits['solutions']:
                last_id = max(int(e.get('id')) for e in original.findall('.//p:cNvPr', NS))
                for i, nav_shape in enumerate(navigation):
                    replacement = copy.deepcopy(nav_shape)
                    props = replacement.find('p:nvSpPr/p:cNvPr', NS)
                    props.set('id', str(last_id + i + 1))
                    props.set('name', f'Solution navigation {i + 1}')
                    # Remove only per-object creation GUIDs, not their appearance.
                    for ext in props.findall('a:extLst', NS):
                        props.remove(ext)
                    tree(original).append(replacement)
            updates[part] = serialize(original)
        presentation = ET.fromstring(src.read('ppt/presentation.xml'))
        ids = presentation.find('p:sldIdLst', NS)
        old_ids = list(ids)
        assert sorted(edits['order']) == list(range(1, len(old_ids) + 1))
        for child in old_ids:
            ids.remove(child)
        for n in edits['order']:
            ids.append(old_ids[n - 1])
        updates['ppt/presentation.xml'] = serialize(presentation)
        with ZipFile(output, 'w') as result:
            for entry in src.infolist():
                result.writestr(entry, updates.get(entry.filename, src.read(entry.filename)))
    with ZipFile(source) as src, ZipFile(output) as result:
        assert set(src.namelist()) == set(result.namelist())
        assert result.testzip() is None
        changed = [part for part in src.namelist() if src.read(part) != result.read(part)]
        assert set(changed) == set(updates)
        evidence = [part for part in src.namelist() if part.startswith(('ppt/media/', 'ppt/charts/', 'ppt/embeddings/'))]
        assert all(src.read(part) == result.read(part) for part in evidence)
    print(json.dumps({'changed_parts': changed, 'order': edits['order'], 'evidence_parts_preserved': len(evidence)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
