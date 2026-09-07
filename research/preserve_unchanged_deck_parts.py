"""Preserve untouched OOXML while installing text/table slides from Artifact Tool.

Only explicitly selected slides change. Appended slides must form a contiguous
suffix and may reference only parts already present in the resulting package.
"""
import argparse
import copy
import json
from pathlib import Path
from lxml import etree as ET
from zipfile import ZipFile

R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
P = "http://schemas.openxmlformats.org/package/2006/relationships"
S = "http://schemas.openxmlformats.org/presentationml/2006/main"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"


def slide_parts(number):
    return (f"ppt/slides/slide{number}.xml", f"ppt/notesSlides/notesSlide{number}.xml")


def rel_part(part):
    part = Path(part)
    return str(part.parent / "_rels" / (part.name + ".rels"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("edited", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replace-slides", nargs="+", type=int, default=[22])
    parser.add_argument("--append-slides", nargs="*", type=int, default=[])
    parser.add_argument("--table-style-reference", type=Path)
    args = parser.parse_args()
    if args.output.resolve() in {args.source.resolve(), args.edited.resolve()}:
        raise ValueError("Output must differ from both input decks")
    with ZipFile(args.source) as source, ZipFile(args.edited) as edited:
        border_template = []
        if args.table_style_reference:
            with ZipFile(args.table_style_reference) as reference:
                ref_xml = ET.fromstring(reference.read("ppt/slides/slide9.xml"))
                tc = ref_xml.find(".//{" + A + "}tcPr")
                border_template = [e for e in tc if ET.QName(e).localname in {"lnL", "lnR", "lnT", "lnB"}]
                if len(border_template) != 4:
                    raise ValueError("Expected four explicit reference table borders")
        parts = tuple(p for n in args.replace_slides for p in slide_parts(n))
        updates = {p: edited.read(p) for p in parts}
        additions = {}
        for part in parts:
            xml = ET.fromstring(edited.read(part))
            if any(k.startswith("{" + R + "}") for e in xml.iter() for k in e.attrib):
                raise ValueError("Inline relationship remapping would be required")
            rel = rel_part(part)
            source_targets = {(r.get("Type"), r.get("Target")) for r in ET.fromstring(source.read(rel)).findall("{" + P + "}Relationship")}
            edited_targets = {(r.get("Type"), r.get("Target")) for r in ET.fromstring(edited.read(rel)).findall("{" + P + "}Relationship")}
            if source_targets != edited_targets:
                raise ValueError("Source and edited slide relationships differ")
        if args.append_slides:
            presentation = ET.fromstring(source.read("ppt/presentation.xml"))
            ids = presentation.find("{" + S + "}sldIdLst")
            count = len(ids)
            if args.append_slides != list(range(count + 1, count + 1 + len(args.append_slides))):
                raise ValueError("Appended slides must be a contiguous suffix")
            relations = ET.fromstring(source.read("ppt/_rels/presentation.xml.rels"))
            content_types = ET.fromstring(source.read("[Content_Types].xml"))
            edited_types = ET.fromstring(edited.read("[Content_Types].xml"))
            next_id = max(int(e.get("id")) for e in ids) + 1
            for offset, n in enumerate(args.append_slides):
                rid = f"rIdItmoAppend{n}"
                if any(e.get("Id") == rid for e in relations):
                    raise ValueError("Relationship id collision")
                ET.SubElement(ids, "{" + S + "}sldId", {"id": str(next_id + offset), "{" + R + "}id": rid})
                ET.SubElement(relations, "{" + P + "}Relationship", {"Id": rid, "Type": R + "/slide", "Target": f"/ppt/slides/slide{n}.xml"})
                for part in slide_parts(n):
                    xml = ET.fromstring(edited.read(part))
                    if any(k.startswith("{" + R + "}") for e in xml.iter() for k in e.attrib):
                        raise ValueError("Appended slide contains inline relationships")
                    for member in (part, rel_part(part)):
                        if member in source.namelist():
                            raise ValueError("Appended part already exists")
                        additions[member] = edited.read(member)
                    # Restore the actual template's cell lines if export omitted them.
                    # Values, table geometry and text remain editable and unchanged.
                    cells = xml.findall(".//{" + A + "}tcPr")
                    if border_template and cells:
                        for cell in cells:
                            for child in list(cell):
                                if ET.QName(child).localname in {"lnL", "lnR", "lnT", "lnB"}:
                                    cell.remove(child)
                            for index, line in enumerate(border_template):
                                cell.insert(index, copy.deepcopy(line))
                        additions[part] = ET.tostring(xml, encoding="utf-8", xml_declaration=True)
                    override = next(e for e in edited_types if e.get("PartName") == "/" + part)
                    content_types.append(override)
            for part, tree in (("ppt/presentation.xml", presentation), ("ppt/_rels/presentation.xml.rels", relations), ("[Content_Types].xml", content_types)):
                updates[part] = ET.tostring(tree, encoding="utf-8", xml_declaration=True)
            app = ET.fromstring(source.read("docProps/app.xml"))
            for element in app.iter():
                if element.tag.endswith("}Slides"):
                    element.text = str(count + len(args.append_slides))
            updates["docProps/app.xml"] = ET.tostring(app, encoding="utf-8", xml_declaration=True)
            package_names = set(source.namelist()) | set(additions)
            for member, data in additions.items():
                if member.endswith(".rels"):
                    for relationship in ET.fromstring(data):
                        target = relationship.get("Target", "")
                        if not target.startswith("/") or target.lstrip("/") not in package_names:
                            raise ValueError("Appended slide dependency is not preserved")
        with ZipFile(args.output, "w") as output:
            for entry in source.infolist():
                output.writestr(entry, updates.get(entry.filename, source.read(entry.filename)))
            for name, data in additions.items():
                output.writestr(name, data)
    with ZipFile(args.source) as source, ZipFile(args.output) as output:
        if set(output.namelist()) != set(source.namelist()) | set(additions):
            raise ValueError("Unexpected package member change")
        changed = [name for name in source.namelist() if source.read(name) != output.read(name)]
        if set(changed) != set(updates) or output.testzip() is not None:
            raise ValueError("Unexpected modified parts or corrupt archive")
    print(json.dumps({"changed_parts": changed, "added_parts": list(additions), "all_other_parts_identical": True}))


if __name__ == "__main__":
    main()
