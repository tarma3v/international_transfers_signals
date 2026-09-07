"""Keep untouched OOXML and embedded chart workbooks during a text-slide edit.

The changed slide is authored with Artifact Tool. This narrow packaging step
preserves all source parts except that slide and its speaker notes verbatim.
It accepts only slide 22 of this fixed defence deck, without inline relations.
"""
import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
P = "http://schemas.openxmlformats.org/package/2006/relationships"
PARTS = ("ppt/slides/slide22.xml", "ppt/notesSlides/notesSlide22.xml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("edited", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.resolve() in {args.source.resolve(), args.edited.resolve()}:
        raise ValueError("Output must differ from both input decks")
    with ZipFile(args.source) as source, ZipFile(args.edited) as edited:
        for part in PARTS:
            xml = ET.fromstring(edited.read(part))
            if any(k.startswith("{" + R + "}") for e in xml.iter() for k in e.attrib):
                raise ValueError("Inline relationship remapping would be required")
            part_path = Path(part)
            rel = str(part_path.parent / "_rels" / (part_path.name + ".rels"))
            source_targets = {(r.get("Type"), r.get("Target")) for r in ET.fromstring(source.read(rel)).findall("{" + P + "}Relationship")}
            edited_targets = {(r.get("Type"), r.get("Target")) for r in ET.fromstring(edited.read(rel)).findall("{" + P + "}Relationship")}
            if source_targets != edited_targets:
                raise ValueError("Source and edited slide relationships differ")
        with ZipFile(args.output, "w") as output:
            for entry in source.infolist():
                output.writestr(entry, edited.read(entry.filename) if entry.filename in PARTS else source.read(entry.filename))
    with ZipFile(args.source) as source, ZipFile(args.output) as output:
        if source.namelist() != output.namelist():
            raise ValueError("Unexpected package member change")
        changed = [name for name in source.namelist() if source.read(name) != output.read(name)]
        if set(changed) != set(PARTS) or output.testzip() is not None:
            raise ValueError("Unexpected modified parts or corrupt archive")
    print(json.dumps({"changed_parts": changed, "all_other_parts_identical": True}))


if __name__ == "__main__":
    main()
