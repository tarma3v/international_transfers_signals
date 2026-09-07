"""Package the reviewed final submission by Talent Track slot.

Only copies explicitly listed artifacts, extracts two already-reviewed PDF
pages, verifies the upload caps, and writes a checksummed ZIP. No model training,
platform uploads, repository checkout, or collection of workspace secrets.
"""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "submission/final-upload-2026-09-07"
DECK = ROOT / "submission/prezentaciya-itogovaya-2026-09-07.pdf"
ARCHIVE = ROOT / "submission/final-upload-2026-09-07.zip"

EXPECTED = {
    "01-prezentaciya": ["prezentaciya-itogovaya-2026-09-07.pdf", "prezentaciya-itogovaya-2026-09-07.pptx"],
    "02-opisanie-proekta": ["01-opisanie-proekta.md"],
    "03-produktovye-materialy": ["04-ai-product-i-tehnicheskaya-svyaz.md", "02-produktovoe-videnie.md", "03-zhurnal-dopushcheniy.md", "11-zapusk-pilot-ogranicheniya.md"],
    "04-dopolnitelnye-materialy": ["00-opis-komplekta.md", "05-tablica-rezultatov.md", "08-dve-metriki-dve-modeli.md", "10-produktovye-chisla.md", "13-ustarevanie-signala.md", "14-nezavisimaya-proverka-vetok.md", "15-itogovaya-model.md", "pushi-finalnye-shablony.md", "16-primer-temperatury.pdf", "17-makety-interfeysa.pdf"],
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build():
    for suffix in ("pdf", "pptx"):
        name = f"prezentaciya-itogovaya-2026-09-07.{suffix}"
        shutil.copy2(ROOT / "submission" / name, PACK / "01-prezentaciya" / name)
    reader = PdfReader(DECK)
    assert len(reader.pages) == 27
    for page, name in [(9, "16-primer-temperatury.pdf"), (23, "17-makety-interfeysa.pdf")]:
        writer = PdfWriter()
        writer.add_page(reader.pages[page])
        writer.add_metadata({"/Title": name[:-4], "/Author": "Команда 11", "/Subject": f"Страница {page+1} итоговой презентации от 07.09.2026"})
        dest = PACK / "04-dopolnitelnye-materialy" / name
        with dest.open("wb") as stream:
            writer.write(stream)
        assert len(PdfReader(dest).pages) == 1

    for folder, names in EXPECTED.items():
        present = {p.name for p in (PACK / folder).iterdir() if p.is_file() and p.name != "README.md"}
        assert present == set(names), (folder, present, names)
        assert (PACK / folder / "README.md").is_file()
    assert sha(DECK) == sha(PACK / "01-prezentaciya" / DECK.name)

    files = sorted(p for p in PACK.rglob("*") if p.is_file() and p.name != "manifest.json")
    assert all(p.suffix in {".md", ".pdf", ".pptx"} for p in files)
    manifest = {
        "package": PACK.name, "model_source_commit": "a3167cc", "main_source_commit": "7cb9335",
        "counts_excluding_instructions": {folder: len(names) for folder, names in EXPECTED.items()},
        "platform_write_performed": False,
        "files": [{"path": str(p.relative_to(PACK)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files],
    }
    (PACK / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(PACK.rglob("*")):
            if p.is_file():
                z.write(p, Path(PACK.name) / p.relative_to(PACK))
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for item in manifest["files"]:
            assert hashlib.sha256(z.read(f"{PACK.name}/{item['path']}")).hexdigest() == item["sha256"]
    print(json.dumps({"archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size, "sha256": sha(ARCHIVE), "counts": manifest["counts_excluding_instructions"], "verified": True}, ensure_ascii=False))

if __name__ == "__main__":
    build()
