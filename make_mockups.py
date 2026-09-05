"""Рендер макетов интерфейса в PNG для комплекта сдачи.

Картинки не рисуются руками и не лежат в репозитории отдельно от исходника:
они собираются из `design/interfeys.html`, того же файла, который открывает
человек. Иначе макет в документе и макет в браузере расходятся молча — ровно
так же, как расходятся захардкоженные числа и свежий прогон.

Требуется Chrome (headless). На macOS путь по умолчанию:
/Applications/Google Chrome.app/Contents/MacOS/Google Chrome
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path("design/interfeys.html")
JOURNEY = Path("design/put-klienta.html")
OUT = Path("submission/figures")

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
]


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if os.path.sep in c:
            if Path(c).exists():
                return c
        else:
            from shutil import which

            found = which(c)
            if found:
                return found
    raise RuntimeError(
        "не найден Chrome. Укажите путь в переменной окружения CHROME_BIN"
    )


def extract_block(html: str, start_pat: str, occurrence: int = 0) -> str:
    """Вырезать <div ...>...</div> с балансировкой вложенных div.

    Регулярка до первого `</div>` вырезала бы кусок макета: у телефона внутри
    полтора десятка вложенных блоков.
    """
    positions = [m.start() for m in re.finditer(start_pat, html)]
    if occurrence >= len(positions):
        raise RuntimeError(f"не нашёл блок {start_pat!r} №{occurrence}")
    i = positions[occurrence]
    depth = 0
    for m in re.finditer(r"<div\b|</div>", html[i:]):
        depth += 1 if m.group(0) != "</div>" else -1
        if depth == 0:
            return html[i : i + m.end()]
    raise RuntimeError(f"незакрытый блок {start_pat!r}")


def render_pdf(chrome: str, html: str, width: int, height: int, dest: Path) -> None:
    """То же содержимое в PDF: векторный текст, растровыми остаются только шрифты.

    Размер страницы задаётся через @page под фактический блок, иначе Chrome
    сверстает макет на A4 и обрежет широкий ряд телефонов. Флаг отключения
    колонтитулов называется --no-pdf-header-footer; похожий по смыслу
    --print-to-pdf-no-header не существует и молча игнорируется.
    """
    page = (f"<style>@page{{size:{width}px {height}px;margin:0}}"
            f"html,body{{width:{width}px}}</style>")
    html = html.replace("</head>", page + "</head>", 1)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(html)
        tmp = fh.name
    try:
        subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--no-sandbox",
             "--virtual-time-budget=10000", "--no-pdf-header-footer",
             f"--print-to-pdf={dest}", f"file://{tmp}"],
            check=True, capture_output=True,
        )
    finally:
        os.unlink(tmp)
    if not dest.exists():
        raise RuntimeError(f"Chrome не создал {dest}")
    print(f"  {dest}  ({dest.stat().st_size // 1024} КБ)")


def render(chrome: str, html: str, width: int, height: int, dest: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(html)
        tmp = fh.name
    try:
        subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--no-sandbox",
             "--hide-scrollbars", "--force-color-profile=srgb",
             "--virtual-time-budget=10000",
             f"--window-size={width},{height}",
             f"--screenshot={dest}", f"file://{tmp}"],
            check=True, capture_output=True,
        )
    finally:
        os.unlink(tmp)
    if not dest.exists():
        raise RuntimeError(f"Chrome не создал {dest}")
    print(f"  {dest}  ({dest.stat().st_size // 1024} КБ)")


def main() -> None:
    if not SRC.exists():
        sys.exit(f"нет исходника {SRC}")
    OUT.mkdir(parents=True, exist_ok=True)
    chrome = os.environ.get("CHROME_BIN") or find_chrome()

    src = SRC.read_text(encoding="utf-8")
    style = re.search(r"<style>.*?</style>", src, re.S)
    fonts = re.search(r'<link rel="stylesheet" href="https://fonts\.[^"]+">', src)
    if style is None or fonts is None:
        sys.exit("в исходнике не найден блок стилей или подключение шрифтов")

    # светлая тема принудительно: PNG уедет в документ и в презентацию, где
    # тёмный вариант читался бы как ошибка вёрстки
    head = (
        '<!doctype html><html lang="ru" data-theme="light"><head><meta charset="utf-8">'
        + fonts.group(0) + style.group(0)
        + "<style>body{background:#FFF;margin:0;padding:26px}"
        ".row{display:flex;gap:22px;align-items:flex-start}"
        ".cap{font-family:var(--mono);font-size:11px;letter-spacing:.06em;"
        "text-transform:uppercase;color:#6E7A88;margin:0 0 10px 4px}"
        ".col{display:flex;flex-direction:column}</style></head><body>"
    )

    # Порядок жёстко привязан к порядку в исходнике: добавите телефон в
    # середину — подписи молча съедут на один. Поэтому список подписей и
    # число телефонов держим рядом и проверяем длину.
    caps = ["01 · экран суммы · 1 сентября", "02 · пуш · 3 сентября, 20:32",
            "03 · тот же пуш переписан · 4 сентября, 20:35",
            "04 · после тапа · 3 сентября, 20:33",
            "05 · открыт позже · 5 сентября, 8:14",
            "06 · уровень · 1 сентября", "07 · список · 1 сентября"]
    phones = [extract_block(src, r'<div class="phone">', k) for k in range(len(caps))]
    found = len(re.findall(r'<div class="phone">', src))
    if found != len(caps):
        sys.exit(f"в макете {found} телефонов, а подписей {len(caps)} — поправьте caps")
    row = "".join(f'<div class="col"><p class="cap">{c}</p>{p}</div>'
                  for c, p in zip(caps, phones))
    print("рендер макетов:")
    doc = head + f'<div class="row">{row}</div></body></html>'
    render(chrome, doc, 2780, 830, OUT / "06-makety-interfeysa.png")
    render_pdf(chrome, doc, 2780, 830, OUT / "06-makety-interfeysa.pdf")

    states = extract_block(src, r'<div class="states">')
    doc = head + f'<div style="max-width:1120px">{states}</div></body></html>'
    render(chrome, doc, 1180, 530, OUT / "07-tri-sostoyaniya-vidzheta.png")
    render_pdf(chrome, doc, 1180, 530, OUT / "07-tri-sostoyaniya-vidzheta.pdf")

    # Схема пути — самостоятельная страница целиком, вырезать из неё нечего.
    # Ширина 1900: при размещении на слайде 11 дюймов шрифт 16,5 px даёт ~7 пт,
    # то есть читается с проектора. Уже 1900 — текст уходит в нечитаемое.
    if JOURNEY.exists():
        doc = ('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
               + JOURNEY.read_text(encoding="utf-8") + "</head><body></body></html>")
        render(chrome, doc, 1900, 1010, OUT / "08-put-klienta.png")
        render_pdf(chrome, doc, 1900, 1010, OUT / "08-put-klienta.pdf")


if __name__ == "__main__":
    main()
