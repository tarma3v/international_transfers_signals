# Что загружать на платформу — финальная сдача

Дедлайн: **07.09.2026, 11:00**. Все пути даны от корня репозитория
https://github.com/tarma3v/international_transfers_signals

Осталось четыре незаполненных слота. Репозиторий уже приложен.

---

## 1. Презентация проекта — финальная версия (лимит 2 файла)

    submission/prezentaciya-finalnaya.pptx
    submission/prezentaciya-finalnaya.pdf

Класть оба. Выступать нужно с .pptx: в PDF пропадают эмодзи в заголовках
текстов пушей — экспорт делался на машине без цветного эмодзи-шрифта.

24 слайда: 15 основных плюс приложение из 9. Слайд с командой — «П7».

---

## 2. Описание проекта — финальная версия (один markdown)

    submission/01-opisanie-proekta.md

Финальная редакция: аннотация на семь абзацев, разделы «Реализованные
компоненты», «Полученные результаты», «Дальнейшие планы по развитию»,
«Команда и распределение задач».

---

## 3. Продуктовые материалы — финальная версия (лимит 10, просят около 3)

Кладём четыре, в этом порядке:

    submission/04-ai-product-i-tehnicheskaya-svyaz.md
    submission/02-produktovoe-videnie.md
    submission/03-zhurnal-dopushcheniy.md
    submission/11-zapusk-pilot-ogranicheniya.md

Первый — самый важный: в задании прямо сказано, что главное здесь это
вклад AI Product и связь продуктовых решений с технической реализацией.
Больше четырёх не нужно: «не требуется готовить все возможные документы».

---

## 4. Дополнительные материалы — финальная версия (лимит 10)

Ровно десять, лимит выбирается полностью:

    submission/00-opis-komplekta.md
    submission/14-nezavisimaya-proverka-vetok.md
    submission/08-dve-metriki-dve-modeli.md
    submission/13-ustarevanie-signala.md
    submission/05-tablica-rezultatov.md
    submission/10-produktovye-chisla.md
    design/pushi-100-variantov.md
    submission/figures/08-put-klienta.pdf
    submission/figures/09-taymingi.pdf
    submission/figures/06-makety-interfeysa.pdf

Не кладём 06-prognon-bustingov.md, 07-ustoychivost-po-godam.md и
остальные рисунки: их содержание уже есть в презентации и в
05-tablica-rezultatov.md.

---

## Что нужно от вас до загрузки

**Степень участия.** Сейчас везде стоит 33 % / 33 % / 33 % — это
заглушка, а регламент требует фактическую долю. Нужна ваша цифра.
Поправить надо в двух местах:

    submission/01-opisanie-proekta.md   — раздел 10, таблица
    make_deck_final.js                  — слайд «П7 · Команда»

После правки презентацию нужно пересобрать:

    node make_deck_final.js
    soffice --headless --convert-to pdf --outdir submission submission/prezentaciya-finalnaya.pptx

---

## Проверка перед отправкой

- презентация открывается и на слайде «П7» стоят настоящие доли участия
- в описании проекта раздел 10 совпадает с презентацией
- ссылка на репозиторий открывается без авторизации
