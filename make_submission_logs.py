"""Собрать сырые логи прогонов в markdown-файлы комплекта сдачи.

Раньше `submission/*.txt` были ручными копиями `results/*.txt`: побайтово
одинаковыми ровно до первого прогона, после которого расходились молча. Здесь
они собираются из источника, поэтому разойтись не могут.

Формат — markdown с огороженным блоком: колонки в логах выровнены пробелами, и
без огораживания markdown их схлопывает, превращая таблицу в кашу.
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("results")
OUT = Path("submission")

# имя в комплекте -> (файл-источник, заголовок, что это, чем получено)
LOGS: dict[str, tuple[str, str, str, str]] = {
    "06-prognon-bustingov": (
        "boosting_output.txt",
        "Прогон бустингов",
        "Сравнение CatBoost и XGBoost с отбором признаков, включая полный "
        "рейтинг внутренней валидации — чтобы выбор конфигурации до теста был "
        "проверяем, а не декларирован.",
        "run_boosting.py",
    ),
    "07-ustoychivost-po-godam": (
        "stability_output.txt",
        "Устойчивость по годам и по коридорам",
        "Проверка того, что результат не держится на одном годе и не держится "
        "на одном коридоре: out-of-time блок внутри каждого коридора плюс "
        "проверка с выброшенным лучшим коридором.",
        "check_stability.py",
    ),
    "09-prognon-dvuh-modeley": (
        "two_models_output.txt",
        "Две метрики, две модели",
        "Сравнение модели и правила под каждую из двух метрик и разложение "
        "итога по дню срабатывания.",
        "run_two_models.py",
    ),
    "10-produktovye-chisla": (
        "product_numbers_output.txt",
        "Продуктовые величины",
        "Размах курса, потолок выгоды, цена ожидания, светофор состояний, "
        "кучность сигналов, эффекты праздников и размер пилота — разбивка по "
        "коридорам и по годам.",
        "run_product_numbers.py",
    ),
    "13-ustarevanie-signala": (
        "staleness_output.txt",
        "Устаревание сигнала",
        "Сколько живёт срабатывание между расчётом и открытием пуша: держится "
        "ли условие, что клиент теряет на задержке и когда остаток выгоды "
        "перестаёт отличаться от случайного дня.",
        "run_signal_staleness.py",
    ),
}


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"нет каталога {SRC}")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (src_name, title, what, script) in LOGS.items():
        src = SRC / src_name
        if not src.exists():
            raise SystemExit(f"нет источника {src} — сначала прогоните {script}")
        body = src.read_text(encoding="utf-8").rstrip("\n")
        if "```" in body:
            raise SystemExit(f"{src}: в логе есть ``` — огораживание сломается")
        dest = OUT / f"{name}.md"
        dest.write_text(
            f"# {title}\n\n{what}\n\n"
            f"Сырой вывод `{script}`, скопирован из `results/{src_name}` "
            f"скриптом `make_submission_logs.py`. Руками здесь не правится "
            f"ничего: любая правка исчезнет при следующей пересборке.\n\n"
            f"```text\n{body}\n```\n",
            encoding="utf-8",
        )
        print(f"  {dest}  ({dest.stat().st_size // 1024} КБ)")


if __name__ == "__main__":
    print("сборка логов комплекта:")
    main()
