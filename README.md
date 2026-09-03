# transfer_2: бенчмарк lift для моделей валютных сигналов

Проект тестирует классические ML-модели и прозрачные rule-based индикаторы для задачи хакатона: находить дни, когда по валютному коридору есть полезный сигнал для трансграничного перевода.

## Данные

По умолчанию проект читает нормализованные курсы ЦБ из соседнего проекта:

```bash
../international_transfers_signals/data/cbr_rates.json
```

Путь к данным можно переопределить через параметр `--data`.

## Окружение

```bash
cd transfer_2
python3 -m venv .venv
. .venv/bin/activate
pip3 install -r requirements.txt
pip3 install -e .
```

Для разработки и тестов:

```bash
pip3 install -r requirements-dev.txt
pytest
```

## Запуск бенчмарка

```bash
python3 scripts/run_lift_benchmark.py --horizon 5 --top-rate 0.15
```

Результат — таблица по моделям и коридорам: hit rate, базовый hit rate случайного дня, lift, частота выбранных сигналов и средняя форвардная выгода в базисных пунктах.
