# T20: иерархическая калибровка не переносится на новый режим

## Короткий вывод

Выбранная до расчёта `hierarchical_beta` не прошла ни одного из 120 строгих
state gates на открытом 2025–2026. Она ухудшила h5/h10 и не дала совместного
улучшения Brier, log-loss и ECE на h20. Локальная проблема тоже не исчезла:
число currency-year-clock-h строк с ECE > 0,08 выросло с 789 у frozen identity
до 816.

Модель не продвигается в router. AP37, T17/T18 и сохранённые push остаются без
изменений.

## Протокол

- fit origin: 01.01.2025 00:00 MSK;
- train: только запросы 2024 года, target которых полностью созрел строго до
  fit origin минус двухдневный embargo;
- test: открытый 2025–2026, только diagnostic;
- 20 clocks, два receipt-сценария, h5/h10/h20;
- 1 750 / 1 715 / 1 645 train rows на h5/h10/h20;
- 3 015 / 2 980 / 2 910 evaluation rows на одно состояние;
- всего 356 200 evaluation rows.

Primary-кандидат обучался отдельно для каждого `scenario × clock × h`.
Признаки: base probability в Beta-форме, currency intercept/slope и causal
regime `phase|source_kind|confidence|freshness`. Сильный L2 shrinkage был
зафиксирован как C=0,05.

## Средний результат primary против frozen identity

| h | Δ Brier | Δ log-loss | Δ ECE | PASS |
|---:|---:|---:|---:|---:|
| 5 | +0,00591 | +0,01484 | +0,01944 | 0 / 40 |
| 10 | +0,00394 | +0,01150 | +0,01850 | 0 / 40 |
| 20 | −0,00030 | +0,00060 | +0,01102 | 0 / 40 |

Отрицательный Δ Brier означает улучшение. На h20 средний Brier чуть лучше,
но log-loss и ECE хуже, а максимальная верхняя bootstrap-граница остаётся
положительной. В `no_same_day_receipt` h20 point log-loss почти не меняется
(−0,00007), однако строгая 20/50-day неопределённость всё равно не позволяет
принять калибратор.

## Контрольные варианты

| Модель | h5 Brier | h10 Brier | h20 Brier |
|---|---:|---:|---:|
| frozen identity | 0,17428 | 0,14358 | 0,11828 |
| fixed logit shrink 0,8 | 0,17856 | 0,14921 | 0,12328 |
| global Platt | 0,18008 | 0,14839 | 0,11790 |
| hierarchical Beta | 0,18020 | 0,14752 | 0,11798 |

Простой shrink ухудшает все горизонты. Global Platt немного снижает средний
h20 Brier, но повышает средний log-loss и ECE; это diagnostic control, а не
новый победитель. Frozen probabilities неожиданно устойчивее одноразовой
перекалибровки на 2024.

## Почему подход провалился

1. Один 2024 год слишком узок для устойчивых currency/regime поправок.
2. У h20 основная проблема не только calibration: T19 показал слабую и
   нестабильную discrimination, которую monotone map исправить не может.
3. Фазы и source states меняют состав в 2025–2026. Категориальная поправка,
   выученная на старом составе, переносит bias.
4. Иерархия была реализована как penalized one-shot fit, а не динамическая
   state-space модель с контролируемым forgetting.

## Следующий вывод

Не следует снова перебирать C и набор interaction по открытому тесту. Для h20
нужна новая голова, улучшающая rank/discrimination, а не только mapping
вероятности. Для локальной калибровки разумнее prospective online update с
фиксированным forgetting и минимальным effective sample size. До накопления
новых исходов production должен удерживать frozen probability и явно снижать
confidence на h20 и слабых ранних режимах.

Полные результаты: `results/research/temperature/t20_hierarchical_calibration/`.

