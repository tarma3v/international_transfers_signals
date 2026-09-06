# T36: source-driven h20-router нашёл сигнал, но был переуверен

## Идея

T36 убрал расписание из T35 и для каждого запроса восстановил фактически
доступный h20-источник:

- `cbr_history` -> последний T34 через backward-only publication join;
- receipt-dependent source -> T22, только в receipt replay;
- market, bridge, perpetual, stale market и hold -> точный T19 identity.

Маршрут зависит только от информации, доступной в `as_of`; target и будущие
строки не участвуют. В no-receipt сценарии T22 не появляется ни разу.

## Результат на открытом 2025-2026

| Сценарий | Brier: база -> T36 | AUC: база -> T36 | ECE: база -> T36 |
|---|---:|---:|---:|
| calendar-assumed receipt | 0,11903 -> **0,11345** | 0,5659 -> **0,7018** | 0,03492 -> 0,03941 |
| no same-day receipt | 0,11754 -> **0,11347** | 0,6005 -> **0,7024** | 0,03196 -> 0,04143 |

Оба pooled Brier CI полностью ниже нуля, оба AUC CI полностью выше нуля. Оба
компонента по отдельности улучшили Brier/log-loss/AUC: T22 после receipt имеет
AUC 0,7383, raw T34 на history - 0,6502.

## Почему formal pass всё равно false

Заранее было запрещено ухудшать ECE больше чем на 0,01 в любом срезе
`scenario x clock`. Raw T34 пересаживал вероятности слишком далеко: 12 из 40
состояний нарушили локальный ECE-gate, хотя Brier и AUC в них улучшились.
Порог после просмотра не ослаблялся.

`retrospective_route_passed=false`, `production_promoted=false`. T36 доказал,
что source-driven граница правильна и rank полезен, но пользовательская шкала
0-100 требует более осторожной силы поправки. Это мотивировало один заранее
заданный shrink в T37.

Полные результаты: `results/research/temperature/t36_source_driven_h20_router/`.
