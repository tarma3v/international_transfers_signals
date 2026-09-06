# T43: нелинейные взаимодействия улучшают rank, но не proper score

## Итог

T43 проверил одну заранее замороженную замену линейному history-эксперту T40:
`HistGradientBoostingClassifier` на тех же 41 причинных CBR-признаках и на тех
же annual rolling-origin выборках. Калибровка, causal prior, 50% log-odds
blend, исторические стадии и gates оставлены без изменения. Это изолирует один
вопрос: помогает ли нелинейная связь признаков различать рыночные режимы.

Нелинейность оказалась содержательно полезной, но недостаточной. На screen
2019–2022 AUC delta к prior выросла до **+0,03895**, а Brier-вред уменьшился
до **+0,00203** — лучший ранний результат среди T40–T43. Но Brier и log-loss
остаются статистически значимо хуже prior, ECE ухудшается на **+0,01672**, и
не проходит ни одна из девяти локальных year/currency групп.

Frozen historical gate отклонён. Target/model metrics 2025–2026 не
открывались. T43 не меняет production, T37, sparse push, expected future-only
bps или runtime router. `production_promoted=false`.

## Что именно проверялось

Для каждого evaluation year `Y`:

1. Взята точная T40 training mask до `Y-1` с mature h20 labels и двухдневным
   embargo.
2. Обучен один histogram gradient boosting с заранее заданными параметрами:
   160 деревьев, learning rate 0,05, максимум 15 листьев, minimum leaf 40 и
   L2=5. Early stopping выключен, чтобы все годы имели одинаковую capacity.
3. На полном предыдущем году применён тот же Platt calibrator, не видящий
   evaluation year.
4. Калиброванная вероятность смешана 50/50 в log-odds с causal prior.
5. Прогноз оценён на следующем полном году без выбора валют или строк.

Параметры унаследованы от существующей AP1 classical family. T43 не перебирал
глубину, число деревьев, признаки, веса или режимные пороги и не добавлял
ручной флаг СВО.

## Исторические результаты

| Стадия | N | Brier delta | Log-loss delta | ECE delta | AUC delta | Local pass |
|---|---:|---:|---:|---:|---:|---:|
| Screen 2019–2022 | 4 920 | **+0,00203** | **+0,00935** | **+0,01672** | **+0,03895** | **0 / 9** |
| Validation 2023–2024 | 2 475 | −0,00143 | −0,00582 | −0,00048 | +0,09820 | 5 / 7 |

На screen Brier CI целиком выше нуля:

- 20-date block: **[+0,00043; +0,00378]**;
- 50-date block: **[+0,00032; +0,00382]**.

Это доказывает ухудшение probability quality к prior, несмотря на лучший AUC.
AUC CI при этом широк и пересекает ноль. Validation снова привлекательна в
точке, но Brier CI пересекает ноль, 50-date AUC CI равен примерно
`[-0,01175; +0,22068]`, а 2024 и KZT не проходят local non-inferiority.

## Где модель помогает и где ломается

По годам screen выглядит неоднородно:

| Год | Brier delta | ECE delta | AUC delta |
|---:|---:|---:|---:|
| 2019 | +0,00149 | +0,00687 | +0,01964 |
| 2020 | +0,00291 | +0,02898 | **+0,22247** |
| 2021 | +0,00151 | +0,02571 | +0,01104 |
| 2022 | +0,00224 | +0,00392 | **−0,04390** |

2020 показывает главный конфликт: модель отлично сортирует строки по риску,
но превращает rank в слишком неточную probability. В 2022 ломается уже и
направление rank. Нелинейная capacity не устраняет regime instability.

Дополнительный диагностический сигнал — знак Platt slope, обученного только на
предыдущем году. Он отрицателен для evaluation years 2019, 2022 и 2024. Это
означает, что на доступном calibration year raw rank приходилось переворачивать.
Знак заметен причинно до evaluation year, но T43 заранее не использовал его
как gate. Подбирать правило после результата нельзя; допустимый следующий
эксперимент должен отдельно зарегистрировать причинный quality gate на
disjoint calibration evidence.

## Сравнение проверенных механизмов

| Механизм | Screen Brier delta | Screen AUC delta | Local pass |
|---|---:|---:|---:|
| T40: linear fixed blend | +0,00654 | +0,00439 | 0 / 9 |
| T41: mature quarterly shrink | +0,00425 | +0,01264 | 2 / 9 |
| T42: label-free OOD shrink | +0,00262 | +0,00337 | 0 / 9 |
| T43: nonlinear interactions | **+0,00203** | **+0,03895** | **0 / 9** |

T43 лучше извлекает порядок, но всё ещё не даёт честно калиброванную h20
температуру. Поэтому нельзя заменить T37 или заявить, что boosting решил
режимную проблему. Следующая гипотеза — не увеличивать capacity ещё раз, а
причинно разрешать expert только по независимому зрелому калибровочному куску.

## Воспроизводимость и аудит

- preregistration:
  `research/temperature_t43_nonlinear_history_registered.md`;
- расчёт:
  `research/temperature_t43_nonlinear_history.py`;
- реконструкция:
  `research/temperature_t43_nonlinear_history_audit.py`;
- сохранённый пакет:
  `results/research/temperature/t43_nonlinear_history/`;
- тесты:
  `tests/test_temperature_t43_nonlinear_history.py`.

Аудит воспроизвёл восемь annual fits, все probabilities, метрики, bootstrap и
gates, подтвердил фиксированные 160 итераций, maturity/embargo и неизменность
раннего prefix при порче будущих features/targets. `historical_gate_passed=false`,
`open_evaluated=false`, `production_promoted=false`.
