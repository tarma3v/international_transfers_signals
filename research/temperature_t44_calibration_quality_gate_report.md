# T44: disjoint mature quality gate безопасен, но не доказывает новый прирост

## Итог

T44 проверил причинный selector для нелинейного h20 history-эксперта T43.
Для каждого следующего года предыдущий год заранее делился пополам: на первой
половине обучалась probability-map, а на второй, не участвовавшей в её fit,
проверялось, можно ли вообще разрешить эксперту влиять на прогноз. Все h20
ответы должны были полностью созреть до соответствующего origin с двухдневным
embargo.

Механизм выполнил защитную задачу: он закрыл эксперт во всех четырёх годах
screen 2019-2022 и тем самым сделал результат побитово равным причинному prior.
На validation он открыл эксперт только в 2023 и улучшил pooled Brier, log-loss,
ECE и AUC; в 2024 снова выбрал prior. Однако frozen протокол требовал на screen
не просто отсутствия вреда, а статистически доказанного улучшения. Равенство
baseline даёт нулевые deltas и нулевые интервалы, поэтому gate формально не
пройден. Model/outcome metrics 2025-2026 не открывались, production и T37 не
изменены.

## Как построено решение

Для query year `Y`:

1. T43 `HistGradientBoostingClassifier` обучается на данных строго до `Y-1`.
2. На mature январе-июне `Y-1` обучается ранний Platt calibrator.
3. На mature июле-декабре `Y-1` проверяется смесь 50/50 с causal prior.
4. Эксперт разрешается только при положительном Platt slope, одновременном
   улучшении pooled Brier/log-loss/AUC, допустимой ECE и non-inferiority всех
   пяти валют.
5. Если gate закрыт, весь год `Y` получает causal prior без численного
   изменения. Если открыт, применяется прежняя T43-модель с Platt на полном
   предыдущем году.

В каждом полугодии не менее 400 строк. Нет grid, выбора валют, ручного SVO-флага
или использования query-year outcomes.

## Решения gate по годам

| Query year | Brier delta на gate | Log-loss delta | ECE delta | AUC delta | Валюты pass | Решение |
|---:|---:|---:|---:|---:|---:|---|
| 2019 | +0,00114 | +0,00372 | +0,00945 | +0,14680 | 3/5 | prior |
| 2020 | -0,00064 | -0,00288 | -0,01404 | +0,20315 | 2/5 | prior |
| 2021 | +0,00848 | +0,02283 | +0,07095 | +0,24373 | 0/5 | prior |
| 2022 | -0,00083 | -0,00362 | -0,00359 | +0,28178 | 3/5 | prior |
| 2023 | **-0,00099** | **-0,00282** | **-0,00129** | **+0,13745** | **5/5** | **expert** |
| 2024 | +0,00710 | +0,02204 | +0,06873 | +0,27839 | 0/5 | prior |
| 2025 | -0,00363 | -0,01437 | -0,01279 | +0,21135 | 5/5 | expert, outcome закрыт |
| 2026 | +0,00763 | +0,04577 | +0,07381 | +0,34076 | 0/5 | prior |

Строка 2025 сообщает только решение, принятое из полностью прошлого 2024 gate.
Она не является оценкой качества 2025: target/model metrics позднего периода
условно не вычислялись после провала historical stage.

## Historical scorecard

| Стадия | N | Brier delta | Log-loss delta | ECE delta | AUC delta | Local pass |
|---|---:|---:|---:|---:|---:|---:|
| Screen 2019-2022 | 4 920 | 0 | 0 | 0 | 0 | 9/9 |
| Validation 2023-2024 | 2 475 | **-0,00098** | **-0,00357** | **-0,00749** | **+0,12835** | **7/7** |

Validation point metrics выглядят хорошо. Но 20/50-date paired intervals для
Brier имеют верхние границы +0,00101/+0,00112, а для AUC нижние границы
-0,00116/-0,00413. То есть доказательство улучшения на этой стадии тоже не
достаточно сильное.

## Что мы узнали

- Независимая mature competence-проверка реально умеет предотвращать вред:
  четыре плохих screen-года стали точным baseline вместо отрицательного T43.
- Причинная осторожность и новый predictive gain - разные свойства. Selector,
  который всегда воздерживается, безопасен, но не является улучшенной моделью.
- Высокий AUC на gate недостаточен. В 2020 и 2022 pooled rank был сильным, но
  несколько валют нарушили proper-score/non-inferiority условия.
- T44 подтверждает, что общая regime-проблема находится в переносе
  probability-map между периодами, а не только в capacity бустинга.

Следующий честный шаг - не ослаблять T44 после просмотра, а оставить его как
prospective control и собирать новые outcomes. Для нового offline-механизма
нужен независимый observable source-state, которого не было в T40-T44, либо
иная цель/источник данных с отдельной preregistration.

## Воспроизводимость

- preregistration:
  `research/temperature_t44_calibration_quality_gate_registered.md`;
- расчёт:
  `research/temperature_t44_calibration_quality_gate.py`;
- аудит:
  `research/temperature_t44_calibration_quality_gate_audit.py`;
- результаты:
  `results/research/temperature/t44_calibration_quality_gate/`;
- тесты:
  `tests/test_temperature_t44_calibration_quality_gate.py`.

Аудит пересобрал восемь annual fits, обе калибровки, решения gate, predictions,
historical metrics и bootstrap. Подтверждены maturity/embargo, exact-prior в
закрытых годах и invariance раннего prefix при порче будущих features, targets
и maturity. `historical_gate_passed=false`, `open_evaluated=false`,
`production_promoted=false`.
