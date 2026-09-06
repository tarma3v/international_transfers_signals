# После курса ЦБ на завтра: новые residual-модели и causal Hedge

6 сентября 2026 | AP34-AP36 | Today-effective CBR | Решение после receipt

## Итог сверху

Проверены три принципиально новые идеи: continuous residual survival,
distributional monotone CatBoost и mature-only Brier Hedge. Все используют
завтрашний объявленный курс как доступный anchor и не видят незрелые ответы.

Ни одна не обошла AP33. Лидер остается календарным decision-router:
**h3/h5/h10/h20 = 2,421 / 2,490 / 2,471 / 2,516**, min rate **1,009**.

| Кандидат | h3 | h5 | h10 | h20 | Min lift | Min rate |
|---|---:|---:|---:|---:|---:|---:|
| **AP33 calendar router** | **2,421** | **2,490** | **2,471** | **2,516** | **2,421** | 1,009 |
| AP34 Ridge residual survival | 2,412 | 2,468 | 2,428 | 2,433 | 2,412 | **1,047** |
| AP35 distributional Cat | 2,391 | 2,449 | 2,414 | 2,402 | 2,391 | 1,039 |
| AP36 mature Brier Hedge | 2,401 | 2,449 | 2,437 | 2,418 | 2,401 | 1,032 |

AP34 и AP36 проходят строгие point-gates, но оба хуже AP33. AP35 проваливает
h3. AP35 h20 и AP36 h5/h20 доказанно хуже AP33 на 20-date paired bootstrap.
Период 2024-2026 открыт; AP33 остается frozen challenger, не production winner.

---page---

## 1. AP34: continuous residual survival

Вместо бинарного y20 AP34 предсказывает минимальную неизвестную просадку после
уже объявленного завтрашнего курса:

residual_floor20 = effective_floor20 − known_change.

Каждый из 17 quarterly OOS fits использует standardized Ridge(alpha=100) на
59 compact features. Target winsorized только по train 1%/99%. Ошибка модели
корректируется per-currency decayed state с half-life365 и shrinkage n/(n+150),
после чего train-only empirical residual CDF дает вероятность выживания текущего
курса. Global error weights имеют half-life730.

AP34 прошел early selector и late strict gates:

| Метрика | Результат |
|---|---:|
| h3/h5/h10/h20 | 2,412 / 2,468 / 2,428 / 2,433 |
| Min lift / min rate | 2,412 / 1,047 |
| h5 signals | 718 |
| h5 symmetric / future-only | 72,93 / 130,33 bp |

Но новый core стал слишком активным: 615 rolling-primary, 86 residual-pace и 6
month-rescue решений. После calendar-router осталось только 11 fallback AP23.
AP33 сохраняет более точные 673 AP26 core и добавляет 31 адресный fallback.
Все lift-delta CI AP34-AP33 пересекают ноль, но point-result ниже на каждом
неизвестном горизонте.

---page---

## 2. AP35: monotone distributional CatBoost

AP35 учит условную CDF residual floor, а не среднее. Для каждой зрелой train
строки созданы пять anchor-порогов: actual known change плюс
-200/-100/0/+100/+200 bp. Label показывает, пережил бы residual floor этот
синтетический запас.

Контекст - compact features без direct known-change transforms. Anchor добавлен
явно и имеет positive monotonic constraint. Каждый quarterly CatBoost использует
320 trees, depth 6 и train-only half-life 730 weights. Все fitted probe-curves
монотонны.

| Метрика | Результат |
|---|---:|
| h3/h5/h10/h20 | 2,391 / 2,449 / 2,414 / 2,402 |
| Min lift / min rate | 2,391 / 1,039 |
| h5 signals | 718 |
| h5 symmetric / future-only | 74,10 / 128,50 bp |

Модель прошла early gate, но late h3 ниже 2,4. Она выбрала 94 pace-решения и
оставила calendar-router почти без работы: всего 3 AP23 fallback. На h20 потеря
к AP33 равна -0,114 lift, 20-date delta CI **[-0,252; -0,021]**. Distributional
regularization делает probability логичнее, но ranking редких дней хуже.

---page---

## 3. AP36: честный mature-only Brier Hedge

AP36 объединяет три frozen OOS score: AP26 direct y20, AP34 Ridge survival и
AP35 distributional Cat. Каждый превращается в same-currency causal rank.

Перед каждым решением Brier loss считается только по предыдущим 730 дням и только
для y20, который полностью созрел минимум два дня назад. Global loss shrinks к
0,25 с strength 40; currency loss — к global с strength 40. Softmax
exp(−25 × loss) задает веса. Это online regime adaptation без post-hoc будущего.

Средние late weights:

| Expert | Mean weight |
|---|---:|
| AP26 y20 | **58,8%** |
| AP34 Ridge survival | 32,1% |
| AP35 distributional Cat | 9,1% |

Несмотря на правильный приоритет, Hedge dilutes AP26:
**h3/h5/h10/h20 = 2,401 / 2,449 / 2,437 / 2,418**, min rate 1,032.
H5 потеря к AP33 −0,041, CI **[−0,098; −0,004]**; h20 −0,099,
CI **[-0,207; -0,026]**. Mature Brier хорошо распознает среднюю калибровку,
но не редкую будущую competence именно в верхнем decision-tail.

---page---

## 4. Что теперь известно

Независимые audits подтвердили:

- 17 exact Ridge fits и 17 exact distributional CatBoost fits;
- train-only maturity masks, clipping, recency weights и empirical CDF;
- positive monotonicity всех AP35 fitted probes;
- все causal expert ranks, mature Brier masks/counts/losses/Hedge weights;
- nested AP21 source policy и frozen AP33 calendar-router;
- prefix invariance после порчи будущих features, labels, anchors и experts;
- max2/week, zero empty months и paired 20/50-date uncertainty.

Практический вывод: узкое место сейчас не probability calibration. AP26 работает
за счет редкого прямого y20 specialist и его точного decision-tail. Continuous
floor magnitude, synthetic CDF и average Brier склонны добавлять слишком много
средних pace-точек. AP33 выигрывает тем, что почти не меняет AP26 core и отдельно
решает cadence.

### Следующий bounded ход

AP33 заморожен. Не подбирать alpha, offsets, eta, half-life или calendar numbers
на открытом периоде. Следующая новая идея должна работать только с disagreement
tail: causal conformal uncertainty/veto или mature precision именно для редких
добавочных решений, не blend всех scores. Нужен один preregistered вариант.

Ограничения прежние: фактические receipt timestamps не сертифицированы, 18:00 не
гарантируется, а official-CBR benefit не равен банковской экономии без executable
bank price, spread, fees и подтверждения возможности перевода после receipt.
