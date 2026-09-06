# После курса ЦБ на завтра: mature precision и режимы решений

6 сентября 2026 | AP37-AP39 | Today-effective CBR | Решение после receipt

## Итог сверху

AP37 дал новый лучший строгий point-result. Он не меняет 673 точных AP26 core
решения, а причинно фильтрует только редкий calendar fallback по надёжности
согласия трёх моделей на уже созревшем прошлом.

| Кандидат | h3 | h5 | h10 | h20 | Min lift | Min rate |
|---|---:|---:|---:|---:|---:|---:|
| AP33 calendar leader | 2,421 | 2,490 | 2,471 | 2,516 | 2,421 | 1,009 |
| **AP37 mature precision** | **2,429** | **2,509** | **2,479** | **2,525** | **2,429** | **1,009** |
| AP38 guarded core | 2,432 | **2,521** | 2,486 | **2,597** | 2,432 | 0,955 |
| AP39 one-week runway | **2,435** | 2,509 | 2,478 | 2,572 | **2,435** | 0,971 |

AP37 проходит все point-gates: 1-2 сигнала на валюту в неделю, zero empty
months и max2/week. AP38/AP39 точнее на части горизонтов, но проваливают cadence
и ранний selector, поэтому это только accuracy-frontier, не победители.

Прирост AP37 к AP33 мал и статистически не доказан: поздний период 2024-2026
многократно открыт. AP37 нужно заморозить для prospective shadow.

---page---

## 1. AP37: что именно делает новая модель

Есть три независимых честных OOS score:

- AP26 direct y20 specialist;
- AP34 continuous residual survival;
- AP35 monotone distributional CatBoost.

Для каждого дня каждый score переводится в причинный percentile своей валюты.
Support stratum - сколько из трёх ranks не ниже 0,70, то есть находятся в
существующем top-30 tail. Возможные значения: 0, 1, 2 или 3.

Цель precision-модели строгая: успешен ли день одновременно на h3, h5, h10 и
h20. Перед текущим решением используются только прошлые строки, для которых
полный h20 созрел минимум два дня назад по publication-calendar. Same-date,
immature и future labels исключены.

Оценка иерархическая и expanding:

1. общая precision не-core дней shrink к 0,50 с strength 40;
2. precision текущего support stratum shrink к общей с strength 40;
3. precision stratum конкретной валюты shrink к global stratum с strength 40.

AP23 fallback в четверг или пятницу разрешается, только если локальная shrunk
precision не хуже причинной общей precision. Silence-10 rescue сохраняется даже
при плохом score, чтобы продукт не исчезал надолго. AP26 core не меняется.

Правило было зарегистрировано одним вариантом. До 2024 года support=3 имел
59,8% строгого all-horizon success против примерно 27,5-33,6% у support=0-2;
поздний период при выборе правила не использовался.

---page---

## 2. Результат AP37

| Метрика | AP33 | AP37 | Delta |
|---|---:|---:|---:|
| h3 adjusted lift | 2,421 | **2,429** | +0,008 |
| h5 adjusted lift | 2,490 | **2,509** | +0,019 |
| h10 adjusted lift | 2,471 | **2,479** | +0,008 |
| h20 adjusted lift | 2,516 | **2,525** | +0,009 |
| h5 signals | 704 | 695 | -9 |
| h5 symmetric benefit | 74,60 bp | 74,55 bp | -0,05 bp |
| h5 future-only | 132,94 bp | **133,73 bp** | +0,79 bp |

AP37 сохранил 693 решения AP33, удалил 11 и добавил 2 более поздних. Итоговый
late поток: 673 AP26 core, 1 precision late-week и 21 silence-10 fallback.
Минимальная rate по всем неизвестным горизонтам и валютам равна 1,0087;
пустых полных currency-month нет, максимум 2 решения в ISO-неделю.

20-date paired delta CI для lift AP37-AP33:

- h3: +0,008, CI [-0,011; +0,044];
- h5: +0,019, CI [-0,004; +0,059];
- h10: +0,008, CI [-0,015; +0,041];
- h20: +0,009, CI [-0,017; +0,039].

Все интервалы пересекают ноль. Это новый лучший frozen point-challenger, но не
доказанный production winner. H1 не участвует в выборе: после receipt первый
следующий курс уже известен.

---page---

## 3. AP38 и AP39: почему не стали победителями

AP38 применил такую же mature precision к самому AP26 core. Слабый regime можно
пропустить только когда текущая trailing-365 rate уже не ниже 1. Это удалило 36
решений и добавило 3 поздних замены.

Accuracy выросла: h5 2,521, h20 2,597, h20 delta к AP37 +0,072 с 20-date
CI [+0,006; +0,154]. Но min currency rate упала до 0,955, ранний 2023 также
дал только 0,856 по худшей валюте и два пустых месяца. AP38 не проходит ТЗ.

AP39 заменил текущий rate guard на один полный week runway:
вычёркивать слабый core можно, только если n_selected/(elapsed_weeks+1) >= 1.
Это продуктово заданный, заранее зарегистрированный вариант, не grid.

Он оставил 669 h5 сигналов: h3/h5/h10/h20 =
2,435 / 2,509 / 2,478 / 2,572. Средняя h5 rate почти ровно 1,000, но худшая
валюта только 0,979, all-h min rate 0,971; ранний selector также провален.

Главный вывод: weak-core режим распознаётся, но редкие будущие возможности не
приходят по расписанию. Текущий causal rate и даже одна неделя запаса не могут
гарантировать corridor cadence. Подбирать две, три или четыре недели по уже
открытому scorecard нельзя - это был бы post-hoc tuning.

---page---

## 4. Аудит, статус и следующий новый класс

Независимые audits подтвердили:

- 5 755 строк и effective-reference targets пересобраны из источника;
- все три frozen OOS expert rank совпадают точно;
- maturity masks, support strata, counts и трёхуровневый shrinkage совпадают;
- AP37/AP38/AP39 rate, silence, reason, veto, runway и weekly cap восстановлены;
- порча будущих experts, labels и source decisions не меняет прошлый prefix;
- early selection, paired 20/50-date uncertainty и max2/week пересчитаны.

Практический статус: AP37 - новый strict point leader. AP38 - интересная
accuracy upper bound, которая показывает потенциал regime filtering, но не
продукт. AP39 - отрицательный cadence repair. Ни один результат не является
fresh holdout; historical receipt timestamps и банковская исполнимость не
сертифицированы.

### Следующий bounded ход

AP37 заморожен. Не подбирать support threshold, shrinkage или runway. Следующий
новый класс - causal weekly optimal stopping: по созревшим прошлым неделям
оценивать, выгоднее ли принять текущую возможность или сохранить один слот до
следующего дня, при этом Friday/silence rescue остаётся жёстким. Нужна одна
предварительно зарегистрированная классическая модель и проверка на том же
today-effective target.
