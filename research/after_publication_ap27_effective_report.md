# После курса ЦБ на завтра: adaptive pace и specialist models

6 сентября 2026 | AP23-AP27 | Today-effective CBR | Решение после receipt

## Главный результат сверху

Новый этап подтвердил две разные границы качества. Лучший строгий point-result
дает AP23 mature-only competence pace: minimum adjusted lift **2,407** по
h3/h5/h10/h20, минимальная частота **1,039**, ноль пустых полных месяцев и
максимум 2 сигнала в ISO-неделю. Лучший accuracy-result дает AP26 y20 specialist
со shrinkage: minimum lift **2,417**, h5 **2,483**, но min rate **0,971**.

AP27 добавляет всего 26 causal backstop решений и возвращает specialist к
строгому cadence: h3/h5/h10/h20 = **2,405 / 2,465 / 2,444 / 2,467**, min rate
**1,001**, zero empty months, max2/week.

| Кандидат | h3 | h5 | h10 | h20 | Min lift | Min rate | Статус |
|---|---:|---:|---:|---:|---:|---:|---|
| AP21 rolling/Cat | 2,402 | 2,446 | 2,429 | 2,411 | 2,402 | 1,009 | Strict control |
| **AP23 adaptive pace** | **2,407** | **2,466** | **2,465** | **2,472** | **2,407** | **1,039** | Best strict point |
| AP26 y20 shrink200 | **2,417** | **2,483** | 2,446 | **2,495** | **2,417** | 0,971 | Accuracy frontier |
| AP27 early-selected | 2,413 | 2,461 | 2,433 | 2,482 | 2,413 | 0,986 | Late rate fail |
| AP27 specialist strict | 2,405 | 2,465 | 2,444 | 2,467 | 2,405 | 1,001 | Strict late passer |

Сильный point-frontier достигнут без утечки, но нового fresh winner нет. AP23
не прошел early cadence, а AP27 strict не был выбран ранним selector. Период
09.01.2024-25.08.2026 многократно открыт; все новые записи являются
ретроспективными challengers для будущего frozen shadow.

---page---

## 1. AP23: веса экспертов только по созревшим ошибкам

AP22 смешивал rolling и local scores без labels и ухудшил результат. AP23 впервые
разрешает адаптацию по фактическому качеству, но использует только исходы, чей
полный h20 созрел раньше текущего дня минимум на два дня.

Для каждого эксперта вычисляется precision его прошлых top30% прогнозов по
utility = mean(y3,y5,y10,y20). Currency precision стягивается к global precision,
а веса обновляются на окне 730 дней. H1 не входит ни в response, ни в selector.

Главная находка - адаптировать только pace expert. Rolling ExtraTrees остается
primary, а вес между AP18 blend и CatBoost utility меняется по зрелой прошлой
точности. Попытка так же маршрутизировать primary ухудшает minimum lift до
2,33-2,35.

| AP23 competence pace | h3 | h5 | h10 | h20 |
|---|---:|---:|---:|---:|
| Adjusted lift | **2,407** | **2,466** | **2,465** | **2,472** |
| Средняя rate | 1,057 | 1,059 | 1,052 | 1,047 |
| Min rate валюты | 1,051 | 1,053 | 1,046 | **1,039** |
| Symmetric, б.п. | 61,6 | **73,8** | 78,2 | 86,4 |
| Future-only, б.п. | 122,0 | **131,0** | 130,2 | 124,7 |

Против AP21 h5 delta +0,020, h10 +0,036, h20 +0,061. Все paired интервалы
пересекают ноль, хотя для h20 нижняя граница близка к нулю. Значит direction
полезен, но статистически доказанного скачка пока нет.

Early-2023 min rate равен 0,985, поэтому AP23 не прошел ранний joint gate и не
может называться выбранным победителем. Его корректный статус - лучший строгий
late point challenger.

---page---

## 2. AP24: direct ranking не переносится как primary

AP24 сменил класс задачи: пять CatBoostRanker учились выбирать лучшие дни внутри
currency-quarter или currency-month, а не предсказывать каждую строку отдельно.
Проверены PairLogit и YetiRankPairwise, expanding и rolling-1095 history, mean
utility и y20 target. Все 85 quarterly fits были OOS и mature-only.

Ранний 2023 selector выбрал quarter PairLogit mean utility как primary. Он прошел
все ранние gates, но на 2024-2026 minimum lift упал до **2,233**, h5 до **2,289**.
Это настоящий transport failure, а не ошибка воспроизводимости: независимый
refit и физическая порча будущих feature/labels дали тот же prefix.

| Роль ranker | Лучший late h5 | Min lift | Min rate | Вывод |
|---|---:|---:|---:|---|
| Primary, early-selected | 2,289 | 2,233 | 1,059 | Не переносится |
| YetiRank pace | **2,464** | **2,397** | 1,039 | Полезен как второй голос |
| AP23 competence pace | 2,466 | 2,407 | 1,039 | Сильнее ranker |

Причина интерпретируема: relative ordering внутри режима 2023 меняется после
режимного сдвига, а primary принимает слишком много решений. Rolling ExtraTrees
устойчивее за счет recent-window обучения и менее агрессивной цели. Ranking
модель допустима только как редкий pace expert.

Практический запрет для следующих раундов: не заменять rolling primary сложным
ranker по хорошему раннему scorecard. Сначала требовать regime-transport и
prospective shadow.

---page---

## 3. AP25-AP26: модель специально для трудных cadence-точек

AP25 обучает модель только на outcome-free hard pool: known-next-not-lower,
rolling rank не выше top30%, reserve rank выше top30%. Это именно дни, из которых
контроллер вынужден добирать частоту. Labels входят только после полного maturity.

Проверены CatBoost mean utility, y20 classification, future5 benefit regression,
rolling-3-year CatBoost и ExtraTrees regression. Лучший y20 specialist:
minimum lift **2,403**, h5 **2,474**, future-only **136,45 б.п.**, но min rate
0,940. Проблема не в качестве, а в cold start: до 2024 hard pool содержит меньше
100 зрелых строк.

AP26 вводит заранее заданное shrinkage к global CatBoost. Вес specialist равен
n/(n+200), где n - число доступных hard-pool train rows на текущем OOS origin.
Никакой late metric в вес не входит.

| AP26 y20 shrink200 | h3 | h5 | h10 | h20 |
|---|---:|---:|---:|---:|
| Adjusted lift | **2,417** | **2,483** | **2,446** | **2,495** |
| H5 signals |  | **674** |  |  |
| H5 symmetric, б.п. |  | **74,45** |  |  |
| H5 future-only, б.п. |  | **135,31** |  |  |

Это лучший новый accuracy-frontier и сильнее AP21 по каждой точечной метрике.
Однако min currency rate по всем неизвестным горизонтам 0,971, поэтому нельзя
выдавать AP26 за строгое продуктовое решение. Его роль - high-quality anchor,
которому нужен очень редкий causal backstop.

---page---

## 4. AP27: как вернуть cadence почти без потери качества

AP27 оставляет AP26 y20 shrink200 основным pace score и добавляет CatBoost
backstop только если specialist не сработал, trailing365 rate ниже 0,95,
reserve в top30%, а CatBoost rank выше top40%. Primary и недельный cap не меняются.

На позднем периоде backstop добавил 26 решений. Итоговый h5: 691 сигнал,
adjusted lift **2,465**, symmetric **73,67 б.п.**, future-only **131,79 б.п.**
Минимальная rate по h3/h5/h10/h20 равна **1,001**, пустых месяцев нет.

Против AP21 point delta по h3/h5/h10/h20:
+0,003 / +0,018 / +0,016 / +0,056. Все paired 20- и 50-date CI пересекают
ноль. Против AP17 symmetric h5 прирост +4,93 б.п. имеет положительный CI, но
future-only и lift отдельно не доказаны.

### Честные статусы

- AP23 competence pace - лучший строгий point-frontier: min lift 2,407.
- AP26 y20 shrink200 - лучший accuracy-frontier: min lift 2,417, rate fail.
- AP27 r60 - specialist strict late challenger: min lift 2,405, min rate 1,001.
- AP27 early selector выбрал r70; он получил min lift 2,413, но min rate 0,986.
- Ни один новый вариант не является fresh independent holdout winner.

### Аудит и следующий шаг

AP23 проверяет maturity, expert precision, counts, weights и corruption prefix.
AP24 дважды пересобирает 85 ranker fits, включая испорченное будущее. AP25
пересобирает hard pool и 85 specialist fits. AP26-AP27 независимо восстанавливают
counts, weights, ranks, причины и state. Полный suite: **271 тест, все прошли**.

Следующий корректный шаг - заморозить AP23 и AP27 рядом с AP21, не менять
thresholds и собирать prospective shadow. Параллельно можно улучшать specialist
только через заранее заданные low-data методы: hierarchical pooling или Bayesian
shrinkage, без нового grid по открытому 2024-2026. Отдельно требуется подтвердить
фактический timestamp публикации и исполнимый банковский курс: текущий benefit
измеряется относительно официального ЦБ, а не клиентской сделки.
