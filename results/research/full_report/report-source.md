# Полный отчёт по прогнозированию выгодного курса

Состояние исследования: 04.09.2026. Ветка отчёта: `ivan-experiments`.

Этот документ объединяет результаты трёх веток и явно сохраняет происхождение
каждого подхода:

- `main` - исходная продуктовая постановка, базовые правила и классические ML;
- `version_b` - отдельный rolling benchmark Даниила с logistic regression;
- `ivan-experiments` - расширенная история 2010-2026, leakage-safe протокол,
  режимы, ансамбли, внешние данные и conditional publication models;
- `version_b + ivan audit` - повторная проверка модели из `version_b` с
  причинными порогами.

## 1. Итог в нескольких строках

До публикации следующего курса ЦБ у нас пока нет универсальной модели, которая
доказанно держит lift выше `1.30` во всех режимах при частоте 1-2 сигнала на
валюту в неделю.

| Роль | Подход | Lift / частота | Источник | Статус |
|---|---|---:|---|---|
| Честный baseline до публикации | multiscale range anchor | 1.295 / 1.02 | `ivan-experiments` | locked |
| Объяснимый recent challenger | causal logistic regression | 1.307 / 1.11 | `version_b + ivan audit` | causal retrospective |
| Ровный ML после 2022 | reset XGBoost | 1.288 / 1.07 | `ivan-experiments` | challenger |
| Максимум past-only | Online local Hedge | 1.434 / 1.63 | `ivan-experiments` | retro, нестабилен |
| После публикации | conditional ExtraTrees | 2.459 / 1.07 | `ivan-experiments` | selected on earlier blocks |
| Окно закрывается | upper-range rule | 1.182 / 1.46 | `ivan-experiments` | не проходит 1.30 |

Основная продуктовая развилка - момент принятия решения. Если новый курс ЦБ на
завтра уже опубликован, но сегодняшний клиентский курс ещё доступен, conditional
модель намного сильнее. Это отдельный information set, а не улучшение обычного
предварительного прогноза.

## 2. Данные и target

Основные валюты: TJS, UZS, KGS, AMD, KZT. Контекст: USD и CNY. В отдельных
экспериментах проверялись RUONIA, ключевая ставка, Brent и широкий долларовый
индекс.

Длинная история охватывает 2010-01-01 - 2026-09-02 и содержит около 4.1 тыс.
публикаций на валюту. Единица времени - публикация Банка России, не календарный
день. Выходные не размножаются forward-fill строками; календарный разрыв хранится
отдельным past-only признаком.

Основной target:

```text
fav_h5(t) = 1, если v[t] <= min(v[t+1], ..., v[t+5])
```

Меньший курс RUB/FX выгоднее отправителю рублей. `h=5` означает пять публикаций
ЦБ, не пять календарных дней.

Lift:

```text
hit rate среди сигналов / base rate среди всех допустимых OOS-строк
```

Lift 2.459 означает, что вероятность удачного момента среди сигналов в 2.459
раза выше случайного выбора с тем же тестовым scope. Это не доходность 145.9%.
Денежный эффект считается отдельно в базисных пунктах.

## 3. Протокол честности

- Development: до 2016.
- General validation: 2017-2020.
- Transition/calibration: 2021.
- Shock validation: 2022-2023.
- Retrospective final: 2024-2026.
- h=5 label допускается в train только после фактической пятой будущей
  публикации.
- Порог каждой валюты строится по предыдущему calibration периоду.
- Rolling threshold использует только прошлые scores с `shift(1)`.
- Частота проверяется фактически, а не задаётся по будущему test quantile.
- Block bootstrap ресемплирует четырёхнедельные блоки целиком.

Термины:

- `locked` - конфигурация зафиксирована до чтения оцениваемого блока;
- `causal retrospective` - каждый сигнал воспроизводим онлайн, но период уже
  просмотрен;
- `posthoc` или `retro max` - победитель выбран после просмотра test;
- `non-causal threshold` - решение использует будущую шкалу test scores.

Причинные признаки сами по себе не делают backtest честным. Утечка может
возникнуть при выборе порога, окна, гиперпараметров или лучшего результата.

## 4. Ветки и различия протоколов

### `main`

Содержит исходную продуктовую систему, правила уровня/моментума/разворота,
logistic regression, random forest, gradient boosting, CatBoost и XGBoost.
Для h=5 logistic regression показывает около 1.14 при частоте 1.22. Простое
upper-range правило достигает около 1.39, но частота только 0.60.

### `version_b`

Автор ветки по git history - Daniil Nedaiborsch. Используется 36-месячное
rolling обучение и шестимесячные test-folds. Logistic regression заявляет mean
lift 1.450, однако сигналы выбираются как top 15% внутри уже полностью известного
test-fold. Это хорошая rank diagnostic, но не воспроизводимая online policy.

### `ivan-experiments`

Содержит четыре волны исследования: длинные временные блоки, EDA, модели рядов,
глобальные/локальные ML, residual towers, rankers, external data, recency,
mixtures, state models, post-publication и window-closing контуры, bootstrap и
circular-shift multiplicity audit.

Цифры из разных веток нельзя сравнивать без оговорок: отличаются начало
истории, train window, calibration и способ агрегирования валют.

## 5. Главные закономерности данных

1. Общий фактор движений пяти валют очень силён. PC1 объясняет около 70% вариации
   в 2017-2020, 85% в 2022-2023 и 92% в 2024-2026. Это поддерживает global
   pooling.
2. После 2022 заметен структурный сдвиг трендов и волатильности. Однако жёсткая
   бинарная дата слабее rolling/decay/reset.
3. Сезонность нестабильна. Месяцы меняют порядок между эпохами и валютами;
   sin/cos и праздники остаются слабыми дополнительными признаками.
4. Соседние h=5 targets перекрываются четырьмя будущими точками. Iid bootstrap
   занижает неопределённость.
5. В данных нет клиентского спроса, комиссий и фактического курса банка. Поэтому
   нельзя доказать гипотезу о росте спроса после ограничений SWIFT.

Самые полезные past-only признаки:

- положение курса в диапазонах 30/90/180 публикаций;
- ret/trend 5/20/60 и расстояния до прошлых экстремумов;
- volatility, volatility ratio, streak up/down;
- currency identity, общий фактор и отклонение валюты от него;
- past-only USD/CNY, calendar gap и слабый календарный контекст.

## 6. Past-only лидеры

| Подход | Lift final | Freq | Min year | Future bps | Источник / статус |
|---|---:|---:|---:|---:|---|
| Multiscale anchor | 1.295 | 1.02 | 0.508 | +35.0 | `ivan`, locked |
| Causal logit rolling q20 | 1.307 | 1.11 | 1.050* | +51.8 | `version_b + audit`, retro |
| Reset XGBoost | 1.288 | 1.07 | 1.188 | +32.7 | `ivan`, current regime |
| Geometric consensus | 1.264 | 1.11 | 1.138 | +33.4 | `ivan`, challenger |
| Soft regime router | 1.325 | 1.54 | 1.161 | +32.0 | `ivan`, retro |
| Trend anchor | 1.406 | 1.03 | 1.227 | +47.4 | `ivan`, posthoc |
| Online local Hedge | 1.434 | 1.63 | 1.043 | +42.2 | `ivan`, retro unstable |

`*` Для causal logit min включает 2022; на 2024-2026 minimum aggregate year
равен 1.102.

Multiscale anchor:

```text
0.5 * pct_range_90 + 0.3 * pct_range_30 + 0.2 * pct_range_180
```

Он максимально объясним и строго past-only. Слабое место - 2025 год. Trend
anchor достигает 1.406 на final, но его формула появилась после просмотра final
и не переносится на general/shock. Online Hedge достигает 1.434 aggregate, но
частота по годам равна 3.09, 0.48 и 1.33; macro-year lift только 1.251.

Reset XGBoost - наиболее ровный современный ML: 2024/2025/2026 lifts примерно
1.290/1.188/1.437 и frequency около 1.02-1.17.

## 7. Объяснимость

| Семейство | Уровень | Причина |
|---|---|---|
| Range anchor, known-next gate | очень высокий | точная формула |
| Logistic regression | высокий | знаки и стандартизированные коэффициенты |
| Empirical Bayes / Markov | средне-высокий | вероятности состояний и переходов |
| ExtraTrees / XGBoost | средне-низкий | feature importance/SHAP, но нет одной формулы |
| Equal/geometric ensemble | средний | согласие нескольких экспертов |
| Learned router / Hedge | низкий | динамические веса по режимам и ошибкам |
| GRU | низкий | скрытое состояние, прироста нет |

## 8. Честный аудит `version_b`

Исходный commit: `aa44f10a47bb9bd72379331bd4596eab3c4944b0`.

| Политика на истории 2019-2026 | Mean lift | Aggregate | Freq | Min FX |
|---|---:|---:|---:|---:|
| Future test top-15% | 1.450 | 1.452 | 0.73 | 1.319 |
| 30m fit / 6m calibration / fixed q20 | 1.347 | 1.366 | 1.08 | 1.212 |
| 30m fit / 6m calibration / rolling120 q20 | 1.307 | 1.306 | 1.11 | 1.198 |
| Past OOF rolling120 q30 | 1.335 | 1.344 | 1.29 | 1.160 |

На длинной истории 2013-2026:

| Политика | Mean lift | Freq | 95% aggregate CI |
|---|---:|---:|---:|
| Future top-15%, diagnostic | 1.319 | 0.73 | [1.188; 1.454] |
| Past OOF expanding q20 | 1.225 | 1.04 | [1.099; 1.370] |
| Nested fixed q20 | 1.085 | 1.16 | [0.982; 1.202] |
| Nested rolling120 q20 | 1.137 | 1.09 | [1.038; 1.252] |

Вывод: ranker содержит recent-regime signal, но не подтверждает стабильные 1.45.
Для будущего теста полезен замороженный causal rolling q20 challenger.

## 9. После публикации следующего курса

В этом контуре v[t+1] уже опубликован. Если он ниже v[t], target fav_h5 уже
невозможен. Если выше, известен запас перед оставшимися четырьмя неизвестными
публикациями.

| Абляция | Lift | Freq | Future bps | Источник |
|---|---:|---:|---:|---|
| Known-next gate | 1.959 | 1.369 | +77.8 | `ivan` |
| Known margin only | 2.342 | 1.139 | +127.4 | `ivan` |
| Gate + past-only logit | 2.363 | 1.080 | +112.5 | `ivan` |
| Published t+1 logit | 2.493 | 1.066 | +135.5 | `ivan` |
| t+1 + margin logit | 2.515 | 1.080 | +137.9 | `ivan` |
| Logit + ExtraTrees | 2.553 | 1.076 | +142.0 | `ivan`, retro finalist |

Selected conditional ExtraTrees:

- 2017-2020: lift 2.632, frequency 1.055;
- 2022-2023: lift 2.395, frequency 1.090;
- 2024-2026: lift 2.459, frequency 1.069;
- final 95% CI: [2.160; 2.797];
- minimum lift по валюте: 2.371;
- circular-shift max-adjusted p: 0.00025.

Главный feature - `known_margin_vol`, то есть известная разница v[t+1]-v[t],
нормированная на недавнюю волатильность. Далее идут размер общего объявленного
движения и позиция нового курса в коротком диапазоне.

Операционная граница: production обязан проверять фактический timestamp
публикации. Кроме того, бизнес должен подтвердить, что сегодняшний клиентский
курс ещё доступен после появления завтрашнего официального курса.

## 10. Window-closing и две цены дня

`close_h5(t)=1`, если `v[t+5] > v[t]`.

| Подход | Lift | Freq | Future bps | Статус |
|---|---:|---:|---:|---|
| Trend anchor | 1.132 | 1.629 | +25.0 | selected, не проходит |
| Upper-range | 1.182 | 1.458 | +32.1 | retro best |
| ExtraTrees | 1.162 | 1.072 | +18.0 | не проходит |

Future-only выгода измеряет достижимое преимущество относительно будущего.
Симметричная +/-5 метрика включает прошлые публикации и может награждать уже
запоздалый сигнал. Например, locked anchor имеет +35.0 bps future-only, но -26.9
bps по симметричной метрике; known-next gate +77.8 и +48.3 bps соответственно.

## 11. Полный журнал основных семейств

| Семейство | Лучшее наблюдение | Почему не финал | Источник |
|---|---:|---|---|
| Seasonal naive | AUC 0.458 | сезонность не переносится | `ivan` |
| ETS | AUC 0.478 | уровень плохо соответствует barrier target | `ivan` |
| SARIMA | AUC 0.523 | слабый сигнал | `ivan` |
| GRU | AUC 0.540 | сложность не окупилась | `ivan` |
| Global ExtraTrees | shock 1.367 | final 1.254 | `ivan` |
| XGB ranker | general до 1.375 | shock/final около 1.0 | `ivan` |
| Local logit -> global residual XGB | shock 1.075 | перенос старых ошибок | `ivan` |
| Short-window ExtraTrees mix | shock 1.384 | frequency 0.864 | `ivan` |
| Equal mix 6 experts | final 1.328 | min year 0.882 | `ivan` |
| Soft regime router | final 1.325 | retrospective | `ivan` |
| External macro data | shock до 1.03 | lagged proxies не помогают h5 | `ivan` |
| Cross-sectional rank | general 1.371 | shock 1.038 | `ivan` |
| Per-currency champions | general 1.342 | shock 0.888 | `ivan` |
| Pooled hazard | final 1.206 | ошибки шагов накапливаются | `ivan` |
| Empirical Bayes states | selected final 1.228 | ниже 1.30 | `ivan` |
| Markov + anchor | retro 1.316 | frequency 0.961 и слабые эпохи | `ivan` |

## 12. Что дало максимальный прирост

1. Изменение information set: известный v[t+1].
2. `known_margin_vol` в post-publication модели.
3. Multiscale положение в прошлых диапазонах 30/90/180.
4. Global pooling пяти валют благодаря общему фактору.
5. Recency/reset вместо жёсткой бинарной даты 2022.
6. Простая смесь независимых экспертов вместо сложного learned router.
7. Причинный rolling threshold и контроль годовой частоты.

## 13. Статистический вывод

- Locked anchor: CI [0.964; 1.596]; превосходство над random нестрого на 95%.
- Все новые ordinary модели имеют CI разности с anchor, пересекающий ноль.
- version_b causal recent: интервалы пересекают 1.30.
- version_b causal long: стабильное превышение 1.30 не подтверждается.
- Post-publication selected: CI [2.160; 2.797], adjusted p=0.00025.
- Window closing: best CI [1.021; 1.343], adjusted p=0.165.

В ordinary поиске записано 157 политик. Выбор максимума после просмотра test
создаёт selection bias, который не исправляется обычным bootstrap. Поэтому
headline должен быть заранее замороженным результатом.

## 14. Рекомендуемая система

### До публикации

- основной benchmark: multiscale anchor;
- challengers: causal logistic regression `version_b`, reset XGBoost,
  geometric consensus;
- для новых данных фиксировать features, model, q20/rolling120 и per-currency
  thresholds.

### После публикации

- основной: conditional ExtraTrees 7y, q22, rolling250;
- challenger: logit + ExtraTrees;
- обязательный actual-publication timestamp gate.

### Window closing

- production winner пока отсутствует;
- upper-range rule оставить только для monitoring.

## 15. Пилот и следующий holdout

1. Считать frozen policies параллельно, но не перенастраивать их.
2. Хранить publication timestamp, scoring timestamp, показ сигнала и
   фактический клиентский курс.
3. До продуктового запуска получить историю spread, комиссии и доступности
   текущего курса после публикации.
4. На заранее назначенной дате сравнить lift, frequency, future bps, minimum
   year/currency и бизнес-конверсию.
5. После 04.09.2026 не менять параметры по новым данным, иначе следующий период
   снова станет development, а не holdout.

## 16. Воспроизводимость

Основные материалы:

- `EXPERIMENTS_SUMMARY.md`;
- `results/research/round2/report-source.md`;
- `results/research/round3/report-source.md`;
- `results/research/round4/report.md`;
- `results/research/version_b_honest_audit/report.md`;
- `research/round4_research.py`;
- `research/version_b_honest_audit.py`;
- полные CSV grid, bootstrap, protocol JSON и сохранённые OOF outputs в
  `results/research/`.

Все 56 автоматических тестов прошли.

Внешние ссылки:

- условия кейса: <https://talenttrack.aitalenthub.ru/hackathon/cases/455>;
- динамика курсов ЦБ: <https://www.cbr.ru/currency_base/dynamics/>;
- срок действия курса: <https://www.cbr.ru/faq/dkp/04/>;
- время публикации: <https://www.cbr.ru/Reception/TopicalMessage/Page/2661>.
