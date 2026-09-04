# Claim-source ledger

| Утверждение | Источник | Тип | Использование |
|---|---|---|---|
| Официальные курсы и номиналы валют | https://www.cbr.ru/currency_base/dynamics/ | первичный | построение ряда и нормировка |
| Официальный курс обычно публикуется до 18:00 МСК, точное время не гарантируется | https://www.cbr.ru/Reception/TopicalMessage/Page/2661 | первичный | ограничение after-publication policy |
| Формулировка кейса и две метрики | https://talenttrack.aitalenthub.ru/hackathon/cases/455 | первичный | target и отчётность |
| Предсказуемость валют зависит от sample, horizon и evaluation | https://doi.org/10.1257/jel.51.4.1063 | peer reviewed | интерпретация нестабильности |
| Параметрическая нестабильность ухудшает exchange-rate forecasts | https://doi.org/10.1017/S1365100506050085 | peer reviewed | мотивация regime/recency |
| Global forecasting может выигрывать за счёт общей оценки связанных рядов | https://doi.org/10.1016/j.ijforecast.2021.03.028 | peer reviewed | pooled/global модели |
| Delayed feedback требует задержанного обновления online learner | https://proceedings.mlr.press/v28/joulani13.html | peer reviewed | causal Hedge/SGD |
| Delay-aware online learning допускает optimism/corrections | https://proceedings.mlr.press/v139/flaspohler21a.html | peer reviewed | альтернативы online weights |
| Concept drift может сделать старую модель неточной | https://proceedings.mlr.press/v139/tahmasbi21a.html | peer reviewed | reset и забывание |
| Выбор rolling/averaging зависит от масштаба и частоты breaks | https://doi.org/10.1007/s00181-021-02137-w | peer reviewed | смеси окон |
| First-passage formulation соответствует событию пересечения барьера | https://arxiv.org/abs/1107.1174 | working paper | path/barrier модели |
| Discrete survival раскладывает вероятность по условным hazards | https://doi.org/10.1093/acprof:oso/9780195337518.003.0003 | academic book | pooled hazard |
| Массовый перебор требует поправки data snooping | https://doi.org/10.1111/1468-0262.00152 | peer reviewed | multiplicity audit |
| SPA оценивает superior predictive ability | https://doi.org/10.1198/073500105000000063 | peer reviewed | ограничение значимости |
| Model Confidence Set сохраняет несколько неразличимых моделей | https://doi.org/10.3982/ECTA5771 | peer reviewed | вывод о нескольких challengers |
| Conditional predictive ability формализует состояние-зависимое сравнение | https://doi.org/10.1111/j.1468-0262.2006.00718.x | peer reviewed | regime router |
| Все численные результаты моделей | локальные CSV в results/research и results/research/round3 | первичный расчёт | таблицы и выводы отчёта |

