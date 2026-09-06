# Итоги всех модельных экспериментов

Короткий навигатор по исследованию. Полные таблицы, отрицательные результаты,
протоколы и сохранённые прогнозы не удалены: этот файл помогает понять, куда
смотреть в первую очередь.

Навигатор обновлён 06.09.2026; последний модельный прогон — 06.09.2026.
Для внутренней оптимизации сохраняется
`h=5`, официальный scorecard теперь считается сразу на `h=1/3/5/10/20`.

## T3-T16: непрерывная температура от утра до 23:00

Презентация из основной ветки формулирует продукт как экранный индикатор плюс
отдельное редкое уведомление. T16 причинно объединил фазовые модели в один
API-контракт: `temperature_0_100`, вероятности h=1/3/5/10/20, ожидаемые
future-only б.п. ЦБ, `last_source_at`, freshness, phase, confidence и
`push_now`. Ретроспективный пакет содержит **60 370** уникальных временных
снимков и позволяет получить последнее допустимое состояние по каждой из пяти
валют в любой календарный день. На выходных значение удерживается, но явно
становится stale; future-snapshot corruption не меняет прошлые ответы.

T9 проверил формальное начало биржевой сессии вместо предположения. В современном
режиме 2025–2026 same-day CNY-покрытие до 09:30 оказалось ровно **0%**, поэтому
рисовать «обновление рынком» утром было бы неверно. К 10:00 покрытие достигает
**79,2%**. T10 выпускает снимок только на этих строках и иначе точно сохраняет
T4. Для h5 на 2025–2026 Brier улучшился **0,1951 -> 0,1716**, AUC
**0,5547 -> 0,7301**; 20- и 50-date paired bootstrap Brier-delta полностью
ниже нуля. Улучшение h1/h3 также устойчиво на 2024 и 2025–2026; h20 слабее.

T13 нашёл источник для более раннего окна: завершённые часовые perpetual
фьючерсы CNYRUBF/USDRUBF. В 09:00 causal CNY basis-rank на screen-2024 улучшил
Brier h1 **0,24365 -> 0,22530**, h3 **0,24547 -> 0,23147**. Оба 20/50-date
интервала полностью ниже нуля. На открытом 2025–2026 h1 даёт Brier/AUC
**0,20239 / 0,75254** против **0,24496 / 0,58719**, h3 —
**0,18983 / 0,73365** против **0,21726 / 0,58615**. H5 на позднем периоде
также заметно лучше, но оба screen-интервала пересекают ноль, поэтому
предзарегистрированный gate его отклонил. В 10:00 ни один perpetual-кандидат
устойчиво не обошёл уже сильный T10.

T14 добавил 1 930 физических 09:00-снимков только для h1/h3. H5/h10/h20 на
том же snapshot остаются T4, а expected future-only bps — mature prior. Чтобы
интерфейс не выдавал старый h5 за обновлённый рынком, API теперь предпочитает
`source_at/source_kind/phase/confidence` конкретного горизонта. Все прежние
строки T12 и push-счётчики остались неизменны.

Качество неодинаково по времени суток. T4 history-only до появления рыночных
данных даёт на 2025–2026 AUC около **0,594/0,604/0,596** для h1/3/5: это
полезный fallback, но не сильный торговый сигнал. T5 последовательно использует
только завершённые свечи; средний AUC по пяти горизонтам растёт с **0,757** в
10:30 до **0,785** в 15:30, Brier снижается с **0,1562** до **0,1477**. T3
закрывает промежуток 15:30–появление нового курса без чтения будущих свечей.
T6 улучшает средний MAE ожидаемой future-only выгоды с **135,26** б.п. у
исторического prior до **120,45** в 15:30 и **120,17** в 16:30; premarket
регрессия prior не обошла и потому должна использоваться осторожно. Более того,
все чистые pre-receipt Ridge были нестабильны на 2024. T11 исправляет это
причинным квартальным регулятором: вес 0/.25/.5/.75/1 выбирается только по
созревшим прошлым MAE. На 15:30 screen-2024 MAE h1/h3/h5/h10 стал
**54,50/81,13/98,13/126,06** против prior
**61,71/87,70/102,65/131,77**; на 2025–2026 —
**55,32/84,85/108,96/148,19** против
**70,54/97,51/119,39/159,11**. h20 и интервалы до 10:00 остаются на prior.

AP50 калибрует четыре after-publication head: общий Brier становится
**0,165/0,163/0,164/0,133** на h3/5/10/20. AP51 снижает MAE ожидаемой выгоды до
**56,65/85,91/132,68/197,08** б.п. против prior
**93,80/113,06/148,78/206,96**. T7 был честно инвалидирован: сохранённая
after-publication точка уже соответствовала 18:30, поэтому добавленные якобы
«новые» свечи пересекались с её информацией. Исправленный T7B использует только
свечи, впервые доступные после 18:30: обновление 19:00 немного улучшает средний
Brier с **0,1507 до 0,1492** и AUC с **0,7124 до 0,7161**; 20:00 не улучшает
ранжирование.

Push остаётся отдельным решением AP37 с max2/week. В T12 сохранено
AMD/KGS/KZT/TJS/UZS = **142/139/138/139/137** сигналов за весь открытый
период — примерно один на валюту в неделю. Это не означает, что вся высокая
температура порождает уведомление.

T15 проверил вечерние completed-prefix 20:00/21:00/22:00/23:00 по
CNYRUBF и USDRUBF. Физическое dual-покрытие составляет 100% в 2024, 2025 и
2026. Ни `perp_cny_logit`, ни `perp_dual_logit`, ни `perp_dual_hgb` не прошли
оба 20/50-date screen-Brier gate против T7B, поэтому новые probability-heads не
приняты. Это фиксирует честный отрицательный результат: новая свеча не обязана
двигать температуру.

Для expected benefit dual residual Ridge прошёл оба screen-MAE gate в семи
комбинациях: 20:00 h3, а в 21:00/22:00/23:00 h3 и h5. На screen-2024 в 23:00
MAE стал **53,14 / 80,85** против **58,08 / 87,88** б.п. На открытом
2025–2026 эффект меньше, но обычно того же знака. T16 добавил 9 855 вечерних
снимков и отдельные `benefit_last_source_at`, age, freshness и source kind.
Все probability и push-решения сохранены точно; h10/h20 остаются на control.

[Контракт интерфейса](research/interface_prediction_contract.md) ·
[T16 protocol](research/temperature_t16_evening_router_registered.md) ·
[T16 audit](results/research/temperature/t16_evening_router/audit_checks.json) ·
[финальный алгоритм простыми словами](output/pdf/ivan_final_anytime_algorithm_for_everyone.pdf) ·
[PDF any-time checkpoint](output/pdf/ivan_continuous_temperature_anytime.pdf) ·
[финальная презентация с интерфейсом](output/presentation/international_transfers_final_with_interface_2026-09-06_v2.pptx).
Финальный T16-прогон и полный набор из **328 тестов** прошли.

## Последние AP37-E/AP39-E: mature precision даёт новый strict point

[PDF](output/pdf/ivan_after_publication_ap39_effective.pdf) ·
[отчёт](research/after_publication_ap39_effective_report.md) ·
[AP37 protocol](research/after_publication_ap37_effective_registered.md) ·
[AP38 protocol](research/after_publication_ap38_effective_registered.md) ·
[AP39 protocol](research/after_publication_ap39_effective_registered.md).

AP37 сохранил frozen AP26 core и изменил только редкий AP23 calendar fallback.
Три causal OOS rank - AP26 y20, AP34 residual survival и AP35 distributional
CatBoost - образуют support stratum 0..3 по существующей границе top30. Для
каждого stratum expanding precision строгого `min(y3,y5,y10,y20)` оценивается
только на publication-h20-mature прошлом с двухдневным embargo и трёхуровневым
shrinkage global/stratum/currency. Late-week fallback проходит при local
precision не хуже causal overall; silence10 rescue не блокируется.

AP37 прошёл ранние и поздние strict gates. Late h3/h5/h10/h20=
**2,428674/2,509188/2,479316/2,524987**, min rate **1,008734**, zero empty
months,max2/week. H5:695 сигналов,rate1,038,currency1,023–1,061,
symmetric74,55б.п.,future-only133,73б.п. Он сохранил693 решения AP33, удалил11
и добавил2; итог —673 core,1 precision late-week,21 silence fallback. Deltas
lift к AP33 +.0082/+.0192/+.0084/+.0087, но все 20/50-date CI пересекают ноль.
Это новый frozen strict point leader, не fresh production winner.

AP38 применил ту же зрелую precision к core и разрешил veto при текущей
trailing rate>=1. Accuracy выросла до h5 **2,520964**, h20 **2,597393**;
h20 delta к AP37 +.0724 имеет 20-date CI **[+.0065;+.1543]**. Но min rate
упала до **0,955240**, ранний selector дал minrate.856 и два пустых месяца.
AP38 - только accuracy upper bound.

AP39 заранее задал один week runway: veto только если частота останется>=1 при
добавлении недели к знаменателю. Late **2,434597/2,508938/2,478471/2,572246**,
средняя h5 rate.9996, но min currency h5.9787 и all-h minrate.9705. Ранний
cadence также провален. Дальше подбирать runway по открытому scorecard нельзя.

Все AP37-AP39 audits восстановили maturity, support, precision, shrinkage,
rate/silence/veto/runway/cap и future-corruption prefix. Полный набор -
**289 тестов**. Следующий новый класс: одна preregistered causal weekly
optimal-stopping модель, а не настройка support/shrinkage/runway.

## AP34-E/AP36-E: residual-модели и Hedge не обошли AP33

[PDF](output/pdf/ivan_after_publication_ap36_effective.pdf) ·
[отчёт](research/after_publication_ap36_effective_report.md) ·
[AP34 protocol](research/after_publication_ap34_effective_registered.md) ·
[AP35 protocol](research/after_publication_ap35_effective_registered.md) ·
[AP36 protocol](research/after_publication_ap36_effective_registered.md).

AP34 заменил бинарный direct y20 на continuous residual survival. Target равен
минимальной будущей effective-просадке за вычетом уже известного объявленного
изменения. В каждом из 17 quarterly OOS fit standardized Ridge обучается только
на зрелом прошлом; train-only residual CDF и причинный per-currency error-state
превращают прогноз в survival score. Late h3/h5/h10/h20=
**2,411500/2,468417/2,427785/2,432848**, min rate **1,046943**, zero empty
months, max2/week. AP34 проходит strict point-gates, но уступает AP33 на каждом
горизонте; все delta CI пересекают ноль.

AP35 проверил принципиально другую постановку: monotone distributional CatBoost.
Каждая зрелая train-строка расширена на пять заранее заданных anchor-порогов
−200/−100/0/+100/+200 б.п.; классификатор учит условную CDF residual floor, а
positive monotonic constraint делает score согласованным с запасом. Late
**2,391461/2,449391/2,414259/2,402068**, min rate **1,039301**. H3 gate
провален; потеря h20 к AP33 −0,114 имеет 20-date CI **[−0,252; −0,021]**.

AP36 объединил AP26 direct y20, AP34 survival и AP35 distributional score через
causal mature-only Hedge. Same-currency ranks считаются до текущей строки, а
trailing-730 Brier loss видит только y20, полностью созревшие не менее двух дней
назад. Средние late weights: AP26 **58,8%**, AP34 **32,1%**, AP35 **9,1%**.
Late **2,401057/2,448897/2,436977/2,417616**, min rate **1,031659**. Strict
point-gates пройдены, но h5 и h20 статистически хуже AP33: delta CI
**[−0,098; −0,004]** и **[−0,207; −0,026]**.

Вывод: знание завтрашнего курса действительно полезно, но узкое место уже не
средняя calibration. AP26 выигрывает редким точным decision-tail, а continuous
floor, synthetic CDF и average Brier добавляют слишком много средних pace-точек.
AP33 остаётся frozen strict leader. Все независимые fit/maturity/state/
monotonicity/weight/future-prefix audits прошли; полный набор — **284 теста**.

## AP33-E: календарный fallback улучшает strict frontier

[PDF](output/pdf/ivan_after_publication_ap33_effective.pdf) ·
[отчёт](research/after_publication_ap33_effective_report.md) ·
[protocol](research/after_publication_ap33_effective_registered.md) ·
[results](results/research/after_publication/ap33_effective).

AP33 проверил ровно одну заранее зарегистрированную decision-policy. Frozen AP26
core имеет приоритет; frozen AP23 fallback разрешается только при trailing365
rate ниже1, после84-дневного warmup и либо в четверг/пятницу без сигнала текущей
ISO-недели, либо после10 дней причинного молчания. Новый поток снова ограничен
max2/week. Ни outcomes, ни будущая часть недели, ни score models не читаются.

Политика прошла early2023 joint gate и поздние strict gates. Late
h3/h5/h10/h20=**2,420521/2,489967/2,470952/2,516300**, min lift
**2,420521**, mean lift **2,474435**, min currency rate **1,008734**, zero
empty months,max2/week. H5:704 сигнала,rate1,051868,currency1,02348–1,06830,
hit72,73%,symmetric74,60б.п.,future-only132,94б.п. Причины:673 core,
18 late-week fallback,13 silence-10 fallback.

Против AP32 lift-delta h3/h5/h10/h20 равна
+0,0174/+0,0211/+0,0153/+0,0271, но все paired lift-CI пересекают ноль.
Future-only delta к AP32 на h3 +1,87б.п. CI[+0,30;+4,57] и h5 +2,80б.п.
CI[+0,28;+7,28] положительна на20-date и сохраняется на50-date blocks.
Ранние AP33/AP32/AP26 decisions совпадают, поэтому early pass не доказывает
преимущество calendar-filter. AP33 — новый лучший strict point и frozen shadow
challenger, но не independent holdout winner. Audit прошёл; полный набор —
**279 тестов**.

## Последние AP28-E/AP32-E: ансамбль на уровне решений

[PDF](output/pdf/ivan_after_publication_ap32_effective.pdf) ·
[отчёт](research/after_publication_ap32_effective_report.md) ·
[AP32 protocol](research/after_publication_ap32_effective_registered.md) ·
[AP32 results](results/research/after_publication/ap32_effective).

AP28 проверил один заранее заданный hierarchical pooled y20 CatBoost: все
зрелые строки получили вес1, а outcome-free hard-pool — вес4. Общий prior убрал
cold start, но размыл редкий сигнал: late h3/5/10/20=
**2,408359/2,457002/2,411398/2,377662**, min rate **0,978166**. Ранний gate
не пройден.

AP29 добавлял AP23 только как mature-competence backstop к AP26 core. Частота
восстановилась до min rate **1,001092**, но h3=**2,397969**, поэтому строгий
lift-gate не пройден. AP30 смешивал AP26/AP23 только в причинном rank-space
75/25; это также размыло core: min lift **2,397866**, h5 **2,441546**.
AP31 разложил событие на `P(y3) × P(y20|y3=1)` через 34 quarterly OOS
CatBoost fits. Интерпретируемая survival-факторизация прошла ранний selector,
но compounded calibration error дала min lift **2,333349**, min rate **0,993450**.

AP32 впервые ансамблирует только бинарные решения. AP26 y20-shrink200 имеет
приоритет; если он молчит и собственный trailing365 rate meta-router ниже1,
разрешается AP23 soft730 decision, после чего применяется новый sequential
max2/week. Это одна заранее зарегистрированная политика без позднего grid.
Она прошла ранний selector и поздние strict gates: h3/5/10/20=
**2,403143/2,468871/2,455698/2,489157**, min lift **2,403143**, mean lift
**2,454217**, min currency rate **1,039301**, zero empty months, max2/week.
На h5:720 сигналов,rate1,075774,currency1,05336–1,09072,symmetric73,46б.п.,
future-only130,14б.п.; причины —673 AP26 core и47 AP23 fallback.

Относительно AP23 изменения h3/h5/h10/h20 равны
−0,00349/+0,00282/−0,00881/+0,01710; все 20/50-date CI пересекают ноль.
Поэтому AP23 сохраняет лучший strict minimum **2,406634**, AP26 — accuracy
frontier **2,417356** с провалом rate, а AP32 становится новым early-selected
strict Pareto challenger, но не fresh independent winner. Все AP28–AP32 audits
прошли; полный набор — **277 тестов**.

## Последние AP23-E/AP27-E: mature competence, specialists и causal backstop

[PDF](output/pdf/ivan_after_publication_ap27_effective.pdf) ·
[отчёт](research/after_publication_ap27_effective_report.md) ·
[AP23 protocol](research/after_publication_ap23_effective_registered.md) ·
[AP27 protocol](research/after_publication_ap27_effective_registered.md) ·
[AP27 results](results/research/after_publication/ap27_effective).

AP23 проверил пять заранее зарегистрированных competence-router: веса экспертов
оцениваются по top30 precision на mean(y3,y5,y10,y20), но только по строкам, чей
h20 полностью созрел раньше текущей даты минимум на два дня. Global precision
стягивается к 0,5, currency precision — к global; primary и state AP21 остаются
неизменными. Главный результат: адаптация primary ухудшает min lift до 2,33–2,35,
а адаптация только pace полезна. Лучший строгий late point `soft730_pace`:
**2,406634/2,466054/2,464507/2,472062** на h3/5/h10/h20, min rate
**1,039301**, zero empty months, max2/week; h5 symmetric **73,82 б.п.**,
future-only **131,01 б.п.**. Он не прошёл early cadence (0,985) и потому является
late challenger, не early-selected winner. Все paired lift-CI против AP21
пересекают ноль.

AP24 сменил постановку на direct grouped ranking: пять CatBoostRanker вариантов
PairLogit/YetiRankPairwise выбирали день внутри currency-quarter/month, 85 fits
были quarterly OOS и mature-only. Ранний selector честно выбрал ranker-primary,
но late min lift упал до **2,232731**: сильный transport failure. YetiRank полезен
только как pace (h5 **2,463666**, min lift **2,397212**) и не обошёл AP23.

AP25 выделил outcome-free hard cadence pool: известное завтра не ниже текущего,
rolling rank не выше top30%, reserve rank выше top30%. На нём отдельно обучены
Cat mean utility, y20 classifier, future5 benefit regressor, rolling Cat и
ExtraTrees. Y20 specialist дал h5 **2,473910**, future-only **136,45 б.п.**, но
min rate всего **0,939956**. Причина — cold start: до 2024 causal hard-pool train
содержит меньше 100 зрелых строк.

AP26 заранее задал shrinkage specialist к global Cat по числу доступных train
rows. Лучший accuracy-frontier `y20_shrink200` получил h3/5/10/20=
**2,417356/2,483314/2,445996/2,495087**, h5 symmetric **74,45 б.п.**,
future-only **135,31 б.п.** Это сильнее AP21 по каждой точечной метрике, но min
currency rate по неизвестным горизонтам **0,970524**, поэтому strict gate не пройден.

AP27 добавил редкий причинный Cat backstop только если specialist pace не сработал
и прошлый rate/silence показывает дефицит; reserve, known-down veto, month rescue
и max2/week сохранены. Early selector выбрал r70, но late min rate **0,985808**.
Единственный строгий late вариант r60 добавил 26 backstop-решений и дал
**2,405426/2,464633/2,444496/2,467493**, min rate **1,001092**, zero empty
months, max2/week. H5: 691 сигнал, rate1,032, symmetric73,67, future131,79.
Против AP21 все lift-CI пересекают ноль; symmetric h5 против AP17 выше на
4,93 б.п. с положительным 20-date CI, но это не доказывает новый общий winner.

Итог frontier: AP23 — лучший strict point, AP26 — лучший accuracy result,
AP27 r60 — specialist с literal strict cadence. Ни один не является fresh
independent holdout winner. Все AP23–AP27 audits прошли; полный набор —
**271 тест**.

## Последние AP18-E/AP22-E: CatBoost и разделение ролей экспертов

[PDF](output/pdf/ivan_after_publication_ap22_effective.pdf) ·
[отчёт](research/after_publication_ap22_effective_report.md) ·
[AP21 protocol](research/after_publication_ap21_effective_registered.md) ·
[AP21 results](results/research/after_publication/ap21_effective).

AP18 проверил шесть заранее заданных full/recent/pace blend через полностью
замороженный AP17 controller. Early selector выбрал 50/50 full/recent:
поздние h3/5/10/20 = **2,387024/2,447501/2,409138/2,468434**, min rate
**1,016376**, zero empty months. Прирост h5 к AP17 +0,010078 имеет CI
[-0,030982;0,054304]: это усиленный контроль, а не доказанный скачок.

AP19 добавил новые multi-horizon ExtraTrees и CatBoost: четыре классификатора,
регрессию средней полезности по неизвестным горизонтам и PairLogit. Лучший новый
предиктор CatBoost mean utility дал **2,369258/2,463342/2,506324/2,466066** на
h3/5/10/20, h5 symmetric +73,02 б.п. и future-only +127,82 б.п. Остаточные
Hist/Ridge/logit stacks и fixed blends AP20 не дали устойчивого улучшения min lift.

Главная находка AP21 — не смешивать scores одинаково для всех решений. Rolling
ExtraTrees используется как primary expert, а CatBoost utility только как pace
expert, когда trailing365 rate валюты ниже1. При exact AP17 thresholds и month24
rescue зарегистрированный `roll_cat_dual_pace_month24_cap2` получил поздние
h3/5/10/20 = **2,402124/2,446225/2,428982/2,411100**, min lift **2,402124**,
min currency rate **1,008734**, zero empty months и max2/week. H5 symmetric
+72,90 б.п., future-only +130,24 б.п.

Это первый зарегистрированный строгий вариант выше 2,4 по всем неизвестным
горизонтам, но его нельзя выдавать за fresh winner: ранний selector выбрал другой
вариант, а поздний период многократно открыт. Разницы к AP17 по h3/5/10/20
+0,026/+0,009/+0,037/-0,055 имеют 20-date CI, пересекающие ноль. Local-Cat
dual pace показывает верхнюю границу h5 **2,498576** и min lift **2,431360**,
но min rate **0,970524**, поэтому продуктовый cadence не проходит.

AP22 проверил восемь causal rolling/local rank-consensus вариантов: ни один не
прошёл ранние совместные условия, лучший поздний fresh min lift только **2,338042**.
Вывод — outcome-free rank consensus размывает сигнал; следующий новый класс
должен оценивать компетентность экспертов только по уже созревшим прошлым ответам.
Все AP18–AP22 refit/prefix audits прошли; полный набор — **257 тестов**.

## Последние AP15-E/AP17-E: deficit pacing закрывает строгий cadence

[PDF](output/pdf/ivan_after_publication_ap17_effective.pdf) ·
[отчёт](research/after_publication_ap17_effective_report.md) ·
[AP17 protocol](research/after_publication_ap17_effective_registered.md) ·
[AP17 results](results/research/after_publication/ap17_effective).

AP15 проверил5 заранее заданных узких порогов/adaptive/month rescue. Early
selector выбрал Extra silence14+month24; поздний h5 **2,475700**,rate1,005550,
zero empty months, но mincurrency0,986126. Ни одна AP15 policy не прошла late
strict h5/all-h cadence. Вывод: calendar gap и corridor deficit - разные задачи.

AP16 зарегистрировал5 causal pacing controllers. Early selector выбрал
`pace365_p55_r70`: дополнительная точка только если past365 decision rate<1,
Extra rank>.55 и reserve rank>.70. Late h3/5/10/20=
2,383619/**2,446617**/2,403340/2,480249; minrate1,031659,h5currency
1,045891-1,075774. Он доказанно выше strict top35 на h5:+0,071788,
CI[0,020143;0,121321], и reserve7:+0,106379,CI[0,008863;0,200504].
Но два currency-month остаются пустыми.

AP17 зарегистрировал один вариант: тот же выбранный pacing+month24 rescue.
Late h3/5/10/20 **2,375650/2,437423/2,392312/2,465753**; minlift2,375650,
minrate1,031659,h5rate1,065315,currency1,045891-1,075774,zero empty months,
max2/week,sym68,747499,future130,172438. Против AP12 h5 delta-0,037805,
CI[-0,113234;0,022645], потери не доказано; против top35 +0,062594,
CI**[0,010720;0,110466]**; против reserve7 +0,097185,
CI**[0,006393;0,186485]**. 50-date sensitivity сохраняет оба положительных знака.

AP17 добавил к AP16 две month-rescue точки, одна заменила pacing decision:net+1.
Аудит подтвердил365-day rate,ranks,reasons,month-first,veto,cap и future-prefix.
Полный набор из **242 тестов прошёл**. Период открыт; AP17 - новый strict
retrospective control, не independent winner.

## Последний AP14-E: восстановление частоты почти без потери lift

[PDF](output/pdf/ivan_after_publication_ap14_effective.pdf) ·
[подробный отчёт](research/after_publication_ap14_effective_report.md) ·
[protocol](research/after_publication_ap14_effective_registered.md) ·
[результаты](results/research/after_publication/ap14_effective).

До результата зарегистрированы4 fixed score: AP13 rolling2/local, outcome-free
50/50 blend и AP12 expanding ExtraTrees. Для каждого проверены top32.5,top35,
silence14/r80,silence21/r70 и causal adaptive105 -20fresh policy. Все используют
prior250 rank,warmup40,known-down veto,max2/week; adaptive видит только прошлые
решения.14 политик прошли early2023 gates.

Early selector выбрал `extra_roll2_silence21_r70_cap2`. Поздние h3/5/10/20=
**2,425379/2,485115/2,423254/2,431420**, но h5rate0,968196,
mincurrency0,933831: поздний cadence gate провален. Точность против AP12 не
отличается, deltaCI[-0,075784;0,096772]. Это честно выбранный score, не продукт.

Самый полезный поздний diagnostic `extra_ap12_silence14_r80_cap2`: h5
**2,480451**,rate**1,004055**,mincurrency0,986126,sym67,255,future133,096.
Он добавляет23 net решения к frozen AP12 и почти точно сохраняет lift:
delta+0,005223,CI**[-0,027293;0,036963]**. Против AP13 reserve7 прирост доказан:
CI[0,038489;0,259375]. Один пустой полный месяц остаётся, поэтому это
near-cadence, а не literal strict pass.

Строгий `extra_ap12_top35_cap2`: h3/5/10/20=
2,326256/**2,374829**/2,305872/2,369384; rate по валютам1,031-1,113,
zero empty months,max2/week. Цена к AP12 доказана: -0,100399,
CI[-0,171546;-0,037851]. Но он доказанно выше AP10 known-z и AP11 hazard.
Adaptive105 даёт h52,433364 и mincurrency h5>1, но имеет редкие пустые месяцы.
Простой50/50 ensemble фронт не улучшил.

Audit пересобрал5755targets,20 policy, controller state и prefix-corruption
checks; **229 тестов** прошли. Это opened retrospective, не fresh holdout.

## Последний AP13-E: recent/local ExtraTrees и цена cadence

[PDF](output/pdf/ivan_after_publication_ap13_effective.pdf) ·
[подробный отчёт](research/after_publication_ap13_effective_report.md) ·
[protocol](research/after_publication_ap13_effective_registered.md) ·
[результаты](results/research/after_publication/ap13_effective).

До fit были зарегистрированы шесть новых scores: ExtraTrees с rolling train
730/1095 дней, recency half-life730, отдельный local ExtraTrees со shrink к
frozen global, meta-router AP12 Extra/localHist и causal Brier365-router. Для
каждого score проверены primary top30, reserve после 7 дней молчания и rescue
с 24-го числа месяца. Всего 18 свежих политик и 8 frozen controls; выбор только
по раннему 2023, неизвестным h3/5/10/20 и заранее заданным cadence/benefit gates.

Поздний local ExtraTrees дал новый точечный h5 **2,531074**, rate **0,905443**,
hit **74,59%**. Rolling2 дал h5 **2,517586** и лучший minimum по неизвестным
горизонтам: h3/5/10/20 **2,435721/2,517586/2,470410/2,468674**, min **2,435721**.
Но прирост к AP12 full ExtraTrees 2,475 не доказан: local CI
**[-0,022661;0,149765]**, rolling2 **[-0,038610;0,125435]**. Оба доказанно
выше AP1 cap2: соответственно **[0,052128;0,388392]** и
**[0,021824;0,389225]**.

Строгий rolling2 reserve7 даёт h5 **2,340238**, rate **1,154963**, диапазон
валют **1,1206–1,1953**, максимум2/ISO-week, maxgap29дней и ни одного пустого
полного месяца. Rolling3 reserve7 похож: **2,352265**, rate **1,177375**.
Добор примерно150 событий имеет реальную цену к AP12: rolling2 delta CI
**[-0,260273;-0,024701]**, но всё ещё доказанно выше AP10 и AP11 hazard.

Ранний selector выбрал `router_extra_local_month24_cap2`: поздний h5
**2,348520**, rate **1,027962**, sym **69,7836**, future **125,527**. Он
доказанно хуже AP12 ExtraTrees, CI **[-0,250426;-0,008052]**, поэтому не заменяет
простой контроль. Router также хуже калиброван: Brier0,225916 против0,211861 у
AP12; лучший AP13 Brier0,210891 у decay730. Главный отрицательный результат:
сложный causal выбор эксперта менее устойчив, чем одно большое лесное ранжирование.

Train-only importance rolling2 показывает совместную роль annual phase,
вечернего CNY basis, `known_change_z`, движений announced/effective рядов,
волатильности и USD/TJS факторов. Это impurity importance, не причинные эффекты.
Аудит восстановил5755targets,17origins,102 fit/router записи и все сигналы;
**224 теста** прошли. Период 2024–2026 уже открыт многими итерациями: результаты
— retrospective candidates, а не fresh holdout.

## Последние AP11-E/AP12-E: новый ExtraTrees до 2,475

[PDF](output/pdf/ivan_after_publication_ap12_effective.pdf) ·
[подробный отчёт](research/after_publication_ap12_effective_report.md) ·
[AP12 protocol](research/after_publication_ap12_effective_registered.md) ·
[результаты](results/research/after_publication/ap12_effective).

Исправлена структура задачи после получения курса на завтра: h1 уже известен,
а модели учат только неизвестные шаги2..h. AP11 conditional hazard HistGB
получил поздний h5 **2,135019** против AP10 **2,054037**, delta CI
**[0,024340;0,144939]**. Это первый доказанный модельный прирост над простой
базой в одинаковой сегодняшней опоре.

AP12 исключил h1 из selector и заранее зарегистрировал пять новых scores:
полный ExtraTrees, компактные Hist/ExtraTrees, local Hist с global shrink и
мультикласс первого провала. 16 свежих политик используют prior250 strict-rank,
veto известного снижения и cap2;3 прошли ранние совместные условия.

Ранний 2023 selector выбрал `local_hist_h5_r30_nogap_cap2`. Поздние
h3/5/10/20 **2,249813/2,268181/2,305287/2,270683**; h5 rate **1,020491**,
sym **67,713089**, future **115,912842**. К AP10 h5 delta CI
**[0,063612;0,388060]**, но к AP11 hazard **[-0,018206;0,290788]** - отдельное
преимущество не доказано.

Лучший новый поздний challenger `extra_h5_r30_nogap_cap2`:
h3/5/10/20 **2,385630/2,475228/2,350489/2,418814**, h5 hit **73,50%**,
rate **0,969691** (по валютам0,948773–0,993597), sym **66,581498**,
future **134,674142**. Он имеет лучший поздний Brier **0,211861**. h5 прирост
к AP10 CI **[0,205703;0,636313]**, к AP11 hazard **[0,138520;0,545055]**, к
старому AP1 exact **[0,051474;0,370064]**. Против AP1cap2 CI
**[-0,010171;0,335440]**, не доказан. 50-date чувствительность сохраняет
прирост к AP10 **[0,188165;0,680708]**.

ExtraTrees прошёл ранние условия, но localHist имел выше ранний minimum lift;
поэтому поздние2,475 - заранее определённый сильный diagnostic challenger, а
не fresh-holdout выбранный рекорд. Частота слегка ниже нашей внутренней границы
1; max2/ISO-week соблюдён, один пустой полный месяц, maxgap43дня. Поздняя смесь
known70/hazard30 даёт h5 **2,382949** при rate **1,177375**, но не прошла ранний
rate gate. Аудит восстановил5755targets,17масок/85fit-строк, все scores/signals;
**218 тестов** прошли; исторические receipts и bank execution по-прежнему не
сертифицированы.

## Последний AP10-E: знание нового курса при сегодняшней опоре

Пользователь уточнил target: опора **действующий сегодня ЦБ** и без знания, и
со знанием завтрашней фиксации. Результат её знания теперь проверяется после
переобучения одинаковых моделей, а не переносом старых сигналов на другую цель.
[PDF](output/pdf/ivan_after_publication_effective_information.pdf) ·
[протокол](research/after_publication_effective_information_registered.md) ·
[сильный биржевой контроль](research/after_publication_effective_market_control_registered.md) ·
[результаты](results/research/after_publication/ap10_effective_extended).

| h5, те же даты/модель | Без нового ЦБ | С новым ЦБ | С новым + veto известного снижения |
|---|---|---|---|
| HistGB только ЦБ | 0,997804 | 1,975266 | 2,014332 |
| HistGB + биржа | 1,717072 | 2,049456 | 2,072437 |
| Survivalh5 + биржа | 1,760820 | 2,041189 | 2,083183 |
| Logit + биржа | 1,613942 | 1,984409 | 2,083477 |

Прирост Hist с одинаковой биржей **+0,332384**, CI **[0,184434;0,472143]**;
с veto к тому же контролю CI **[0,202896;0,496957]**. Survival сveto
**[0,180986;0,462866]**. CBR-only Hist **[0,784993;1,211539]**. Все сравнения
при18:30, одна информация биржи и20мин задержка, не фактический before/after
clock. Контроль убирает прямую фиксацию ВСЕХ валют и перебазирует биржу на текущий ЦБ.

Ранние условия проходят15 политик. Выбрано простое `known_change_z_urgent_cap2`:
ранний minlift1,755432, benefitLB2,883796, forward-ratiomin1,136258 кknown-sign.
Поздние h1/3/5/10/20 **1,935727/2,071153/2,054037/2,077536/2,075095**.
h5=927/3260,09.01.2024–25.08.2026,rate1,385059,sym55,313108,fwd87,565409.
h5CI[1,860043;2,263515],50дат[1,832863;2,305466]; все pooledliftLB>1,3,
sym/fwdLB>0. h20fwdCI[28,041345;143,282781],50дат[11,943804;158,971125].
75 валютно-годовых точечных lift минимум1,650022, не одновременная значимость.
Макс2/неделю,пауза21день,нет пустых полных месяцев,4,3–9,4% недель пустые.

Простой known-sign с3днями паузы h5=1,949810. Его усиление нормированным
изменением до2,054037 имеет deltaCI[0,015262;0,195382]. Сложный gatedmarketlogit
2,083477 против2,054037 deltaCI[-0,040035;0,098267], не доказанный новый лучший.
Старый AP1 change_z_r25 даёт2,259851 на matchedh5, и не забывается; следующая
проверка сравнит его частоту/кластеры сновымcap2, без позднего выбора поlift.

12модельных вариантов,27политик;68квартальных записей (4наборы×17), маскиодинаковы,
в каждой3модели. Эффективные targets при прежнем более строгом publicationh20
train-cap. 27 555 перебазировок проверено, maxerror1,1e−13. **206 тестов** прошли,
11новых; независимая индексация5755событий, порча будущих данных всехвалют/свечей,
незрелыхlabels. Фиксированное h1послеreceiptужеизвестно, не навык неизвестного прогноза.
Календарные receipts не сертифицированы, периоды многократно открыты, не bankP&L.

Дополнительно все **510 сохранённых экземпляров политик** AP1–AP9 пересчитаны
по обеим опорам с matched датами/сигналами: 10 200 строк, без переобучения или
выбора нового победителя. AP3publication1,630058→effective1,030942 отражает смену
задачи, НЕ вред дополнительной информации. Предложенный AP10volume/shape был
остановлен дообучения по уточнению цели; все173455основныхvalue/volume null,
объёмной модели не было. Следующийпоиск — неизвестныйостаток и conditional/local-global.

## Сохранённый AP9: частично известные ответы и частота обновления

[PDF AP9](output/pdf/ivan_after_publication_ap9.pdf) ·
[предрегистрация](research/after_publication_ap9_registered.md) ·
[все прогнозы, экспозиции и проверки](results/research/after_publication/ap9).

**Нового лучшего lift нет.** Шесть семейств при квартальном/месячном обновлении:
прямой HistGB full20/mature5, крупные survival-интервалы full20/partial,
20 отдельных survival-шагов full20/partial. Всего 12 модельных вариантов и
34 политики, прежние 133 признака, срез 18:30 и задержка свечей 20 минут.
Используется только наблюдённый префикс до origin−2 дня: без будущих нулей,
без шагов после первого падения, без отбора одних быстро известных неудач.

Ранний выбор `cny50_quarter_fine_full20_urgent_cap2` оставил квартальное
обновление и полный h20. Ранние условия проходят 11 строк, включая 4 строки
прежних контролей/точных копий; 7 новых проходов. Ранний minlift **1,349995**,
minbenefitLB **11,806261**, minforward-ratio **0,999124**.

| h, следующие наблюдения | 1 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| Lift выбранной модели | 1,478 | 1,511 | **1,612** | 1,566 | 1,572 |

h5: **908/3260**, **09.01.2024–25.08.2026**, частота **1,356670**,
выгода ТЗ **36,818448**, future-only **55,765918 б.п.** Пауза максимум 18 дней,
максимум 2 сигнала в неделю, пустых полных месяцев нет; 7,2–10,9% недель пустые.
Lift delta CI к AP3 **[-0,077732; 0,031199]**, к AP4 **[-0,065825; 0,027325]**.
Симметричная выгода к AP3 +3,424830 б.п., CI **[0,270638; 6,545539]**, но
future-only delta CI **[-3,382141; 5,073589]**. Превосходство над AP4 не доказано.
Все pooled lift lowerCI>1,3 и sym>0 при блоках 20/50 дат, однако KZT-2026 h1
**1,226804**, future-only h20 CI **[-12,991694; 113,060969]**.

| Свежесть train, без CNY, h5 | Full20 | Свежие ответы | CI разницы |
|---|---|---|---|
| Квартальный direct | 1,657 | 1,603 | [-0,133; 0,017] |
| Квартальная крупная survival | 1,616 | 1,618 | [-0,068; 0,071] |
| Квартальная пошаговая survival | 1,629 | 1,631 | [-0,071; 0,075] |
| Месячный direct | 1,632 | 1,605 | [-0,091; 0,033] |
| Месячная крупная survival | 1,607 | 1,647 | [-0,034; 0,127] |
| Месячная пошаговая survival | 1,643 | 1,673 | [-0,056; 0,119] |

Учащение обновления при неизменном full20 тоже не дало подтверждённого h5-буста.
Месячная partial пошаговая уменьшила Brier **0,18940 → 0,18108**, но её поздние
**1,672724** не основание для выбора: ранние minlift **1,194231** и forward-ratio
**0,787236** не проходят. Ни один вычисленный h5-парный CI не подтвердил прирост.

На обновлении 01.01.2025 full20 даёт 3610 событий до 02.12, mature5 — 3685 до
23.12, хотя бы один наблюдённый шаг — 3705 до 27.12. Дополнительные 95 строк —
валюта-дни, не независимые даты. Частичных первых падений 83, ещё не упавших 12.
Пошаговые строки риска **23671 → 23933**, первые падения **2871 → 2954**.
Последний использованный receipt **28.12.2024 < 30.12.2024**.

**195 тестов** прошли, 9 новых. Независимый аудит восстановил все 408 записей
обучения и 51 снимок префиксов, хеши строк риска, сигналы и прогнозы контролей
(ошибка 0). Причинность не доказывает независимость цензурирования от режима.
Исторические receipts условные, периоды многократно изучены, CI не учитывают
весь поиск. AP3 остаётся контролем, AP4 — альтернативой. Далее проверяются
неиспользованные volume/value свечей с проверкой единиц и price-only контролями.

## Сохранённый AP8: предельная польза вечерних данных

[PDF AP8](output/pdf/ivan_after_publication_ap8.pdf) ·
[предрегистрация](research/after_publication_ap8_registered.md) ·
[все срезы, прогнозы и проверки](results/research/after_publication/ap8).

**Выбран простой CNY в19:30, превосходства над18:30 нет.** Четыре заранее
заданных времени, неизменные133 признака и параметры: CNY, CNY+HistGB50/50,
CNY+survival50/50. Полное mature20 до origin−2дня, квартальные обновления.
12 первичных вариантов +6 frozen18:30 контролей +2 прежних именованных=20.
Ранние условия проходят13 строк, включая7 первичных (5 новых по времени).

| Время, h5 lift | CNY | CNY+HistGB | CNY+survival |
|---|---|---|---|
| 18:10 | 1,503 | 1,553 | 1,583 |
| 18:30 | 1,638 | 1,630 | 1,630 |
| 18:50 | 1,564 | 1,651 | 1,661 |
| 19:30 | **1,653** | 1,587 | 1,605 |

Выбранный `t1930_cny_urgent_cap2`: ранний minlift1,392638, minbenefitCI1,362567,
minforward-ratio1,078324. Поздние h1/3/5/10/20:
**1,507 /1,572 /1,653 /1,634 /1,586**. h5: **905/3260**, даты
**09.01.2024–25.08.2026**, частота **1,352188**, симметричная выгода **35,114453**,
future-only **59,221593 б.п.** Максимум2/неделю, пауза14дней, пустых полных
месяцев нет,9,4% пустых недель. По181 сигналу на валюту в одинаковые даты:
не905 независимых решений; bootstrap объединяет все валюты одной даты.

К AP3 h5 lift CI разницы **[-0,093693;0,150194]**, sym-разницы
**[-7,422264;11,421188]**, forward-разницы **[-9,019021;17,418494]**.
К AP4 lift **[-0,087410;0,138163]**. Ни один новый h5-вариант не доказал буста.
Все pooled lift lowerCI>1,3 и sym>0 при блоках20/50, но h20forwardCI
**[-3,053233;113,490130]**;50дат **[-11,860483;127,872995]**. Слабые клетки:
KGS2025h20 **1,172999**, KZT2026h1 **1,253314**. Прежние18:30 контроли сохранены.

Чистый эффект раннего времени:18:10 против18:30 CNY lift deltaCI
**[-0,217673;-0,057628]**, CNY+HistGB **[-0,149432;-0,010869]**, хуже.
19:30 против18:30 толькоCNY **[-0,072194;0,108710]**, прирост не подтверждён.
Среднее числоCNY-свечей47/49/51/54,00; на поздних датах CNY есть100%, новые
свечи к19:30 есть100%, basisменяется99,2%. Прямые пары: новые44,8%, доступные71,5%.
Это архивная доступность с20мин допущением задержки, не реальные receipts.

Все **186 тестов** прошли,9новых. Перестроены23020 событий снимков,68train-масок,
21063 строки дедлайнов источников. CNY/HistGB/survival18:30 максимальная ошибка
к сохранённым прогнозам **0**. Frozen18:30 контроли дают точно те же сигналы.
Полностью сохранены данные и отрицательные сравнения; следующие эксперименты
про зрелость отдельных горизонтов/цензурирование, без смены времени незаметно.

## Сохранённый AP7: аналоги и условные траектории

[PDF AP7](output/pdf/ivan_after_publication_ap7.pdf) ·
[предрегистрация](research/after_publication_ap7_registered.md) ·
[70 политик, пути и все прогнозы](results/research/after_publication/ap7).

**11 распределений; 6 ранних совместных проходов, из них 3 новых.** Выбран
`path_ridge_empirical_forward25_urgent_cap2`: Ridge предсказывает нормированный
20-шаговый путь, более поздняя отдельная часть train даёт остаточные пути,
скор — 75% прошлого ранга p5 + 25% ранга ожидаемой future-only выгоды.
Библиотека начинается в 2022, обновляется ежемесячно и содержит только mature20
до origin минус два дня. Никаких будущих путей в признаках или внутри недели.

Поздние h1/3/5/10/20: **1,394 / 1,417 / 1,497 / 1,480 / 1,401**.
h5: **900 / 3260**, 09.01.2024–25.08.2026, частота **1,3447**, максимум 2/неделю,
пауза **28 дней**, пустых полных месяцев нет. Выгода ТЗ **+17,47 б.п.**,
future-only **+44,15 б.п.**. Превосходства нет; относительно AP4 h5 lift CI
**[-0,2765; -0,0010]**, относительно AP3 **[-0,2830; +0,0040]**. Симметричная
выгода h5 снизилась к AP3 на **15,92 б.п.**, CI **[-24,39; -8,48]**.
h20 lift нижний CI **1,197**, forward CI **[-27,07; +104,18]**; часть срезов
ещё слабее. AP3 основной контроль, AP4 альтернатива; новый кандидат не продвигается.

| Распределение, h5 | Lift | Частота | Вывод |
|---|---|---|---|
| 128 общих наблюдений «валюта-день», 7 признаков | 1,515 | 1,29 | Ниже контроля |
| 128 ближайших, 24 признака | 1,337 | 1,30 | Прирост CI [-0,308; -0,060] |
| 32 ближайших дня своей валюты | 1,493 | 1,30 | Локализация не дала буста |
| Частичное локальное/общее смешивание | 1,506 | 1,29 | Нет подтверждённого буста |
| Общая библиотека последних 730 дней | 1,534 | 1,26 | Нет подтверждённого буста |
| Хронологически разделённые деревья | 1,534 | 1,32 | Нет подтверждённого буста |
| Безусловно все прошлые пути | 0,776 | 0,66 | Пауза 238 дней |

AP4 +25% ранга распределения деревьев: диагностический **1,661**, CI к AP4
**[-0,0326; +0,0870]**. Ранний benefit lower CI **-7,41 б.п.**, поэтому не
выбран. Ridge с реальными остатками Brier **0,18832**, с гауссовскими **0,20951**;
разница lift не доказана. Выгоды усредняются по отдельным сценариям, не считаются
от средней траектории: нелинейность формулы проверена отдельным тестом.

Все **177 тестов** прошли, 9 новых. Повторно восстановлены **10 340** наборов
соседей/весов/прогнозов, 51 месячный пример деревьев, 306 fit-записей, все paths
и targets; численная ошибка выгоды меньше **1e-8 б.п.**, обрезанных сценариев 0.
Сохранены 56 870 scenario-логов. Время получения ЦБ остаётся предположением;
2023 и 2024–2026 не являются свежими тестами, CI не скорректированы на весь поиск.
Следующий пакет — дополнительные завершённые вечерние свечи и фиксированные
простые модели, с явным разделением времени выдачи сигнала. Target активен.

## Сохранённый AP6: OOS-сочетание моделей и условные ошибки

[PDF AP6](output/pdf/ivan_after_publication_ap6.pdf) ·
[предрегистрация](research/after_publication_ap6_registered.md) ·
[51 политика и все прогнозы](results/research/after_publication/ap6).

**16 моделей; 8 ранних совместных проходов, из них 6 новых.** Выбранный по
2023 году `ap4_w25_stack_local_residual_urgent_cap2` даёт поздний h5 **1,613434**,
не новый рекорд. 75% прежней ранговой смеси AP4 + 25% прошлого ранга локальной
логистической смеси, скорректированной глобальным HistGB по её прежним OOS-ошибкам.
Обучение только на выданных ранее экспертных оценках после разрешения h20 и
двухдневного embargo; ежемесячно. Не in-sample stacking и не будущий weekly top-k.

Поздние h1/3/5/10/20: **1,481 / 1,508 / 1,613 / 1,543 / 1,542**.
h5: **900 / 3260**, 09.01.2024–25.08.2026, частота **1,3447**, максимум 2/неделю,
пауза 16 дней, пустых полных месяцев нет. Симметричная выгода **+33,59 б.п.**,
future-only **+54,67 б.п.**. Парный h5 lift CI к AP4 **[-0,0656; +0,0234]**,
к AP3 **[-0,0695; +0,0335]**. AP3 основной контроль, AP4 альтернатива; AP6
не продвигается как улучшение. KZT-2026 ниже 1,3 на всех h, forward h20 CI
**[-16,85; +107,54]**. Объединённые lift/symmetric интервалы проходят пороги
при блоках 20/50 дат, но это условная многократно открытая ретроспектива.

| Проверка с неизменным контроллером, h5 | До | После | Вывод |
|---|---|---|---|
| Линейная смесь + контекст | 1,643 | 1,576 | Нет подтверждённого выигрыша |
| HistGB по экспертам + контекст | 1,631 | 1,584 | Нет подтверждённого выигрыша |
| Локальная линейная смесь + глобальная коррекция | 1,645 | 1,583 | CI [-0,150; +0,008] |
| Равная смесь AP5 + глобальная коррекция | 1,665 | 1,560 | Ухудшение; CI [-0,213; -0,006] |
| Source-only HistGB + экспертные прогнозы | 1,529 | 1,584 | CI [-0,007; +0,115], не подтверждено |

Диагностическая положительная линейная смесь rolling730: **1,656**, Brier h5
**0,17886** против **0,18630** у равной AP5; ранний min lift **1,278**, min
benefit CI **-20,21 б.п.**, поэтому не выбрана. Pairwise same-currency ranking
на зрелых парах в пределах 60 дней: **1,289**, частота **1,07**, пауза **69 дней**,
до 4 пустых месяцев; sym **+45,92**, forward **+25,25 б.п.**. Не замена AP3.

Сохранены коэффициенты, 32 метапризнака, 969 fit-логов, origins локальной
модели, поправки и все 51 сигнал. Все **168 тестов** прошли, включая 9 новых;
повторно восстановлены train-маски, масштабы, пары, outcomes и точные контроли.
Исторические receipts не сертифицированы. Следующий пакет — аналоги и условные
траектории, а не новая сетка весов прежних экспертов. Target остаётся активным.

## Сохранённый AP5: калибровка и задержанная адаптация экспертов

[PDF AP5](output/pdf/ivan_after_publication_ap5.pdf) ·
[предрегистрация](research/after_publication_ap5_registered.md) ·
[все66 политик](results/research/after_publication/ap5).

**Нового выбранного решения нет:** совместные ранние критерии прошли только
AP3 и AP4. Селектор снова выбрал контроль AP4 с h5 **1,630232**, не новый
кандидат. Все дальнейшие цифры новых вариантов — диагностика, не поздний отбор.

Шесть экспертов: CNY, прямые HistGB/ExtraTrees, survival HistGB, глобальная
и локальная survival-логистика. Пять калибровок: expanding/rolling365 logistic,
частично локальная, isotonic, frozen2023. Ежемесячно, только прежние квартальные
OOS-прогнозы с полным h20 до начала месяца минус2дня. Параметры не найдены по
поздним данным. Вероятности горизонтов приведены к невозрастанию.

Ранний Brier h5 глобальной/локальной логистики **0,3271/0,3436 → 0,2437/0,2471**
после rolling365. Поздний **0,2024/0,2311 → 0,1983/0,2082**.
Но поздний lift **1,397/1,380 → 1,346/1,297**; у локальной модели пауза126дней.
Замороженная монотонная калибровка CNY и двух логистик не изменила ни одного
позднего сигнала: точность вероятностей и ранжирование дней — разные задачи.

Delayed weighting: средний Brier пяти горизонтов на реально выданной тогда
вероятности, обновление только после mature20+3 календарных дней. Полураспад
63/252дня, eta2/10/30, global/local-shrink, 10% массы всегда равномерно.
67 210 строк журнала, все cutoff/веса проверены. Порядок валют внутри дня
не влияет на общее состояние, будущие ответы не меняют прежние веса.

| Диагностическая смесь, h5 | Lift | Частота | Brier |
|---|---|---|---|
| Равные rolling365-вероятности | 1,665 | 1,30 | 0,18630 |
| Global252/eta10 | 1,674 | 1,30 | 0,18570 |
| Веса зафиксированы01.01.2023 | 1,687 | 1,30 | 0,18570 |

Парный CI прироста адаптивного lift к равному **[-0,0041; 0,0273]**, к
замороженным весам **[-0,0323; 0,0018]**. Все12 интервалов прироста к равным
весам пересекают0. Максимальная пауза28дней. Сложность не оправдана приростом.
Равная изотоническая смесь даёт поздние1,704, но тоже не проходит ранний отбор.

Все **159 тестов** прошли, включая9 новых проверок зрелости, будущей порчи,
порядка строк, локального смешивания, замороженных карт/весов и монотонности.
Данные/часы/target неизменны, исторические receipt timestamps не сертифицированы.
Следующий шаг — OOS-метамодель и режимные взаимодействия по целям сигнала,
не только по усреднённой Brier-ошибке. Target активен, почасовой задачи нет.

## Сохранённый AP4: first-passage модели и мягкая полезность

[PDF AP4](output/pdf/ivan_after_publication_ap4.pdf) ·
[предрегистрация](research/after_publication_ap4_registered.md) ·
[46 политик и все результаты](results/research/after_publication/ap4).

Новый target — время до первого курса строго ниже текущей опубликованной цены.
Пять условных вероятностей первого снижения дают согласованную survival-кривую.
Проверены глобальные HistGB/логистика, локальная логистика, регрессия ограниченного
времени ожидания, глобальные поправки к прежним OOS-прогнозам локальной базы.
Никаких будущих цен в признаках: только зрелые будущие ответы для train.

Ранний выбор из четырёх кандидатов, прошедших совместные критерии2023, —
`cny50_hazard_hist_h5_urgent_cap2`. Поздний adjusted lift h1/3/5/10/20:
**1,478 / 1,520 / 1,630 / 1,566 / 1,559**. h5: **903/3260**, частота **1,35**,
09.01.2024–25.08.2026, максимум2/неделю, пауза16дней, нет пустых полных месяцев.

Нового рекорда нет: h5 **1,630232** против AP3 **1,630058**,
парный CI разницы **[-0,046; 0,050]**. Симметричная выгода
**+33,39 → +36,61б.п.**, парный CI **[0,79; 5,84]**;
future-only **+54,91 → +56,35б.п.**, CI прироста **[-2,18; 5,07]**.
Блоки50дат сохраняют этот вывод; при них h1 lift едва значимо хуже AP3
(CI разницы **[-0,0537; -0,00005]**). AP3 остаётся основным контролем.

Объединённые CI lift выше1,3 и симметричной выгоды выше0 на всехh при блоках
20/50дат. Но **KZT-2026 h1/h3 = 1,236 / 1,290**; future-only h20
**+54,07б.п., CI[-11,55; 111,87]**. Условная ретроспектива не стала новым holdout.

Сами модели h5: survival HistGB **1,616**, глобальная логистика **1,397**,
локальная **1,380**, ограниченное время ожидания **1,540**.
Последнее хуже обычного рыночного HistGB1,657, парный CI **[-0,221; -0,022]**.
Локальный mean-survival **1,375**, глобальная OOS-поправка25/50%:
**1,353 / 1,397**; значимого улучшения нет. Сырые кривые и логи сохранены.

Мягкие поправки не решили задачу. При весе25%: только известное прошлое
(minпоh) **1,541 / +42,86 / +52,46**, полный прогноз
**1,563 / +42,20 / +52,26**, только будущее(meanпоh)
**1,607 / +35,47 / +53,25** (lift / симметричныеб.п. / future-onlyб.п.).
При весе прошлого50% симметричная выгода растёт до **+49,02**, а будущая
падает до **+28,43** против **+54,91** у AP3. Это не утечка, но и не улучшение
предсказания будущего. Добавленная ценность прогнозной половины относительно
известной при25% не доказана: CI прироста lift **[-0,035; 0,086]**.

Новые веса CNY/HistGB/ExtraTrees тоже не дали сильного буста. Brier h5
обычного HistGB **0,18813**, survival HistGB **0,18828**, ExtraTrees **0,18057**;
Brier и lift отобранных дней — разные критерии. Следующий пакет исследует
калибровку и адаптацию экспертов только по уже созревшим исходам.
Все **150 тестов** прошли, включая8 новых проверок первого снижения,
at-risk интервалов, будущей порчи данных, зрелости OOS и точного контроля AP3.

## Основной контроль: полезность и регулярность после публикации (AP3)

[PDF AP3](output/pdf/ivan_after_publication_ap3.pdf) ·
[зафиксированный пакет](research/after_publication_ap3_registered.md) ·
[все таблицы](results/research/after_publication/ap3).

Проверены **57 политик**, шесть новых нормированных local/global/residual scores
и пять отдельных моделей будущей средней цены. По зрелому 2023 году единственный
кандидат, прошедший совместные ранние критерии, — `cny_hist50_urgent_cap2`.
Это 50/50 прошлых рангов CNY и глобального рыночного HistGB плюс причинный
недельный ограничитель. Числа ниже — уже изученная ретроспектива 2024–2026.

| h | Adjusted lift | 95% CI lift | Симметричная выгода, б.п. | 95% CI выгоды |
|---|---|---|---|---|
| 1 | 1,503 | [1,418; 1,593] | +19,68 | [15,81; 24,07] |
| 3 | 1,548 | [1,420; 1,695] | +30,08 | [23,38; 38,06] |
| 5 | 1,630 | [1,477; 1,799] | +33,39 | [25,25; 42,48] |
| 10 | 1,535 | [1,393; 1,696] | +37,65 | [24,86; 52,20] |
| 20 | 1,547 | [1,396; 1,712] | +46,87 | [20,73; 74,64] |

h5: **908/3260**, 09.01.2024–25.08.2026, **1,36 сигнала/валюту/неделю**,
максимум2/неделю, максимальная пауза15дней, нет пустых полных месяцев.
Около9–12% недель пустые: средняя частота не гарантирует минимум каждую неделю.
Интервалы условны, без коррекции на все повторные исследования.

По сравнению с AP2-D20 lift **1,654 → 1,630**, парный CI разницы
**[-0,223; 0,193]**: превосходство не доказано. Зато симметричная выгода
**+6,13 → +33,39б.п.**, парный CI прироста **[14,97; 39,12]**, пауза34→15дней.
Это компромисс: future-only выгода h5 **+71,20 → +54,91б.п.**,
парный CI изменения **[-31,49; -0,63]**. На h20 future-only CI **[-13,98; 109,44]**.

По отдельным годам h5 **1,545 / 1,914 / 1,488**; по валютам **1,561–1,722**.
Более узкий срез **UZS-2026**: h3/h10/h20 **1,288 / 1,228 / 1,214**.
Поэтому объединённое прохождение критериев не объявлено универсальной
устойчивостью, свежим holdout или доказанной банковской экономией.

Что не помогло: нормированная локальная Ridge **1,383**, нормированный локальный
прогноз +50% глобального OOS-остатка **1,281** против сырой Ridge **1,473**.
Паузы231/339дней. Даже деление готовых прогнозов на доступную волатильность
не устранило проблему. Все raw/rescaled/refitted controls сохранены.

Полезный, но не выбранный компромисс: ExtraTrees с положительным прогнозом
симметричной выгоды на всехh **1,833**, выгода **+69,87б.п.**, но частота **0,65**,
пауза94дня и до6 пустых месяцев. Прогнозировалась только неизвестная будущая
составляющая, известная прошлая сумма добавлялась точно. Следующая идея —
мягкий учёт этой полезности, а не жёсткий запрет большинства дней.

Информация и временная схема AP2-D20 неизменны: решение18:30, дополнительная
задержка20мин, target от последнего опубликованного курса. Исторические
received_at не сертифицированы. Все **142 теста** прошли. Активный target
продолжается без почасовой задачи; AP1/AP2 и прежние результаты не удалены.

## Сохранённый результат: вечер после публикации (AP2 / AP2-D20)

[PDF результата и ограничений](output/pdf/ivan_after_publication_ap2.pdf).
Проверены 55 политик и два полных replay, включая повторное обучение с
20-минутной задержкой рыночных свечей. Основной сценарий AP2-D20: решение
18:30, target от нового опубликованного курса, все будущие шаги неизвестны.

Выбранная только по 2023 году смесь `cny_cbr_w25_r25`:
**adjusted lift 1,554 / 1,646 / 1,654 / 1,559 / 1,542** при h1/3/5/10/20.
h5: **1,25** сигнала/валюту/неделю, **838/3260** сигналов/строк,
09.01.2024–25.08.2026, 95% CI lift **[1,440; 1,886]**. Эти поздние годы уже
просматривались; реальная история получения данных не сертифицирована.

Самый большой вклад — вечерняя рыночная информация: одинаковый HistGB с ней
даёт **1,805** против **1,228** без неё; парный CI прироста h5 **[0,296; 0,900]**.
Частоты при одинаковом правиле отбора различаются. Контроль со старым CNY —
**1,001**. Простая CNY-база **1,741**, ExtraTrees **1,840** остаются заранее
включёнными диагностическими кандидатами, не поздно выбранными победителями.

Прямые валютные пары при долях25/50/100%, умноженных на качество, дали
**1,703 / 1,619 / 1,528** — слабее чистого CNY. Локальная Ridge **1,473**;
глобальная коррекция её честных квартальных OOS-остатков с весами25/50/100%
дала **1,436 / 1,415 / 1,303**. Полные отрицательные результаты сохранены.

Открытые критерии: симметричная выгода выбранной смеси h5 **+6,13 б.п.**,
CI **[-6,20; 19,08]**; максимум5 сигналов в неделе при паузах34дня. Рыночный
HistGB имеет положительные интервалы выгоды на всехh, но встречаются паузы73дня
и годовая частота ниже1. У регрессий паузы ещё длиннее — до502дней с полной
поправкой. Следующий пакет исследует шкалу target, калибровку и совместный
контроль lift/выгоды/регулярности. Все **135 тестов** прошли.

Данные/предрегистрация/код: `research/after_publication_ap2*`;
результаты: `results/research/after_publication/ap2` и `ap2_delay20`.
Задержка20мин — явно заданное допущение, не гарантированное время доставки;
исходный нулевой лаг сохранён отдельно. Почасовая задача не используется.

## Первый пакет после публикации ЦБ (AP1)

[PDF первого пакета](output/pdf/ivan_after_publication_ap1.pdf) ·
[уточнение исходного ТЗ](research/after_publication_tz_decision.md) ·
[активный checkpoint](research/after_publication_next_steps.md).

Проверены 56 политик: простые правила и шесть классических моделей, отдельно
для двух стартовых цен. Все результаты относятся к календарно восстановленной
доступности объявлений и уже просмотренной ретроспективе 2024–2026.

| Сценарий, простое правило `change_z_r25` | adjusted lift h1/3/5/10/20 | Сигналов/валюту/неделю, h5 |
|---|---|---|
| От последнего опубликованного курса | 1,139 / 1,109 / **1,060** / 0,976 / 0,939 | 1,30 |
| От старого действующего курса, первый шаг уже известен | 1,917 / 2,274 / **2,265** / 2,227 / 2,220 | 1,30 |

Первый сценарий — наша консервативная интерпретация ТЗ; в AP1 устойчивые 1,3 и
положительная симметричная выгода не были достигнуты. Во втором сценарии
нормировка объявленного изменения на прошлую волатильность и причинный отбор
сильных изменений дают заметный выигрыш относительно знака с паузой 3 дня
(h5 **1,953**). Это не сравнение при строго одинаковом числе сигналов.
Квантильный бустинг, выбранный на раннем периоде, даёт **2,139**, но превосходства
над простым правилом **2,265** не доказал. Максимальная пауза простого рангового
правила — **43 дня**, несмотря на приемлемую среднюю частоту.

Исправлены прежнее слепое `i+1`, доступность peer-валют и зрелость train-labels.
Все **127 тестов** прошли. Старые **2,459** не реабилитированы: отдельно
[сохранён аудит 147 из 732 календарно недоступных сигналов](output/pdf/ivan_cbr_after_publication_applicability.pdf).
Нельзя сравнивать AP1 и 15:30 без выравнивания дат и target. Следующий пакет
исследует движение после формирования новой фиксации и регулярность сигналов.
Работа продолжается через активный target; почасовая автоматизация удалена.

## Сохранённый раунд: прямые биржевые пары и виджет (round 7)

[Обзор решения и результатов в PDF](output/pdf/ivan_direct_pairs_and_widget_report.pdf).
Собраны **99 013** десятиминутных CETS-свечей для KZT/AMD/KGS/TJS/UZS, TOM и TOD,
с метаданными номиналов и SHA-256. Проверены **43 новых direct/ML варианта** и
**20 residual-вариантов**. Все результаты и отрицательные контроли сохранены в
`results/research/round7/`; протокол: `research/round7_protocol.md`.

Для push сохраняем `15:30 availability_route`: **adjusted h5 2.053**, pooled
2.059, rate 1.19. Веса отдельно по валютам, выбранные на purged 2024, дали на
2025-2026 **1.978**, то есть хуже базы. Средний по горизонтам прирост отрицателен:
**-0.092**, paired 95% CI **[-0.154; -0.030]**. Небольшая общая поправка
`last_basis_soft_w025` дает 2.043; `CNY anchor + quantile residual 10%` - 2.038.
Ни один выбранный по прошлому периоду новый вариант не продвинут. Наилучшие
поздние числа невыбранных вариантов не выдаются за подтвержденный прогресс.

Отдельный результат для виджета: квартальная калибровка по разрешенным OOF
исходам уменьшила Brier **с 0.224 (сырой ранг) до 0.155**; постоянная вероятность
из трейна дает 0.200. Это проверка **только среза 15:30**, еще не live-интерфейс
и не оценка банковской экономии. Весь период 2025-2026 остается ретроспективой.
Исторический checkpoint раунда: `research/round7_next_steps.md`. Текущее
продолжение — `research/after_publication_next_steps.md`, активный target без
почасовой автоматизации.

Предыдущий подробный отчёт с результатами до packet EO, приоритетами по ТЗ,
сравнением всех семейств моделей и разбором крупнейших бустов:
[PDF](output/pdf/ivan_detailed_experiment_report.pdf).

Объяснение лучшего `15:30 availability-router` простыми словами, с полной
арифметикой lift и границей допустимости опубликованного курса на завтра:
[PDF](output/pdf/ivan_best_approach_explained_simply.pdf).

Полная связная версия с постановкой задачи, provenance по веткам, графиками,
статистическим аудитом и рекомендацией для пилота: [итоговый PDF](output/pdf/international_transfers_full_research_report.pdf).
Редактируемый источник: [results/research/full_report/report-source.md](results/research/full_report/report-source.md).

## Как развивался результат до round 7

- После уточнения формулы кейса новый основной кандидат —
  `geometry75_cba_consensus_basis25`: combined lift по пяти горизонтам
  **1.623 / 1.913 / 1.931 / 1.927 / 1.879**, средний **1.855**, частота
  около **1.26** сигнала на валюту-неделю. Все горизонты проходят 1.30.
- Новый независимый `futures_extra` на предыдущих perpetual CNY/RUB и USD/RUB
  сессиях даёт five-horizon minimum **1.534**; stale20 падает до **1.064**.
  Его minimum-consensus с лидером поднимает point minimum до **1.659**, но
  paired CI прироста **[-0.183; +0.098]** пересекает ноль, поэтому incumbent
  пока не заменён.
- Delayed multi-horizon weighting причинно меняет долю incumbent по уже
  завершившимся исходам. Лучший online-вариант даёт point minimum **1.643**
  против **1.623**, но paired CI прироста **[-0.144; +0.081]** включает ноль.
  Fresh-versus-stale minimum gain **+0.262**, CI **[+0.111; +0.593]**:
  информация MOEX настоящая, но её добавочный вес пока оценён неточно.
- Фиксированный срез MOEX до **12:00 MSK** дал новый point-лидер:
  арифметическое среднее рангов incumbent и `noon_hist` достигает five-horizon
  **min 1.714 / mean 1.892**, h5 **2.062 / 1.873** при rate **1.25 / 1.47**
  в 2025/2026. Paired min-gain **+0.091**, CI **[-0.137; +0.202]**, поэтому
  статистически incumbent пока не заменён; double-stale заметно хуже.
- Отдельный 15:30 fixing-proxy даёт **min 1.708 / mean 1.883** и h5
  **1.983 / 1.901** при rate **1.38 / 1.34**. Против stale-20 его paired
  minimum gain равен **+0.926**, CI **[+0.638; +1.167]**, поэтому свежесть
  intraday-информации доказана. Против 12:00 point-лидера преимущества нет:
  minimum difference **-0.006**, CI **[-0.106; +0.178]**. Исторические
  `value/volume` в MOEX candles пусты; честные OHLC-прокси дают тот же ranking,
  а исключать дни без торгов из знаменателя нельзя.
- Причинный availability-router на 15:30 использует fixing-rank в биржевой день
  и noon-rank в день без CNY-сессии, не удаляя строки. Это новый point-лидер:
  **min 1.780 / mean 1.956**, h5 **2.096 / 1.994**, rate **1.19 / 1.28** в
  2025/2026, min-currency **1.832**. Прирост minimum к noon равен **+0.066**,
  но CI **[-0.123; +0.238]** пересекает ноль; один квартал имеет rate **0.955**.
  Поэтому router — frozen retrospective challenger, а не доказанная замена.
- Cadence-first policy screen выбрал более редкий rolling-20%/20 вариант:
  combined min **1.785**, но худшая квартальная частота ухудшилась с **0.955**
  до **0.875**. Политику не продвигаем и не подгоняем дальше по 2025Q2.
- Fixed-parameter 15:30 fixing basis на всей доступной истории 2022–2026 даёт
  годовые minimum lifts **1.531 / 1.514 / 1.578 / 1.740 / 1.665** при
  ежегодной частоте **1.19–1.38**. Combined: **min 1.601 / mean 1.751**,
  h5 **1.751**, 95% CI **[1.612; 1.926]**, min-currency **1.649**. Разрез
  24.02.2022: min **1.471** до и **1.533** после; короткий pre-SVO блок
  описательный, но эффект не выглядит исключительно пост-SVO режимом.
- Cutoff frontier внутри 10:00–15:30 выбрал 15:20 на 2024; на 2025–2026
  raw 15:20 почти равен 15:30: min/mean **1.712 / 1.874** против
  **1.708 / 1.883**. Availability-router в 15:20 даёт **1.770 / 1.952**
  против **1.780 / 1.956** в 15:30. Aggregate non-inferiority проходит margin
  -0.05, но h20 CI **[-0.056; +0.064]** едва не проходит, поэтому формальная
  эквивалентность всех горизонтов не заявляется; это ранний challenger.
- Схема «15:30 fixing anchor + глобальный residual ML» обучалась поквартально
  на всех валютах с полностью разрешёнными h5 labels. Лучший новый ExtraTrees
  residual дал screen min **1.558**, смеси с router — не выше **1.536**, тогда
  как неизменённый router дал **1.607**. Селектор оставил простую формулу:
  48-признаковая коррекция не добавила переносимой информации.
- Outcome-free нормализация fixing basis выбрала на 2024 смесь router 75% +
  trailing-60 robust-z 25%: screen min вырос **1.607 → 1.623**. На 2025–2026
  улучшение не перенеслось: **min/mean 1.775 / 1.934** против
  **1.780 / 1.956** у исходного router; h5 **2.038** против **2.059**.
  Частота 2025Q2 выросла **0.955 → 0.986**, но ценой precision, поэтому
  структурную нормализацию не продвигаем.
- Reference-invariant внутридневные shape-прокси проверяют, что высокий basis
  держался во всех частях окна 10:00–15:30, а не возник одним всплеском. На
  2024 смесь router 75% + minimum three-block mean 25% подняла min
  **1.607 → 1.628**, но на 2025–2026 дала **1.761 / 1.930** против
  **1.780 / 1.956**; h5 **2.043** против **2.059**, minimum-quarter rate
  ухудшился **0.955 → 0.923**. Простое среднее всей сессии остаётся лучше.
- Currency-specific CNY transmission оценивал строго прошлыми доходностями
  rolling beta каждой цели к CNY и mean reversion target/CNY. На 2024 смесь
  router 75% + projected-60 25% дала **min/mean 1.626 / 1.757** против
  **1.607 / 1.653**. На 2025–2026 она откатилась к **1.754 / 1.919** против
  **1.780 / 1.956**; в 2025Q2 h5 lift упал **1.405 → 1.068**. Более равная
  частота не компенсирует нестабильную валютную чувствительность.
- Официальная выгода считается относительно среднего в симметричном окне
  `-h..+h`; её значения у лидера **+18.0 / +31.3 / +38.1 / +47.1 / +70.3
  б.п.** Future-only выгода остаётся дополнительной строгой диагностикой.
- Новый основной past-only кандидат `logit50_extra50` проходит заданный гейт с
  запасом: **lift 1.838 / 1.850** при **1.31 / 1.33** алерта на
  валюту-неделю в 2025/2026. Combined: **1.846 / 1.28**.
- Частично-пулированный challenger `primary75_local_consensus25` поднял
  combined point estimate до **1.867 / rate 1.27** и выгоду до **+76.5 б.п.**,
  но парный CI прироста lift **[-0.043; +0.095]** включает ноль. Поэтому это
  кандидат для будущего теста, а не новая победа по просмотренным годам.
- Он смешивает простую 19-признаковую логистику и ExtraTrees на предыдущей
  завершённой сессии CNY/RUB MOEX; курс ЦБ на завтра, текущий close MOEX и
  другие будущие значения не используются.
- Результат переносится между валютами и временем: минимальный lift коридора
  **1.538**, минимальный lift квартала **1.407**, минимальная квартальная
  частота **1.01**. То есть строгий квартальный cadence-гейт пройден.
- Bootstrap 95% CI по combined lift: **1.562–2.137**. Из 15 соседних вариантов
  порога и окна 14 проходят lift 1.30 и rate 1–2 в обоих годах.
- На отдельном pre-2022 прогоне 2017–2021 тот же механизм даёт combined
  **1.845 / rate 1.147**, а каждый год — lift **1.57–2.18**. Поэтому CNY-сигнал
  не является только пост-СВО артефактом.
- Единая lifecycle-политика 2017–2026 даёт **lift 1.745 / rate 1.201**;
  худший год **1.546**, минимальная годовая частота **1.061**, минимальная
  валюта **1.677**. До 2 000 post-2022 resolved-строк сохраняется expanding
  history, после — механический переход на hard reset.
- Новый smooth-additive challenger `primary75_all_gam25` даёт лучший past-only
  point estimate: **1.867 / 1.894**, combined **1.892**, rate **1.221**,
  выгода **+80.4 б.п.** Но paired CI прироста `[-0.046; +0.153]` включает ноль,
  поэтому frozen primary пока не меняется.
- Независимая causal reliability surface поднимает combined ещё до **1.897**:
  **1.981 / 1.807** по годам при rate **1.18 / 1.19** и min-currency **1.737**.
  Она сильнее в 2025, слабее primary в 2026; paired CI также включает ноль.
- Новый ordered-waveform ExtraTrees по последним 20 завершённым сессиям CNY
  даёт **1.782 / 1.827**, combined **1.827 / rate 1.216**. Тот же waveform с
  задержкой 20 строк падает до **1.248**; paired gain **+0.579**, CI
  **[+0.285; +0.937]**, Holm `p=0.001`. Свежий path-сигнал доказан, но его
  смесь с primary улучшает lift лишь на недоказанные **+0.013**.
- Fixed random-convolution логистика сама даёт только **1.209**, но её
  заранее заданная 25% поправка к primary создаёт сильный point estimate:
  **1.990 / 1.819**, combined **1.911 / rate 1.204**, выгода **+81.1 б.п.**
  Свёртки подтверждены против stale20, но прирост смеси к primary имеет CI
  **[-0.129; +0.298]**, поэтому это новый shadow-challenger, не замена.
- Causal error-regime stack поднимает точку ещё выше: смесь 75% primary + 25%
  regime-logit даёт **1.992 / 1.872**, combined **1.941 / rate 1.254** и
  **+81.7 б.п.** Прирост выгоды подтверждён CI **[+0.5; +24.0]**, но CI lift
  **[-0.011; +0.249]** пересекает ноль и один квартал имеет rate 0.98.
- На полном 2017–2026 lifecycle смесь primary 75% + convolution 25% даёт
  **lift 1.766 / rate 1.175**, min-year **1.522**, min-currency **1.711** и
  все десять годовых гейтов. Однако paired gain к primary всего **+0.021**,
  CI **[-0.050; +0.096]**: переносимость есть, превосходство не доказано.
- Старая `stack50_benefit50` остаётся полезным baseline: **1.502 / 1.314** при
  **1.25 / 1.41**, но теперь заметно уступает модели с лаговым рынком.
- Все эти числа честны по информационному потоку, но 2025-2026 уже много раз
  просматривались; это retrospective research, а не новый pristine holdout.
- Полная новая картина: [round-6 отчёт](results/research/round6/report.md).
- Если следующий эффективный курс ЦБ уже опубликован, выбранная на ранних
  блоках условная модель даёт **lift 2.459**; лучший ретроспективный финалист —
  **2.553**. Это другой момент принятия решения, не обычный прогноз.
- Logistic regression из `version_b` после устранения future-test top-K даёт
  **1.307–1.347** на просмотренном 2022–2026 при допустимой частоте, но на
  длинной истории причинные варианты снижаются до **1.085–1.225**. Это
  recent-regime challenger, а не подтверждённый стабильный lift 1.450;
  bootstrap-интервалы честных recent-вариантов пересекают порог 1.30.

## Что именно предсказывается

Для TJS, UZS, KGS, AMD и KZT target `fav_h5` равен 1, если сегодняшний
нормированный курс не выше каждого из следующих пяти опубликованных курсов:

```text
fav_h5(t) = 1, если v[t] <= min(v[t+1], ..., v[t+5])
```

Меньший курс выгоднее отправителю рублей. `h=5` - пять публикаций ЦБ, а не пять
календарных дней.

Основная метрика - future-only lift. Дополнительно считаются:

- future-only выгода относительно следующих пяти публикаций;
- симметричная выгода относительно окна `t-5 ... t+5` из условий кейса;
- частота сигналов;
- минимум по годам и валютам;
- устойчивость к множественному перебору.

## Лидеры по разным критериям

| Критерий | Подход | Результат | Как трактовать |
|---|---|---:|---|
| Официальный multi-horizon scorecard | Label-free geometry 75% + lagged CBA basis 25% | **min 1.623 / mean 1.855** | h=1/3/5/10/20; symmetric benefit везде положительный |
| Лучший point minimum по всем h | Minimum incumbent/futures agreement | **min 1.659 / mean 1.834** | h1/h3 выше, но paired superiority не доказано |
| Основной balanced pass | CNY logit 50% + CNY ExtraTrees 50% | **1.838 / 1.850** | 2025 / 2026; rate 1.31 / 1.33; все кварталы проходят |
| Максимальный combined point | Primary 75% + causal error-regime logit 25% | **1.992 / 1.872** | combined 1.941; rate 1.254; benefit gain подтверждён |
| Convolution path challenger | Primary 75% + fixed CNY-convolution logit 25% | **1.990 / 1.819** | combined 1.911; rate 1.204; paired lift gain не доказан |
| Лучший uncertainty-expert | Shrunk causal reliability LCB | **1.981 / 1.807** | combined 1.897; rate 1.16; независимая модель |
| Smooth additive challenger | Primary 75% + global all-spline GAM 25% | **1.867 / 1.894** | combined 1.892; rate 1.221; gain статистически не доказан |
| Полный deployment lifecycle | Expanding → hard reset после 2 000 resolved rows | **1.745** | 2017–2026; min year 1.546; rate 1.201 |
| Path-enhanced lifecycle challenger | Primary 75% + convolution 25% | **1.766** | 2017–2026; min year 1.522; rate 1.175 |
| Shock-insured lifecycle, research-only | В 2022–2023 CNY 60% + past-range anchor 40% | **1.784** | min year 1.573; min currency 1.726; grid-found |
| Low-dose local challenger | Primary 75% + local-currency consensus 25% | **1.873 / 1.838** | rate 1.26 / 1.34; combined 1.867; paired gain пока не доказан |
| Линейный partial-pooling fallback | Global logit 50% + hierarchical interaction logit 50% | **1.676 / 1.659** | rate 1.26 / 1.50; combined 1.697 |
| Лучший объяснимый новый | 19-feature CNY + anchor logit | **1.658 / 1.643** | rate 1.30 / 1.48; min currency 1.524 |
| Лучший один нелинейный | CNY intraday ExtraTrees | **1.792 / 1.758** | rate 1.29 / 1.30 |
| Подтверждённый свежий path-expert | 20-session CNY waveform ExtraTrees | **1.782 / 1.827** | combined 1.827; rate 1.216; stale20 только 1.248 |
| Предыдущий balanced pass | Stack 50% + benefit ranker 50% | **1.502 / 1.314** | 2025 / 2026; rate 1.25 / 1.41 |
| Лучший minimum-year classification | Primary 75% + shared-horizon Extra 25% | **1.397 / 1.374** | rate 1.23 / 1.42; 2025 benefit -2.6 bps |
| Строгий cadence near-pass | Та же score, quarter-reset threshold | **1.289 / 1.446** | min quarter rate 1.40 / 0.93 |
| Максимальный новый annual lift | Broad CBR 75% + baseload 25% | **1.429 / 1.670** | сильная квартальная разреженность |
| Стабильный причинный stack | Resolved ExtraTrees stack | **1.411 / 1.470** | min currency 1.305 / 1.335 |
| Объяснимый business ranker | Pairwise benefit ranker + anchor | **1.395 / 1.334** | lift и выгода положительны в оба года |
| Заранее зафиксированный final | Multiscale anchor | **1.295** | главный честный benchmark |
| Post-2022 causal challenger | Reset Hist + anchor 50/50 | **1.463 / 1.391** | 2025 / partial 2026; формальный pass, но clustered |
| Максимальный headline | Online local Hedge | **1.434** | retrospective; macro-year 1.251 |
| Лучшая posthoc-формула | Trend anchor | **1.406** | гипотеза, придуманная после просмотра final |
| Простой ансамбль | Equal mix 6 experts | **1.287 / 1.328** | shock / retrospective final |
| Устойчивый regime router | Soft router | **1.243 / 1.325** | лучше equal mix по минимальному году |
| Лучший отдельный ML на shock | Global ExtraTrees | **1.367** | final снизился до 1.254 |
| Короткое окно | ExtraTrees 2/3/5y mix | **1.384** | частота 0.864 ниже требования |
| Перенос между режимами | Geometric consensus | **1.375 / 1.189 / 1.264** | general / shock / final |
| Современный режим | Post-2022 reset XGB | **1.288** | самый ровный профиль 2024-2026 |
| После публикации курса | Conditional ExtraTrees | **2.632 / 2.395 / 2.459** | general / shock / retrospective final |
| После публикации, max finalist | Logit + ExtraTrees | **2.553** | retrospective; не выбирать по этой цифре |
| Новый Markov-state | Directional Markov + anchor | **1.316** | частота 0.961; не проходит полосу |
| «Окно закрывается» | Upper-range rule | **1.182** | retrospective best finalist |

### Почему 1.434 не объявляется победой

Online Hedge распределяет сигналы между годами неравномерно:

| Год | Lift | Сигналов / валюта / неделя |
|---|---:|---:|
| 2024 | 1.288 | 3.09 |
| 2025 | 1.043 | 0.48 |
| 2026 | 1.422 | 1.33 |

Модель часто стреляет в более удобном 2024 году и почти молчит в трудном 2025.
Aggregate lift получается 1.434, но равновзвешенный средний по годам - 1.251.

## Идеи и признаки, давшие максимальный boost

### 1. Положение курса в прошлом диапазоне

Самая сильная и устойчивая идея - не сложная модель, а положение текущего курса
относительно прошлых минимумов и максимумов:

- `pct_range_30`;
- `pct_range_90`;
- `pct_range_180`.

Locked multiscale anchor:

```text
0.5 * pct_range_90 + 0.3 * pct_range_30 + 0.2 * pct_range_180
```

Почему работает: target спрашивает, находится ли сегодняшний курс около будущего
минимума. Низкое положение относительно нескольких прошлых диапазонов является
простой причинной аппроксимацией этой геометрии.

`pct_range_90` не смотрит вперёд: используются текущая и предыдущие публикации.

### 2. Несколько масштабов вместо одного окна

Комбинация 30/90/180 оказалась устойчивее единственного окна. Короткое окно
быстрее реагирует на режим, длинное уменьшает шум.

### 3. Глобальное обучение по пяти валютам

Общий компонент движений валют заметен во всех блоках. Global ExtraTrees и
другие pooled-модели часто сильнее отдельных моделей на каждой валюте.

Полезные признаки:

- идентификатор валюты;
- общий фактор движений;
- отклонение валюты от общего фактора;
- относительные тренд и волатильность;
- past-only признаки USD/CNY как рыночный контекст.

При этом механическое добавление cross-sectional rank пяти валют не перенеслось
через 2022 год.

### 4. Разнообразный ансамбль и простые веса

Простая смесь логистических, деревьев, локальных, глобальных, rank и
trajectory-моделей оказалась устойчивее сложного обучаемого gate. На небольшом
числе независимых режимов learned router переобучается.

Практический приём: сначала нормировать scores каждого эксперта причинными
percentile ranks, затем использовать equal/geometric mean и только слабую
режимную поправку.

### 4a. Broad CBR panel, прямой benefit ranking и причинный stack

Round 6 добавил три новых источника устойчивости:

- 20 нецелевых валют из официального архива ЦБ: общий RUB-фактор,
  ex-USD-фактор, breadth, dispersion и относительное движение target;
- XGBoost pairwise ranker, который ранжирует дни по будущей достижимой выгоде
  внутри `currency x quarter`, а не оптимизирует среднюю вероятность;
- ExtraTrees meta-model над строго prequential прогнозами пяти экспертов,
  disagreement/tail-признаками и только уже разрешившимися `h=5` исходами.

Самый сильный компромисс получился от заранее фиксированной смеси score
`50/50`: stack отвечает за устойчивость классификации, benefit ranker — за
бизнесово полезный хвост. Policy после обучения не знает будущих target:
сегодняшний score сравнивается только с предыдущими 60 score своей валюты.

### 4b. Предыдущая сессия CNY/RUB на MOEX

Самый большой новый boost без курса ЦБ на завтра дал рынок `CNYRUB_TOM`.
Для решения на дату `t` используются только торги с `TRADEDATE < t`: close
текущего дня запрещён. Полезнее всего оказались прошлые close относительно
WAP, open-to-close движение, intraday range, overnight gap и активность торгов.

Matched ablation при одной модели и одном пороге:

- без MOEX: combined lift **1.310**;
- только CNY: **1.743**;
- тот же CNY с задержкой на 20 строк: **1.315**;
- только intraday CNY: **1.776**.

Это сильный аргумент, что работает актуальное состояние рынка, а не случайная
сложность матрицы. 19-признаковая логистика сохраняет combined lift **1.673**,
а её фиксированный rank-ансамбль 50/50 с ExtraTrees даёт **1.846**. Пересечение
их сигналов имеет lift **1.996**, но слишком редко; ансамбль сохраняет нужную
частоту. Все результаты ретроспективны и требуют будущей замороженной проверки.

Cross-era аудит уточнил роль СВО. В 2022Q3 lift временно падает до **1.135**,
но long-history модель восстанавливается до **1.861** уже в Q4. Ранний hard
reset вреден из-за малого числа новых label. После накопления данных картина
меняется: на 2025/2026 all-history даёт **1.778 / 1.690**, повышенный вес
post-2022 ×3 — **1.713 / 1.822**, hard reset — **1.838 / 1.850**. Значит старую
историю разумно сохранять в переходе, а зрелую модель обучать на новом режиме.

Отдельный `MOEX-CBR CNY basis` не принят: aligned lift **1.734**, а запоздавший
на 20 строк negative control **1.841**. Это медленное состояние режима, не
доказанный свежий сигнал.

### 4c. Локальные модели валют и глобальное сглаживание

Проверена исходная идея: для каждой из пяти target-валют отдельно обучаются
простые CNY-logit и ExtraTrees, а глобальная модель по всем валютам служит
стабилизирующей базой. На каждом квартальном refit используются только уже
разрешившиеся строки после 24.02.2022; минимум — 140 строк на валюту.

Лучше всего сработала маленькая поправка: **75% основного глобального CNY
консенсуса + 25% локального logit/ExtraTrees консенсуса**. Результат:

- 2025: lift **1.873**, rate **1.262**;
- 2026: lift **1.838**, rate **1.344**;
- combined: lift **1.867**, rate **1.265**, выгода **+76.5 б.п.**;
- минимальный lift валюты **1.592**, все квартальные lift выше 1.30.

Модель выбрасывает 29 слабых сигналов primary со средней выгодой **-30.1 б.п.**
и добавляет 21 локальный сигнал со средней выгодой **+63.3 б.п.**. Но изменение
не прошло парную проверку: разница lift **+0.021**, bootstrap 95% CI
**[-0.043; +0.095]**, вероятность отсутствия улучшения **0.302**. Вес 75/25
заморожен; дальше его можно честно оценивать только на новых датах.

Одностадийная линейная версия partial pooling тоже проверена: 14 общих
числовых коэффициентов, 5 валютных intercept и 70 L2-сжатых взаимодействий
`валюта × признак`. Она даёт **1.667 / 1.657**; смесь 50/50 с глобальной
логистикой — **1.676 / 1.659**. Это аккуратный объяснимый fallback, но смесь с
primary снижает combined lift до **1.829**. Значит выигрыш локального challenger
идёт не только от разных линейных коэффициентов, а от нелинейной структуры.

### 4d. Независимый контекст MOEX — полезные отрицательные результаты

IMOEX, RGBI, RUSFAR и GLDRUB_TOM были добавлены строго по последней завершённой
сессии. Aligned модели дали combined lift **1.789–1.797**, но те же признаки с
задержкой 20 target-строк дали **1.801–1.831**. Значит они кодируют медленный
режим, а не доказанный свежий сигнал. All-context упал до **1.737**.

Производные признаки свечи CNY — pressure, положение close/WAP, тело, тени и
z-score — тоже не приняты: aligned **1.770**, stale20 **1.860**. Эти поля и
контекст сохранены в коде как отрицательные эксперименты, но в primary не входят.

### 4e. Smooth additive GAM

Новый класс модели заменяет пороги деревьев гладкими одномерными кривыми:
квадратичные quantile-splines с 5 фиксированными knots и L2-logistic regression.
Сырые входы те же честные 19 признаков: 8 полей предыдущей CNY-сессии, 6
past-only range/return и 5 индикаторов валюты.

Сам global all-spline GAM даёт **1.688 / 1.640**. Главное — он ошибается иначе,
чем primary. Фиксированная смесь `75% primary + 25% GAM` получает:

- 2025: lift **1.867**, rate **1.219**;
- 2026: lift **1.894**, rate **1.294**;
- combined: **1.892**, rate **1.221**, выгода **+80.4 б.п.**;
- min-currency **1.639**, min-quarter rate **1.005**.

Она убирает 51 primary-сигнал с lift **1.318** и выгодой **-13.4 б.п.**, добавляя
24 сигнала с lift **1.711** и выгодой **+87.5 б.п.** Но замен мало: paired gain
**+0.046**, CI **[-0.046; +0.153]**, max-adjusted `p=0.182`. Это сильнейший
research challenger по point estimate, но не доказанная замена primary.

### 4f. Causal reliability surface

Это не ещё один прогноз курса. Для текущего дня берутся causal ranks primary и
GAM и их disagreement. Модель ищет ближайшие похожие примеры только среди
исходов, у которых уже успели разрешиться все пять будущих публикаций.

Для `fav_h5` считается нижняя оценка Beta posterior `mean - 1 sd`, для выгоды —
`mean - 1 standard error`. Локальные 80 соседей валюты сжимаются к глобальным
250 соседям, поэтому редкий коридор не переопределяет решение.

Лучший `shrunk_hit_lcb`:

- 2025: lift **1.981**, rate **1.175**, выгода **+61.5 б.п.**;
- 2026: lift **1.807**, rate **1.195**, выгода **+118.1 б.п.**;
- combined: **1.897**, rate **1.157**, min-currency **1.737**;
- каждый квартальный lift выше **1.52**.

Pooled-вариант чуть слабее (**1.856** combined), но держит каждый квартальный
rate не ниже **1.002**. Физическая порча всех неразрешившихся будущих targets и
benefits не меняет ни одного более раннего score.

Замена primary не разрешена: shrunk-LCB слабее в 2026, paired gain combined
**+0.051**, CI **[-0.089; +0.268]**, max-adjusted `p=0.321`. Это новый
независимый uncertainty-expert для будущей проверки, а не выбранный задним
числом winner.

### 4g. Исторические аналоги сырых траекторий

Отдельно проверен полностью model-free подход: каждый квартал ищутся похожие
уже разрешившиеся участки по multiscale target returns/ranges/volatility и
предыдущей CNY-сессии. Scaling fit только на train; локальные аналоги валюты
сжимаются к глобальным.

- target-only: lift **0.832 / 1.250** — форма самого курса не переносится;
- CNY-only: **1.752 / 1.500**;
- joint target+CNY: **1.803 / 1.554**, combined **1.677**;
- primary 75% + joint 25%: **1.852 / 1.794**, то есть хуже primary в 2026.

Вывод: эффект CNY реален даже в простом analogue search, но сырая евклидова
близость 25 координат хуже обученного сжатия logit/ExtraTrees/GAM. Семейство
сохранено как понятный negative control; менять размер neighbourhood по этим
годам не будем.

### 4h. Ordered waveform последних 20 сессий CNY

Вместо евклидова поиска аналогов сохранён порядок 20 последних доходностей
`CNYRUB_TOM`. К ним добавлены 8 фиксированных DCT-II коэффициентов, средние и
волатильности 5/10/20, downside/upside volatility, skew, autocorrelation,
доли положительных шагов, смены знака, run-up/drawdown и ускорение. Все сессии
строго раньше даты сигнала; same-day close не используется.

- waveform logit: **1.498 / 1.794**, combined **1.630**;
- base + waveform ExtraTrees: **1.782 / 1.827**, combined **1.827**, rate
  **1.216**, min-currency **1.567**;
- идентичная модель с waveform, задержанным на 20 строк: combined **1.248**;
- primary 75% + waveform logit 25%: **1.864 / 1.831**, combined **1.860**.

Парный аудит был объявлен до расчёта разниц. Свежий waveform уверенно сильнее
stale-контроля: **+0.579 lift**, 95% CI **[+0.285; +0.937]**, Holm
`p=0.001`. Но смесь не превосходит frozen primary надёжно: **+0.013**, CI
**[-0.086; +0.126]**, Holm `p=0.349`. Поэтому waveform — подтверждённый
независимый источник информации для будущего shadow-теста, а не повод менять
primary по уже просмотренным датам.

### 4i. Fixed random-convolution motifs по CNY

64 неизменных случайных свёртки длины 3/5/7/9 извлекают из тех же 20 прошлых
CNY-доходностей максимум отклика и долю положительных откликов. Seed, число и
длины kernels зафиксированы до результатов. Поверх 128 нелинейных откликов,
waveform, валюты и шести past-only anchors обучается L2-logistic regression.

- convolution logit: **1.216 / 1.222**, combined **1.209**;
- stale20 convolution logit: combined **0.884**;
- primary 75% + convolution 25%: **1.990 / 1.819**, combined **1.911**,
  rate **1.204**, выгода **+81.1 б.п.**, min-currency **1.624**;
- минимальный квартальный lift смеси **1.531**, rate **1.027**.

Свёрточный эксперт слаб в одиночку, но меняет ranking ровно в полезных местах:
61 новый сигнал имеет lift **2.204** и выгоду **+84.4 б.п.**. Свежесть
подтверждается относительно stale20: paired gain **+0.325**, CI
**[+0.056; +0.625]**, Holm `p=0.022`. Но общая смесь меняет мало строк, поэтому
её преимущество перед primary пока неопределённо: **+0.065**, CI
**[-0.129; +0.298]**, Holm `p=0.219`. Все параметры заморожены для будущей
проверки; подбирать другой вес по тем же годам нельзя.

### 4j. Causal error-regime stack

Это прямая реализация идеи «понять, где какая модель ошибается», но без
подглядывания в будущие ошибки. Берутся OOF-scores primary, waveform ExtraTrees
и convolution-logit. Каждый score превращается в percentile только по 250
предыдущим значениям той же валюты. Модель видит их disagreement, currency,
past-only anchors и waveform, а на каждом квартале обучается лишь на исходах,
для которых все пять будущих публикаций уже наступили.

- regime-logit: **1.803 / 1.744**, combined **1.764**;
- shallow regime-HistGB: **1.848 / 1.699**, combined **1.770**;
- primary 75% + regime-logit 25%: **1.992 / 1.872**, combined **1.941**,
  rate **1.254**, min-currency **1.658**, выгода **+81.7 б.п.**;
- 51 добавленный сигнал имеет lift **2.562**, а 64 удалённых primary-сигнала —
  lift **1.634**.

Результат близок к формальному доказательству, но не проходит его: paired lift
gain **+0.094**, CI **[-0.011; +0.249]**, Holm `p=0.129`. Зато benefit gain
**+10.4 б.п.** имеет полностью положительный CI **[+0.5; +24.0]**. В 2026Q2
частота **0.98**, поэтому cadence-гейт также формально не пройден. Модель
заморожена как главный новый prospective challenger; чинить порог по уже
увиденному кварталу нельзя.

### 4k. Глобальное дообучение остатка primary

Проверена исходная двухступенчатая идея: L2-logit причинно калибрует primary,
после чего HistGB или ExtraTrees по всем пяти валютам предсказывает только
остаток `y - base_probability`. Назад добавляется заранее фиксированная доля
25%; обучение на каждом refit видит только полностью разрешившиеся `h=5`.

- 25% HistGB residual: **1.866 / 1.845**, combined **1.835**;
- тот же HistGB со stale20 CNY-состоянием: combined **1.750**;
- 25% ExtraTrees residual: **1.864 / 1.864**, combined **1.853**, rate
  **1.289**, min-currency **1.573**.

ExtraTrees выглядит ровно, но почти не меняет primary: paired gain **+0.007**,
CI **[-0.080; +0.124]**, Holm `p=0.698`; выгода ниже на 0.9 б.п. Свежий HistGB
лучше stale на +0.084, но CI **[-0.019; +0.218]** также пересекает ноль.
Значит generic residual boosting не даёт устойчивого прироста; causal
error-regime stack полезнее, поскольку моделирует disagreement экспертов
напрямую.

### 5. Забывание старого режима

После 2022 полезны:

- короткие окна обучения;
- rolling/EWMA веса;
- полное исключение старых target-строк в reset-sensitivity.

Жёсткий бинарный признак `post_2022` сам по себе слабее контролируемого забывания.

Теперь это сведено в одну deployment lifecycle-политику. Expanding CNY-модель
работает в 2017–2023; переключение на hard reset происходит не «по факту СВО»,
а на первом квартальном refit с минимум **2 000** уже разрешившихся post-2022
строк — 01.01.2024. На всех десяти годах:

- combined lift **1.745**, rate **1.201**, выгода **+61.2 б.п.**;
- каждый год lift не ниже **1.546** и rate не ниже **1.061**;
- каждая валюта lift не ниже **1.677**;
- bootstrap 95% CI lift **[1.628; 1.861]**.

Ранний reset при 700 строках проваливает 2022 до lift **1.276**. Значит важна
не календарная метка сама по себе, а зрелость новой обучающей выборки.

Новые path-модели также прогнаны единым lifecycle. Convolution-logit обучается
expanding до 2023 и переключается на сохранённый post-2022 reset в 2024:

- convolution alone: lift **1.423**, но 2022 падает до **1.107**;
- primary 75% + convolution 25%: lift **1.766**, rate **1.175**, выгода
  **+65.4 б.п.**, min-year **1.522**, min-currency **1.711**;
- primary до 2023 + regime-blend с 2024: lift **1.756**, min-year **1.546**.

Обе смеси проходят пользовательский гейт во всех десяти годах, но не доказали
превосходство над lifecycle-primary: paired gain **+0.021**, CI
**[-0.050; +0.096]** для convolution и **+0.011**, CI
**[-0.019; +0.045]** для regime-handoff. Отдельные кварталы остаются редкими;
подгонять их порог задним числом нельзя.

Дополнительная страховка простым past-range anchor в 2022–2023 улучшает весь
профиль. Заранее проверенный вес 75/25 даёт lifecycle **1.770**. Sensitivity
grid показывает широкое плато: 4 из 5 соседних весов улучшают оба shock-года.
Grid-максимум 60/40 даёт:

- 2022 lift **1.756**, 2023 **1.622**;
- худший shock-квартал **1.302**, rate **0.992**;
- lifecycle lift **1.784**, rate **1.200**, выгода **+67.0 б.п.**;
- min-year **1.573**, min-currency **1.726**.

Но вес 60/40 найден после просмотра grid. Его paired CI прироста на shock
**[-0.013; +0.365]**, max-adjusted `p=0.056`; формальный заранее заданный
статистический гейт не пройден. Это сильная объяснимая гипотеза для следующего
шокового режима, а не новая доказанная primary.

### 6. Контроль рабочей частоты

Rolling threshold по прошлым scores не увеличивает качество модели напрямую,
но предотвращает ложную победу за счёт подгонки top-rate по тесту.

Главный новый диагностический приём - одновременно смотреть aggregate lift,
macro-year lift и годовую частоту. Именно так обнаружено завышение Online Hedge.

### 7. Факт публикации следующего курса

Самый большой абсолютный boost появился из изменения доступной информации:
после публикации следующий эффективный курс уже известен. Простой gate даёт
1.959, а условная модель, которая использует величину известного запаса и
пытается предсказать оставшиеся четыре публикации, даёт 2.459. Лучший
ретроспективный ансамбль достигает 2.553. Политика допустима только при
runtime-проверке факта публикации.

### 8. Markov-state и иерархические состояния

Round 4 добавил частично-пулированные empirical-Bayes таблицы режимов и модель
последовательностей up/flat/down. На 2024-2026 Markov-гибрид дал 1.316, но его
частота 0.961 ниже полосы, а на general/shock было лишь 1.112/1.175. Полный
cross-era diagnostic нашёл лучший минимальный lift только 1.129. Это новая
полезная recent-regime гипотеза, но не честное стабильное прохождение 1.30.

## Что не дало устойчивого выигрыша

| Эксперимент | Лучшее наблюдение | Почему не стал финальным |
|---|---:|---|
| Seasonal naive | AUC 0.458 | календарная сезонность не переносится |
| ETS | AUC 0.478 | прогноз уровня плохо соответствует barrier target |
| SARIMA | AUC 0.523 | сигнал слишком слабый |
| GRU | AUC 0.540 | сложность не окупилась на малом числе режимов |
| Local logit -> global XGB residual | shock 1.075 | второй слой переносит устаревшие ошибки |
| XGB ranker | general до 1.375 | shock/final около 1.0 |
| RUONIA/key rate/Brent/USD | shock до 1.032 | доступные lagged-прокси слишком грубы для h=5 |
| Direct barrier/path | final 1.224 | ошибки пяти прогнозируемых шагов накапливаются |
| Delayed resolved labels | shock 1.159 | trailing target-rate запаздывает за drift |
| Cross-sectional state | general 1.371 | shock 1.038 |
| Pooled discrete hazard | final 1.206 | последовательные hazard-ошибки накапливаются |
| Per-currency champions | general 1.342 | shock 0.888: переобучение к валюте и эпохе |
| Online logistic SGD | final 1.082 | линейность и шум delayed updates |
| Reset + anchor mixtures | final 1.221 | смешивание не исправило слабые режимы компонентов |
| IMOEX/RGBI/RUSFAR/gold context | aligned до 1.797 | каждый delayed-20 control оказался сильнее |
| Derived CNY microstructure | aligned 1.770 | stale20 1.860: режимный прокси, не свежий сигнал |
| Raw target trajectory analogues | 2025 lift 0.832 | форма target без CNY не переносится |
| Global HistGB/ExtraTrees residual boost | combined до 1.853 | paired gain к primary +0.007, benefit ниже |
| Seven-bank shadow-RUB consensus | screen min lift 1.472 | выбранная 30% смесь падает до min 1.544 против лидера 1.623 на 2025–2026 |
| Cross-bank uncertainty/veto | screen min lift 1.432 | лучшая causal confirmation переносится только до min 1.612 |
| Joint target+CBA+cross-bank ML | fresh logit 1.378 против stale20 1.181 на screen | внешняя информация свежая, но ни logit, ни HistGB, ни ExtraTrees не обошли лидера |
| Within-source cross-bank revisions | лучший допустимый screen min 1.005 | 2024 сразу оставляет incumbent; краткосрочные общие ревизии не предсказывают цель |
| Causally normalized cross-bank factor | standalone screen min 1.233 | 10% смесь переносится до min 1.593 против лидера 1.623 |
| One-sided target state-space | h20 1.914 против 1.879 у лидера | улучшает длинные h и symmetric benefit, но снижает h1 до 1.591 |
| Nonlinear incumbent/state agreement | fresh min 1.598 против stale 1.576 | timing настоящий, но единый trigger всё равно хуже лидера 1.623 |
| MOEX perpetual CNYRUBF/USDRUBF ExtraTrees | five-horizon min 1.534; h5 1.770/1.645 по годам | новый свежий эксперт: stale20 падает до min 1.064, но incumbent сильнее |
| Incumbent/futures minimum geometry | point min 1.659 против 1.623 | paired min-gain CI пересекает ноль; сохранить challenger, не продвигать |
| Delayed all-horizon MOEX weighting | point min 1.643 против 1.623 | fresh beats stale значимо, но gain к incumbent CI [-0.144; +0.081]; не продвигать |
| Noon-Moscow incumbent/HistGB rank mean | five-horizon min/mean 1.714/1.892 | лучший point score; min-gain CI [-0.137; +0.202], сохранить challenger |
| Noon/state balance + shared-horizon noon learner | 2024 retained noon consensus | state correction and five-horizon shared learner add variance, no transport claim |
| Noon spot ML | best screen min lift 1.528 | full boosting loses to the simpler signed basis and to the incumbent/noon consensus |
| Signed noon spot basis | screen min 1.644; later min 1.604; stale20 0.844 | genuinely fresh but regime-dependent partial-fixing signal |
| Online noon/spot weighting | later min/mean 1.714/1.898 | best point mean; paired mean-gain CI [-0.045; +0.076], no promotion |
| 15:30 CNY session-mean basis | later min/mean 1.708/1.883; stale20 min 0.782 | strong explainable timed product, especially h20=1.927 |

Отрицательные результаты сохранены намеренно: они не дают снова повторять те же
дорогие эксперименты и показывают, где именно нарушается переносимость.

## Сезонность и календарные признаки

Проверены:

- день недели и месяц;
- циклические `sin/cos` кодировки;
- начало/конец месяца;
- праздники стран-получателей;
- бинарные окна до/после Нового года и других дат;
- календарный разрыв между публикациями.

Сезонные признаки можно оставлять как слабые дополнительные переменные, но
самостоятельного устойчивого boost они не дали. Порядок сильных месяцев меняется
между временными блоками и валютами.

## Цена дня: две метрики нельзя смешивать

| Политика, 2024-2026 | Future-only, б.п. | Симметричная `+-5`, б.п. |
|---|---:|---:|
| Locked anchor | +35.0 | -26.9 |
| Equal mix | +38.2 | -1.4 |
| Soft router | +32.0 | +0.8 |
| Geometric consensus | +33.4 | +4.3 |
| Online Hedge | +42.2 | +7.4 |
| Reset XGB | +32.7 | +11.7 |
| After-publication known-next gate | +77.8 | +48.3 |
| After-publication selected ExtraTrees | +138.3 | не пересчитывалась в round 4 |

Future-only показывает достижимую выгоду относительно будущего. Симметричная
метрика включает пять уже прошедших публикаций и измеряет локальность минимума.

## Статистическая честность

- Четырёхнедельный block bootstrap сохраняет зависимость соседних target и пяти
  валют одного дня.
- Для CNY-консенсуса `logit50_extra50` bootstrap CI самого lift равен
  **[1.562, 2.137]** на 2025-2026; CI средней будущей выгоды
  **[+28.1, +107.6] bps**.
- Circular-shift max-adjusted p внутри заранее записанного packet AI:
  **0.00025** combined. Это поправка лишь по пяти кандидатам packet AI, а не по
  всему многомесячному перебору.
- Старый общий multiplicity-аудит включал 157 записанных политик и давал
  минимальное max-adjusted `p=0.067`; новый результат ещё не встроен в единый
  across-all-rounds аудит.
- 2024-2026 уже просмотрен, поэтому все новые результаты на нём retrospective.

## Что заморозить для настоящего будущего теста

1. Primary: `logit50_extra50`, target rate 22%, rolling 20, только
   `CNYRUB_TOM` с `TRADEDATE < signal_date`.
2. Low-dose local challenger: `primary75_local_consensus25`, target rate 22%,
   rolling 20; не менять вес 75/25 по уже просмотренным годам.
3. Explainable fallback: `market_anchor_logit`, target rate 22%, rolling 20.
4. Pre-MOEX baseline: `stack50_benefit50`, target rate 22%, rolling 60.
5. Benchmark: locked multiscale anchor, lift 1.295.
6. Stable legacy challenger: geometric consensus, target rate 20%, rolling 120.
7. Current-regime legacy challenger: post-2022 reset XGB, rate 20%, rolling 120.
8. Research-only: Online Hedge, пока не стабилизирована частота.
9. Отдельный продукт: after-publication policy со строгим timestamp gate.
10. Объяснимый recent-regime challenger: logistic regression `version_b` с
   `30m fit / 6m calibration / q20 / rolling-120`; проверять только вперёд.

Round 4 уточняет пункт 5: основной выбранный кандидат — conditional ExtraTrees
с target rate 22% и rolling threshold 250; ансамбль logit + ExtraTrees остаётся
challenger до нового настоящего holdout.

После даты фиксации нельзя менять их признаки, веса и пороги по новым данным, если
мы хотим получить настоящий holdout.

Для этого уже создан формальный shadow-протокол с cutoff **03.09.2026**:
`research/round6_prospective_shadow_protocol.md`. В нём primary и local
challenger обязаны считаться на каждой следующей дате параллельно. Хеши кода,
данных и сохранённых OOF outputs лежат в
`results/research/round6/prospective_freeze/manifest.json` и проверяются тестом,
поэтому незаметно изменить модель задним числом не получится.

## Где лежат полные результаты

### Быстрое чтение

- `output/pdf/ivan_deep_research_round3_full.pdf` - полный визуальный отчёт.
- `results/research/round3/report-source.md` - подробный текст отчёта.
- `results/research/checkpoint_2026-09-04.md` - что было известно до round 2.
- `results/research/round4/report.md` - post-publication, Markov-state и
  window-closing продолжение.
- `results/research/round5/report.md` - post-2022 reset и quarterly-anchor
  эксперименты.
- `results/research/round6/report.md` - текущий CNY/MOEX balanced pass, широкий
  CBR panel, direct ranking, causal stacking и cadence-аудит.
- `results/research/version_b_honest_audit/report.md` - причинная перепроверка
  заявленного lift 1.450 из ветки `version_b`.

### Главные сводные таблицы

- `results/research/round3/master_policy_metrics.csv` - политики и периоды.
- `results/research/round3/master_final_breakdown.csv` - годы и валюты.
- `results/research/round3/round3_block_bootstrap.csv` - интервалы.
- `results/research/round3/round3_circular_shift_multiplicity.csv` - поправка за
  перебор 157 вариантов.
- `results/research/publication_timing_h1.csv` - after-publication для всех `h`.

### Полный журнал семейств

Для каждого направления сохранены:

- `*_general_2017_2020.csv` - первичная проверка;
- `*_stage1.csv` - выбор рабочей точки;
- `*_stage2_2022_2023.csv` - перенос в shock;
- `*_final_2024_2026_retrospective.csv` - последний ретроспективный срез;
- `*_protocol.json` - параметры и ограничения;
- `*_outputs.pkl` - сохранённые OOF scores для повторного аудита и ансамблей.

Семейства находятся в `results/research/round2/` и
`results/research/round3/`; продолжение — в `results/research/round4/`,
`results/research/round5/` и `results/research/round6/`. Ничего из них не
выброшено.

## Проверки

- Все 109 тестов прошли.
- Новые тесты отдельно проверяют соответствие post-publication gate цели h=1,
  точное использование строки `i+1` и purge h=5 до калибровочного года.
- MOEX-тест физически меняет same-day/future цены и подтверждает, что прошлые
  признаки не меняются; отдельно проверена причинность 20-строчной задержки.
- Для 15:30 fixing-proxy дополнительно проверено, что mean-close побитово
  совпадает с frozen feature, пропущенные торговые сессии остаются допустимыми
  нулевыми score, а не исчезают из base-rate знаменателя.
- Та же физическая проверка добавлена для независимого контекста MOEX и
  производных CNY microstructure-признаков.
- Отдельный тест физически портит same-day/future CNY-сессии и подтверждает
  битовую неизменность всех прошлых waveform/DCT-признаков.
- Та же проверка отдельно покрывает фиксированные random-convolution признаки.
- Для regime-stack проверяется выбор own-year OOF test-score вместо повторно
  нормированного calibration-score следующего года.
- Lifecycle-тест подтверждает, что regime-handoff не меняет ни одного score до
  заранее заданной даты 01.01.2024.
- NBG/NBRB-тесты проверяют единицы, совпадение календарей RUB/USD/CNY и
  неизменность прошлого при физической порче будущего; отдельно покрыты
  exponential-threshold и weekly-cap состояния.
- PDF round 3 визуально проверен на всех 16 страницах; PDF round 4 — на всех 6.

## T18: verified receipt gate для product runtime

T17 показал, что физическая доступность рыночного источника и плановый clock —
не одно и то же. T18 применяет тот же принцип к публикации ЦБ. Историческое
допущение 18:30 сохранено для явного research replay, но обычный API/CLI больше
не переключает модель по часам. Нужен фактически зарегистрированный
`verified_receipt_at` текущего московского дня.

На сохранённом примере 01.09.2026:

- запрос 18:45 без receipt возвращает для всех валют последнюю модель 17:30;
- receipt 18:42 разрешает after-publication CBR-строку с source time 18:42;
- receipt 19:02 не позволяет плановой market-строке 19:00 появиться раньше
  19:02;
- future/wrong-day/timezone-free events отклоняются;
- вероятности, expected bps и AP37 `push_now` не переобучаются и не меняются.

Это не accuracy-experiment и не новый holdout, а ремонт причинной доступности
для онлайн-системы. Результаты и независимый rebuild-аудит лежат в
`results/research/temperature/t18_verified_receipt_gate/`; полный набор после
изменения содержит 352 проходящих теста.

## T19: единый аудит температуры в любое время

T19 проверяет финальный T17/T18-router на одной сетке, а не сравнивает метрики
разных экспериментов на разных часах. Сетка содержит 967 календарных дней,
20 московских моментов, пять валют и два receipt-сценария: 193 400 запросов.
Для каждого запроса доказано `snapshot/source_at <= as_of`; сценарий без
receipt не получает ни одного same-day after-publication состояния.

- probability проходит двойной 20/50-date paired gate в 111 из 200 состояний;
- expected future-only benefit проходит в 119 из 200;
- h5 становится устойчиво сильнее past baseline примерно с 11:45;
- h20 не проходит ни один из 40 state gates;
- pooled ECE не выше примерно 0,050, но 1 629 из 3 000 currency-year срезов
  имеют ECE > 0,08.

Итог не использован для post-hoc перенастройки открытого 2024–2026. Следующий
кандидат должен заранее зафиксировать иерархическую causal recalibration и
отдельную long-horizon голову на ранних mature данных. Независимый rebuild-аудит
и полный набор из 355 тестов проходят.

## T20: one-shot иерархическая калибровка не переносится

Primary `hierarchical_beta` был зарегистрирован до расчёта. Он обучается на
созревшем 2024 с двухдневным embargo и один раз применяется к открытому
2025–2026. Внутри каждого `scenario × clock × h` используются base probability,
валюта и causal regime; AP37, benefit и availability route не меняются.

Результат: 0 из 120 строгих state gates. На h5/h10 средний Brier ухудшается на
0,00591/0,00394. На h20 Brier point-wise улучшается на 0,00030, но log-loss и
ECE хуже, а 20/50-date bootstrap не проходит. Currency-year-clock-h строк с
ECE > 0,08 становится 816 против 789 у frozen identity.

Модель не продвигается. Это показывает, что h20 требует новой discrimination
головы, а локальный drift нельзя вылечить единственным fit на 2024. Полные
результаты: `results/research/temperature/t20_hierarchical_calibration/`.

## T21: cross-horizon h20 head нашла rank-сигнал, но не probability

T21 был зарегистрирован до расчёта и использовал только causal h1/h3/h5/h10/h20
probability/benefit-кривые, валюту, relative cross-currency признаки, возраст и
режим. Base заканчивается до 01.09.2024, отдельная calibration — до 2025;
обе части содержат только созревшие h20-метки с двухдневным embargo. Открытый
2025–2026 остаётся diagnostic.

Primary HGB+Platt прошёл 0/40 gates: средний Δ AUC +0,010, но Δ Brier +0,06289,
Δ log-loss +0,15102 и Δ ECE +0,20374. Он отклонён. Не выбранная primary
контрольная logistic-модель показала важный механизм: AUC вырос с 0,576 до
0,682 и улучшился во всех 40/40 состояниях, но Brier ухудшился до 0,443 из-за
сильного сдвига уровня вероятности. Поэтому score можно исследовать как малую
rank-поправку к frozen anchor, но нельзя показывать пользователю как 0–100.

Независимый аудит подтверждает 116 400 evaluation-строк, maturity, embargo,
`source_at <= query_at`, полный bootstrap-grid и независимость predictions от
evaluation labels. Полный набор: 359 тестов. Результаты:
`results/research/temperature/t21_h20_curve_head/`.

## T22: calibration-constrained rank correction разделила receipt-фазы

T22 оставил frozen h20 logit якорем и добавлял только малую остаточную часть
cross-horizon rank. Вес из 0/0,01/0,025/0,05/0,10/0,20 выбирался на disjoint
конце 2024 под жёсткими ограничениями Brier/log-loss/ECE; 2025–2026 не
участвовал в выборе.

Как единая модель подход отклонён: средний AUC 0,576→0,569, Brier
0,11828→0,11897, всего 6/40 gates. Но прошли ровно все шесть
`calendar_assumed_replay` states с 18:45 до 23:15, то есть состояния после
появления нового announced CBR. На них AUC 0,534→0,671, Brier
0,12058→0,11379, log-loss 0,40723→0,38313; ECE не ухудшился. В
`no_same_day_receipt` поправка вредна.

Следствие: новый CBR-record действительно меняет полезное h20-состояние, но
router обязан включать challenger по фактическому `verified_receipt_at`, а не
по 18:00/18:30. Из-за открытого теста и несертифицированной истории receipt это
только frozen shadow, не production promotion. Аудит подтверждает 116 400
строк, maturity, source-time, alpha rebuild и target corruption invariance.
Результаты: `results/research/temperature/t22_h20_rank_correction/`.

## T23: delayed online mapping не переносится

T23 обновлял h20 mapping раз в месяц и видел только полностью созревшие исходы
до monthly origin минус embargo. Внутри максимум 200 полных дат делились 75/25
по времени; trailing screen выбирал identity, anchor-logit или joint-logit.
Текущий месяц не участвовал ни в fit, ни в выборе.

Итог 0/40 gates. Selector оставил identity в 619 из 800 месячных состояний,
выбрал joint 156 раз и anchor 25 раз. Средний AUC 0,576→0,554, Brier
0,11828→0,12020, log-loss 0,39694→0,40635; high-ECE строк 266 против 240.
Даже зрелая история остаётся слишком шумной для частого выбора mapping.

After-receipt point-deltas сохраняют правильный знак, но bootstrap пересекает
ноль; frozen T22 с 6/6 gates сильнее. Решение: не переобучать receipt mapping
ежемесячно, держать T22 frozen shadow и искать новые pre-receipt observables.
Результаты: `results/research/temperature/t23_h20_delayed_online/`.

## T24: новый history-only h20 rank переносится, mapping пока нет

T24 проверил новые multiscale daily observables вместо очередной калибровки.
Model fit заканчивается до 2024, calibration и selection разнесены по двум
частям 2024, evaluation 2025–2026 открыт. Выходные держат score последней
публикации и не создают нулевые returns.

Selector сохранил identity, потому что compact Platt на selection дал Brier
0,21555 против 0,20538 и ECE 0,14093. При этом rank compact-logit был сильным
до evaluation: AUC 0,742. На открытом 2025–2026 он сохранил AUC 0,702 против
0,374 baseline, Brier 0,11744 против 0,12429, AUC 0,662–0,743 по валютам и
0,616/0,699 по годам.

Это первый сильный pre-receipt h20 discrimination signal после T19, но не
production temperature: formal primary не прошёл, а 2026 local calibration
дрейфует. Compact сохраняется как frozen rank shadow; следующий mapping должен
быть зарегистрирован без подбора по открытому evaluation. Результаты:
`results/research/temperature/t24_history_h20_anchor/`.

## T25: anchor-preserving blend улучшает rank, но не проходит Brier-CI

T25 был зарегистрирован до расчёта результатов. Компактная T24-logistic снова
обучена только до 2024; 2024-H1 использован для карты, 2024-H2 — для выбора из
фиксированных residual/copula/permutation-кандидатов, 2025–2026 — только
открытая диагностика. Selector выбрал `residual_a040`: 40% стандартизованного
добавочного rank-сигнала в logit старой h20 temperature.

На 2 910 строках 2025–2026 AUC вырос 0,374→0,563, average precision
0,102→0,209, Brier улучшился 0,12429→0,11825, log-loss 0,42016→0,40454 и ECE
0,03608→0,03326. AUC-delta устойчиво положительна при 20/50-date bootstrap, но
верхние границы Brier-delta равны +0,00088/+0,00244. Формальный `passed=false`:
нельзя ослаблять заранее заданный gate после просмотра результата.

Global copula сохранила полный AUC 0,702, но дала ECE 0,122; daily permutation
точно сохранила набор пяти дневных вероятностей, однако дала лишь AUC 0,393.
В 2025 residual blend улучшает calibration, в 2026 недооценивает новый base
rate. Вывод: rank полезен, а абсолютная вероятность режимно дрейфует. T25
сохранён как premarket shadow, production temperature не изменена. Результаты:
`results/research/temperature/t25_anchor_preserving_map/`.

## T26: causal delayed intercept не выбран

T26 обновлял только общий logit-intercept T25 по уникальным CBR-событиям,
h20-outcome которых полностью созрел до текущего дня минус embargo. В
пререгистрации заморожены окна 30/60/125/250/expanding, ridge/clipping и один
hierarchical control. Календарные holds и выходные feedback не размножали.

На selection-2024-H2 w30 улучшил Brier 0,17693→0,17479 и AUC 0,716→0,767, но
ECE ухудшился на +0,01058 при gate +0,005. Остальные варианты не прошли
совместный Brier/log-loss/ECE/AUC gate. Selector оставил T25; formal
`passed=false`.

На открытом 2025–2026 w250 диагностически дал AUC 0,638, AP 0,302, Brier
0,11326 и log-loss 0,38634 против 0,563/0,209/0,11825/0,40454 у T25. Он
улучшает Brier в обоих годах и пяти валютах, но до 2025 практически совпадал с
expanding: длина 250 стала различимой только на открытом периоде. Поэтому w250
зафиксирован как post-hoc гипотеза, не выбранная модель. Нужна rolling-origin
история до 2025 или prospective shadow. Результаты:
`results/research/temperature/t26_delayed_base_rate/`.

## T27: 2023 rolling-origin история опровергает w250

T27 восстановил 1 235 строк / 247 publication dates 2023. Frozen anchor взят
из квартальных OOS-выходов T4; compact rank заново обучен четырьмя quarterly
fits только на прежних mature labels с embargo. Это позволило различить
w125/w250/expanding до 2025, не используя открытый период для selector.

w250 провалился на 2024-H2: AUC 0,699, Brier 0,22728, log-loss 0,73115 и ECE
0,20406 против 0,716/0,17693/0,53047/0,07575 у T25. Ни один candidate не
прошёл gate; selector оставил T25. Open w250 всё ещё выглядит сильно
(AUC 0,644, Brier 0,11303), но теперь показано, что он не переносится назад и
является режимным post-hoc эффектом.

w30 второй раз улучшил pre-2025 AUC/Brier до 0,767/0,17479, но ECE delta
+0,01058 не прошла лимит +0,005. Следующий допустимый тест — заранее заданный
слабый logit-shrink к w30, выбранный только на 2024-H2; w250 исключён.
Результаты: `results/research/temperature/t27_rolling_origin_history/`.

## T28: weak w30-blend не проходит отдельный Q4

T28 проверил beta 0,10/0,20/0,30/0,40/0,50 в формуле
`T25_logit + beta*(w30_logit-T25_logit)`. Только зрелая часть 2024-Q3 выбирала
вес; зрелая Q4 была отдельной frozen validation. 2025–2026 не участвовал.

Q3 выбрал beta 0,50: AUC 0,578→0,621, Brier 0,10820→0,10016, ECE
0,07005→0,03393. Но Q4 дал другой режим target-rate 29,1%; выбранный вес
улучшил Brier/log-loss/ECE, однако AUC упал на 0,03794. Даже beta 0,10 потеряла
0,00772 при gate 0,005. Validation failed, итоговая модель равна T25.

Причина: daily intercept различается по датам и поэтому меняет междневной rank,
хотя внутри каждого дня является общей монотонной поправкой. Open beta 0,10
чуть улучшает Brier, но ECE delta +0,00534 также едва превышает gate; подбирать
0,09 post-hoc запрещено. Следующий тест должен замораживать intercept на более
крупный период. Результаты:
`results/research/temperature/t28_weak_w30_blend/`.

## T29: месячный/квартальный intercept не проходит перенос

T29 заранее заморозил 10 кандидатов: граница месяц/квартал, 30/60 последних
уникальных созревших publication dates и дозы 0,25/0,50/1,00. Поправка logit
считалась только на границе периода с h20-maturity и двухдневным embargo, а
внутри периода оставалась константой. 2025–2026 не участвовал в выборе.

Зрелая часть 2024-Q3 выбрала `month_w30_b100`: AUC 0,57766→0,66686, Brier
0,10820→0,09662, log-loss 0,36959→0,33316 и ECE 0,07005→0,01324. Но на
отдельном Q4 target rate сменился с 11,25% до 29,06%; Brier/log-loss/ECE
продолжили улучшаться, а AUC упал 0,70020→0,65535. Delta -0,04486 нарушает
заранее заданный предел -0,005. Validation failed, итоговая модель побитово
равна T25, `passed=false`.

Открытые controls дают ещё одно предупреждение против hindsight: месячный
w60_b050 имеет AUC 0,57926 и Brier 0,11758 против 0,56259/0,11825 у T25, но
он провалил Q3 screen и не может быть повышен после просмотра. Аудит подтвердил
source hashes, maturity, disjoint Q3/Q4, постоянный period-delta и отсутствие
выбора по 2025–2026. Результаты:
`results/research/temperature/t29_coarse_intercept/`.

## T30: pre-SVO rank и post-SVO map дают режимный разворот

T30 перестал адаптировать T25 и построил независимую хронологию. Три
логистических rank-модели обучены только до 2022: на всей истории, 2018–2021 и
2020–2021. Их положительно-монотонные Platt-карты обучены на фиксированном
post-SVO окне апрель–ноябрь 2022. Из 15 raw/blend-кандидатов выбирал только
2023; 2024 был отдельной validation, 2025–2026 — открытая диагностика.

Screen 2023 не выбрал ничего. Full-history family улучшала Brier
0,22651→0,20723 и log-loss 0,64313→0,62931, но AUC падал 0,58840→0,49044.
Recent2y имела AUC 0,63125, однако log-loss оставался хуже baseline. Поэтому
`screen_selected=identity_early`, `validation_passed=false`, итог не меняет
router.

Режимный разворот особенно важен: уже на 2024 невыбранная full-history модель
дала AUC 0,62445 и Brier 0,15446 против 0,50086/0,16496 baseline. На открытых
2025–2026 `all_platt_b100` получила AUC 0,74954, AP 0,37744, Brier 0,11827 и
log-loss 0,38896 против 0,56054/0,21742/0,12932/0,43650 у T25. Половинная
карта дала Brier 0,11881 и ECE 0,03357. Повышать их нельзя: в заранее выбранном
2023 screen та же rank-family была хуже случайного порядка.

Практический результат — конкретный prospective challenger
`all_platt_b050`, замороженный без права менять параметры, рядом с T25. Все
карты монотонны, source/maturity/anchor/selection audits и future-target
corruption прошли. Результаты:
`results/research/temperature/t30_presvo_rank_postsvo_map/`.

## T31: mature-only Online Hedge причинно видит поздний режим

T31 проверил ансамбль трёх frozen T30-экспертов без календарного switch. Перед
каждой публикацией вес менялся только по предыдущим датам, чей h20-target уже
полностью созрел до `query - 2 дня`. Проверены 16 заранее зарегистрированных
пар `eta × fixed-share gamma`; 2023 выбирал одну строку, 2024 мог лишь
подтвердить её, 2025–2026 ничего не выбирал.

На screen-2023 лучший компромисс `eta=0,25, gamma=0,10` улучшил Brier
0,22651→0,19114, log-loss 0,64313→0,56730, ECE 0,18576→0,03006 и AUC
0,58840→0,60411. Однако заранее требовалось AUC-delta не меньше +0,02, а
получено +0,01571. Feasible-кандидатов не было; selector сохранил identity и
formal `passed=false`.

Поздняя диагностика подтверждает сам механизм: `eta=2, gamma=0` на открытых
2025–2026 имеет AUC 0,74951, AP 0,37744, Brier 0,11881, log-loss 0,38862 и
ECE 0,03357 против 0,56054/0,21742/0,12932/0,43650/0,04677 у T25. Но
выбирать эту строку после просмотра запрещено. Она заморожена только как
prospective shadow. Аудит восстановил все смеси, maturity/publication cutoffs,
selection/fallback и future-target invariance. Результаты:
`results/research/temperature/t31_mature_fixed_share/`.

## T32: OOS warm-up не решает режимный разворот

T32 проверил, был ли провал T31 на 2023 следствием cold start. Rank снова
обучен до 2022, отдельная probability-map - на 410 созревших строках каждой
семьи апреля-июля 2022. После двухмесячного gap T4-compatible HGB-anchor
обучен один раз на 01.10.2022 по 640 mature rows, а 325 строк Q4 стали OOS
warm-up. К первой screen-дате Hedge уже потребил 42 mature feedback batches.

Warm-up ухудшил 2023: максимум AUC среди 16 строк 0,55223 против 0,58840 у
identity; cold-start T31 на том же screen доходил до 0,60411. Быстрый no-share
Hedge уже к концу 2022 отдал 99,61% веса recent2y и почти не оставил себе
возможности сменить режим. Feasible-строк нет, formal result равен identity.

На 2024 все строки снова проходят gates; невыбранный e2/g0.01 имеет AUC
0,63158 и Brier 0,15332. На open лучший warm control даёт AUC 0,67532 и Brier
0,12363 против 0,56054/0,12932 у T25, но слабее cold-start T31. Это доказывает,
что соседний успешный период может закрепить неправильного эксперта. Слепой
recent-data weighting исключён; следующий адаптер должен стабилизировать
междневной rank или использовать observable state. Результаты:
`results/research/temperature/t32_oos_warm_hedge/`.

## T33: квартальный hold почти проходит frozen screen

T33 оценивал simplex-веса только на границе квартала по полностью созревшим
предыдущим h20 outcomes и держал их неизменными до следующего квартала. Grid:
окна 60/125/250/expanding и ridge 0/0,01/0,10/1,00. Pre-2023 warm-up после T32
не использовался.

Лучший screen rank `w125/ridge1` дал AUC 0,60680 против 0,58840, Brier
0,19007 против 0,22651, log-loss 0,56511 против 0,64313 и ECE 0,02435 против
0,18576. AUC delta +0,01840 недобрала 0,00160 до frozen gate +0,020. Поэтому
feasible-строк ноль и final равен identity; почти пройденный порог не изменён.

На 2024 эта же строка проходит все gates с AUC 0,61988 и Brier 0,15645.
Невыбранный `w250/ridge0` имеет validation AUC/Brier 0,66370/0,15305 и open
0,72503/0,11986 против 0,56054/0,12932 у T25. Квартальный hold подтверждает
вред daily rank-noise, но открытый период не выбирает окно/ridge. Freeze
`w125/ridge1` только как prospective control. Результаты:
`results/research/temperature/t33_quarterly_stacking/`.

## T34: availability-safe cold start проходит screen и validation

T34 проверил один смысловой gate без новой сетки. Если к границе квартала
доступно меньше 20 полностью созревших h20 feedback-batches, выход равен
`identity_early`; иначе используется прежний T33 `qstack_w125_r100`. Порог 20
унаследован из минимального fit-правила T33 и не подбирался.

На 2023 screen AUC вырос 0,58840→0,68834, Brier снизился
0,22651→0,18325, log-loss 0,64313→0,54824, ECE 0,18576→0,06210. На disjoint
2024 validation кандидат также прошёл все frozen gates: AUC 0,61988, Brier
0,15645, log-loss 0,48794, ECE 0,01896. Улучшение AUC и proper scores против
identity наблюдается во всех пяти валютах обоих периодов.

На открытом 2025–2026 T34 даёт AUC 0,63975 и Brier 0,12548 против
0,56054/0,12932 у T25. Paired date-block интервалы доказывают преимущество над
identity, но пересекают ноль против T25. `historical_protocol_passed=true`,
`production_promoted=false`: идея сформирована после T33, а все поздние годы
уже открыты. Это frozen retrospective shadow, не fresh winner. Результаты:
`results/research/temperature/t34_cold_start_identity_gate/`.

## T35: фиксированные часы оказались неправильной границей

T35 собрал единый h20 shadow из T34 в 00:15/06:00/09:15/10:15, неизменного
T19 внутри дня и T22 после assumed receipt. Pooled результат улучшился в обоих
сценариях: calendar Brier 0,11903→0,11562 и AUC 0,5659→0,6426; no-receipt
Brier 0,11754→0,11617 и AUC 0,6005→0,6336.

Однако 10:15 провалил gate в обоих сценариях: upper Brier CI пересёк ноль,
AUC CI пересёк ноль, ECE delta составила +0,01054. Причина - часть дат уже
имела market prefix, а T35 всё равно подменял его T34 по часам. Прошли 38/40
состояний, формально `retrospective_route_passed=false`. Результаты:
`results/research/temperature/t35_unified_h20_phase_router/`.

## T36: source-driven route резко улучшил rank, но ухудшил локальную ECE

T36 маршрутизировал по фактическому `snapshot_source_kind`: T34 только на
`cbr_history`, T22 только на receipt-dependent replay, остальные строки
побитово равны identity. Pooled AUC достиг 0,7018/0,7024, Brier снизился до
0,11345/0,11347; paired Brier и AUC интервалы прошли в обоих сценариях. Оба
компонента по отдельности также прошли point gates.

Но raw T34 probability оказалась переуверенной в смешанных дневных состояниях:
12 из 40 строк нарушили заранее заданный ECE delta ≤ +0,01. Порог не ослаблен,
`retrospective_route_passed=false`. Это отделило две гипотезы: source routing
верен, сила probability correction слишком велика. Результаты:
`results/research/temperature/t36_source_driven_h20_router/`.

## T37: 50% history shrink проходит 40/40 state gates

T37 зарегистрировал одну формулу без alpha-grid: на `cbr_history` смешать
identity и T34 поровну в log-odds; T22 после receipt и T19 на остальных
источниках не менять. В calendar replay pooled Brier/log-loss/AUC/ECE стали
0,11511/0,38586/0,6568/0,03355 против
0,11903/0,39993/0,5659/0,03492. В no-receipt -
0,11525/0,38580/0,6541/0,03068 против
0,11754/0,39395/0,6005/0,03196.

Оба pooled bootstrap gate, оба source-component gate и все 40 local
non-inferiority gates прошли. History-component улучшил Brier на -0,00442,
log-loss на -0,01577, AUC на +0,13286 и ECE на -0,00722. Formal
`retrospective_route_passed=true`, но `production_promoted=false`: T37 был
придуман после просмотра T36, а 2025-2026 уже открыт. Это frozen prospective
shadow, а не свежий независимый победитель. Результаты:
`results/research/temperature/t37_source_driven_h20_shrink50/`.

## T38: pooled h20 устойчив по годам, но локальная калибровка ещё не доказана

T38 не подбирал новую модель и не менял прогнозы T37. Он проверил тот же frozen
output отдельно по валютам, годам и `валюта x год`, сохраняя целые даты в
парном circular moving-block bootstrap с блоками 20/50 дат.

Годовой результат полностью устойчив: **80/80** clock-строк и **4/4** pooled
year-группы проходят. Brier улучшается во всех **680/680** локальных point-срезах,
AUC - в **674/680**; оставшиеся шесть не нарушают допуск non-inferiority. Но
строгий общий итог только **619/680** clock-строк и **25/34** pooled local
групп. Основная причина - ECE у TJS до verified receipt; часть малых
`currency x year` групп также имеет широкие Brier/AUC интервалы, пересекающие
ноль. Поэтому `local_stability_passed=false`, `production_promoted=false`.

Точная формулировка статуса T37 теперь: **сильный pooled/year-stable shadow,
но currency-local calibration ещё не production-proven**. Нельзя задним числом
дать TJS отдельный вес по открытому 2025-2026. Следующий допустимый шаг -
валютно-фазовая shrinkage-карта только на disjoint pre-2025 OOS predictions или
genuinely prospective shadow. Результаты:
`results/research/temperature/t38_h20_local_stability/`.

## T39: pre-2025 currency alpha улучшает aggregate, но портит локальную ECE

T39 реализовал следующий допустимый шаг без open-period selection. Для каждой
валюты один из alpha `0/.25/.5/.75/1` выбирался по Brier на 226 mature OOS
строках 2023 и отдельно проверялся на 228 mature OOS строках 2024. Все валюты
на screen выбрали alpha=1. Validation разрешила его только AMD и UZS;
KGS/KZT/TJS вернулись к T37 alpha=0,5 из-за ECE/AUC gates.

Frozen карта прошла 40/40 объединённых состояний против T37 и даже улучшила
pooled Brier примерно на -0,00072/-0,00078 и AUC на +0,020/+0,022. Но строгая
локальная картина ухудшилась: **619→594/680** clock-строк, **25→22/34** pooled
групп и **4→3/4** year-групп. Новые failures сосредоточены в AMD-2026, UZS и
объединённом 2026. Усиление улучшает rank, но делает probability локально
переуверенной.

`retrospective_repair_passed=false`, `production_promoted=false`; T37 остаётся
без изменения. Две pre-2025 OOS выборки не поддерживают более подробную
currency-specific константу. Следующий шаг - prospective shadow или новый
действительно независимый source-state replay. Результаты:
`results/research/temperature/t39_pre2025_currency_shrink/`.

## T40: длинный annual rolling-origin replay отвергает стационарный CBR-эксперт

T40 был зарегистрирован до fit и до просмотра новых метрик. Для каждого года
2019–2026 один `StandardScaler + L2 LogisticRegression(C=0.1)` обучался на
истории до начала предыдущего года с mature h20 labels и embargo два дня.
Frozen raw-score калибровался на предыдущем году, затем 50/50 смешивался в
log-odds с причинной глобальной базовой частотой. Использовались фиксированные
41 T24 CBR-признак без market/receipt данных и без выбора гиперпараметров.

Исторический screen 2019–2022 провален: на 4 920 строках Brier delta к prior
**+0,00654**, log-loss **+0,02260**, ECE **+0,02503**, AUC delta +0,00439;
локально прошли **0/9** year/currency групп. В 2022 AUC delta упала до
-0,10322, но провал не ограничен одной датой режима: каждый год 2019–2022
нарушил хотя бы один допуск, а Brier ухудшился во всех пяти валютах.

Отдельная validation 2023–2024 дала Brier delta **-0,00239** и AUC delta
**+0,09832**, но только **5/7** локальных групп прошли, а 20/50-date paired
bootstrap пересёк ноль по Brier и AUC. По frozen протоколу
`historical_gate_passed=false`, поэтому 2025–2026 model metrics не открывались:
`open_evaluated=false`, `production_promoted=false`.

Аудит пересобрал восемь annual fits, causal prior, метрики, bootstrap и gate,
проверил maturity/embargo и future-prefix corruption. T37 остаётся неизменным.
Вывод: длинная CBR-история без наблюдаемого state router не решает h20
калибровку; хороший 2023–2024 режим нельзя экстраполировать назад или вперёд.
Результаты: `results/research/temperature/t40_long_rolling_history/`.

## T41: mature quarterly shrink смягчает поздний период, но не проходит историю

T41 заранее зафиксировал один параметр-free механизм поверх T40. В начале
каждого квартала по последним 125 полностью созревшим publication-date batches
закрытой Brier-формулой вычислялся общий `alpha` между causal prior и T40.
Веса не искались сеткой, использовали одинаково для пяти валют и замораживали
на квартал; при менее чем 20 feedback batches включался безопасный prior.

Screen 2019–2022 на 4 920 строках провален: Brier delta **+0,00425**,
log-loss **+0,01279**, ECE **+0,01671**, AUC delta +0,01264 и только **2/9**
local non-inferiority. Brier bootstrap CI при блоках 20 и 50 дат целиком выше
нуля. Validation 2023–2024 улучшила point Brier на −0,00113, log-loss на
−0,00425, ECE на −0,01166 и AUC на +0,07815; прошли 6/7 local групп, но
Brier/AUC CI всё ещё пересекают ноль.

По frozen gate 2025–2026 model metrics не открывались. Аудит полностью
пересобрал состояния, predictions, metrics, bootstrap и решение, подтвердил
maturity/embargo и future-prefix corruption. `historical_gate_passed=false`,
`open_evaluated=false`, `production_promoted=false`; T37 остаётся frozen
shadow. Результаты:
`results/research/temperature/t41_mature_quarterly_shrink/`.

## T42: observable OOD shrink распознаёт 2022, но не спасает history-эксперт

T42 заменил запаздывающую оценку ошибок T41 полностью label-free состоянием.
Для каждого annual OOS года по точной training mask T40 обучается только
`StandardScaler` на тех же 41 CBR-признаках. Средняя квадратичная
стандартизованная удалённость query-строки задаёт единственный вес
`alpha=min(1, 1/energy)` между causal prior и вероятностью T40. Сетки,
calendar/SVO-флага, исходов query-периода и выбора на 2025–2026 нет.

OOD-сигнал содержателен: средняя energy 2022 равна **3,219** против примерно
0,55–0,90 в 2019–2021, а средний вес T40 падает до **0,619**. Screen-вред
T40 по Brier уменьшается с +0,00654 до **+0,00262**. Но это всё ещё
статистически значимое ухудшение к prior: log-loss +0,01205, ECE +0,01514,
только **0/9** local non-inferiority. Validation 2023–2024 даёт Brier
−0,00218 и AUC +0,10370, но pooled Brier CI и 50-day AUC CI пересекают ноль,
а проходят только 5/7 local groups.

Frozen historical gate не пройден, поэтому target/model metrics 2025–2026 не
открывались. Аудит пересобрал annual scaler states, predictions, метрики,
bootstrap и gate, подтвердил maturity/embargo, формулу alpha и
future-prefix corruption. `production_promoted=false`; T37 остаётся frozen
shadow. Вывод: covariate novelty помогает ограничить вред, но не сообщает,
какой эксперт прав после изменения зависимости feature→target. Результаты:
`results/research/temperature/t42_ood_history_shrink/`.

## T43: nonlinear annual expert улучшает rank, но не proper score

T43 заранее заморозил ровно один `HistGradientBoostingClassifier` на точных
41 T40-признаках и annual train/calibration/query masks. Параметры capacity
унаследованы от AP1; сетки и ручного regime/SVO-флага нет. Previous-year Platt,
causal prior и 50/50 log-odds blend оставлены как в T40, чтобы проверить только
эффект нелинейных feature interactions.

Screen 2019–2022 дал лучший ранний rank среди T40–T43: AUC delta **+0,03895**,
а Brier-вред к prior уменьшился до **+0,00203**. Но Brier CI целиком выше
нуля, log-loss delta +0,00935, ECE delta +0,01672 и только **0/9** local
non-inferiority. В 2020 AUC выросла на +0,22247 при ухудшении Brier/ECE; в
2022 AUC упала на −0,04390. Validation 2023–2024 дала Brier −0,00143 и AUC
+0,09820, но CI пересекают ноль, 2024 и KZT не проходят local gate.

Platt slope, причинно доступный из предыдущего года, оказался отрицательным
перед 2019, 2022 и 2024. Это объясняет нестабильное направление raw rank и
задаёт следующий preregistered тест: независимый mature calibration gate,
а не ещё большую capacity. Frozen historical gate не пройден, model metrics
2025–2026 не открывались. Аудит воспроизвёл восемь annual fits, фиксированные
160 итераций, probabilities, bootstrap/gates, maturity/embargo и
future-prefix corruption. `production_promoted=false`; T37 неизменён.
Результаты: `results/research/temperature/t43_nonlinear_history/`.

## T44: disjoint mature quality gate безопасен, но не доказывает gain

T44 заранее отделил обучение probability-map от проверки её компетентности.
Для query year `Y` raw T43-модель по-прежнему обучается до `Y-1`; Platt fit
использует только mature январь-июнь `Y-1`, а решение допустить эксперта
принимается на mature июле-декабре. Требуются лучшие pooled Brier/log-loss/AUC,
допустимая ECE, положительный Platt slope и non-inferiority всех пяти валют.
При отказе прогноз года побитово равен causal prior.

Gate закрыл эксперт во всех screen-годах 2019-2022, поэтому 4 920 строк дали
точные нулевые deltas и 9/9 local pass. На validation он открыл только 2023:
pooled 2023-2024 Brier delta **-0,00098**, log-loss **-0,00357**, ECE
**-0,00749**, AUC delta **+0,12835**, local pass **7/7**. Но Brier и AUC
paired CI пересекают ноль. Кроме того, frozen screen требует доказанного
улучшения, а не просто безопасного равенства baseline, поэтому formal gate
провален и 2025-2026 model metrics не открыты.

Вывод: delayed competence gate успешно предотвращает вред T43, но пока не
создаёт доказанный predictive gain. Порог не ослаблен; T37, runtime, push и
expected future bps не изменены. T44 остаётся prospective control.
Результаты: `results/research/temperature/t44_calibration_quality_gate/`.
