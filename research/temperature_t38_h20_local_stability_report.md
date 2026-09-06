# T38: локальная устойчивость T37 пока не доказана

## Зачем понадобился отдельный аудит

T37 прошёл 40/40 состояний `scenario x clock`, но эти строки объединяли пять
валют и два года. T38 не меняет ни одного прогноза: он повторно оценивает тот
же frozen T37 отдельно по валютам, годам и `валюта x год`, а затем считает
парный circular moving-block bootstrap по целым датам с блоками 20 и 50 дней.

Аудит зарегистрирован после просмотра уже существовавших point-срезов, но до
расчёта локальных bootstrap-интервалов. Поэтому это диагностический evidence
packet, а не новый selector.

## Результат

| Срез | Clock rows: pass / all | Pooled groups: pass / all | Главная причина отказа |
|---|---:|---:|---|
| Валюта | 169 / 200 | 9 / 10 | локальная ECE, главным образом TJS |
| Год | **80 / 80** | **4 / 4** | нет |
| Валюта x год | 370 / 400 | 12 / 20 | ECE и широкие CI малых групп |

Итого проходят 619/680 clock-local строк и 25/34 pooled групп. Формальный
`local_stability_passed=false`, `production_promoted=false`.

При этом картина не похожа на исчезновение сигнала:

- Brier улучшается во всех 680 локальных point-срезах;
- AUC улучшается в 674/680, а оставшиеся шесть не нарушают допуск -0,005;
- все четыре годовых pooled группы проходят Brier/AUC block-CI;
- все пять валют проходят в calendar receipt replay;
- в no-receipt единственный pooled currency failure - TJS: Brier -0,00206,
  AUC +0,04975 и оба CI проходят, но ECE delta +0,01014 на 0,00014 выше
  frozen лимита.

## Где именно остаётся риск

До receipt и без same-day receipt TJS имеет 15 clock-срезов с ECE delta от
примерно +0,0117 до +0,0188. Это связано с history-веткой на датах без
доступного market prefix. В calendar replay вечерние T22-состояния дают
отдельные ECE всплески у AMD/KGS/KZT/TJS, особенно на маленьком 2026 support.

Из восьми failed `валюта x год` pooled строк многие имеют хорошие point
дельты, но Brier или AUC CI пересекает ноль: AMD-2026, KGS-2026, KZT-2025,
TJS-2026, UZS-2025/2026 в no-receipt и KZT-2025/TJS-2026 в calendar replay.
Это ограничение доказательной мощности, а не основание подобрать валютный вес
на уже открытом периоде.

## Решение и следующий допустимый шаг

T37 сохраняется как сильный frozen shadow, но формулировка уточняется:
**pooled и year-stable, currency-local calibration not yet production-proven**.
Нельзя выбирать меньший вес только для TJS или отдельные after-receipt веса по
2025-2026.

Следующий честный эксперимент должен построить валютно-фазовую shrinkage-карту
только на ранее созревших pre-2025 OOS predictions, зафиксировать её до
открытого evaluation и оставить identity fallback при недостатке support.
Если такого disjoint материала недостаточно, вопрос переносится в prospective
shadow с фактическими receipt timestamps.

Полные point metrics, 204 bootstrap-интервала, gates и независимый audit:
`results/research/temperature/t38_h20_local_stability/`.
