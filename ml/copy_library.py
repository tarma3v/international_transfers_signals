"""Past/present-only customer copy for case scenarios.

The model may forecast internally, but customer text must not promise a future
rate or instruct the customer to wait. Templates are keyed by machine-readable
scenario and direction returned by `case_output_as_of`.
"""
from __future__ import annotations


COPY_LIBRARY = (
    {
        "scenario": "sparse_push_and_fresh_widget",
        "direction": "supports_current_moment",
        "variant": "core_history",
        "title": "Обновилась оценка курса",
        "body": (
            "По историческим данным похожие условия чаще совпадали с удачным "
            "моментом. Откройте перевод, чтобы увидеть текущую котировку и условия."
        ),
    },
    {
        "scenario": "sparse_push_and_fresh_widget",
        "direction": "supports_current_moment",
        "variant": "agreement",
        "title": "Несколько индикаторов совпали",
        "body": (
            "Текущие рыночные условия похожи на исторические периоды с более "
            "частыми удачными моментами. В приложении доступна актуальная котировка."
        ),
    },
    {
        "scenario": "fresh_widget_only",
        "direction": "supports_current_moment",
        "variant": "widget_positive",
        "title": "Историческая оценка: выше обычной",
        "body": (
            "Похожие исторические условия чаще совпадали с удачным моментом. "
            "Оценка основана на доступных данных и не фиксирует курс операции."
        ),
    },
    {
        "scenario": "fresh_widget_only",
        "direction": "mixed_or_neutral",
        "variant": "widget_neutral",
        "title": "Выраженного сигнала нет",
        "body": (
            "Исторические данные дают смешанную оценку. Текущая котировка и "
            "полные условия перевода показаны отдельно."
        ),
    },
    {
        "scenario": "fresh_widget_only",
        "direction": "weak_support_for_current_moment",
        "variant": "widget_weak",
        "title": "Историческая оценка: ниже обычной",
        "body": (
            "Похожие исторические условия редко совпадали с удачным моментом. "
            "Это справка по прошлым наблюдениям, а не рекомендация."
        ),
    },
    {
        "scenario": "widget_historical_context_only",
        "direction": "neutral",
        "variant": "stale_or_limited",
        "title": "Свежих данных для оценки недостаточно",
        "body": (
            "Показана последняя доступная историческая оценка. Время источника "
            "и актуальная котировка указаны на экране перевода."
        ),
    },
    {
        "scenario": "customer_level_alert",
        "direction": "level_reached",
        "variant": "level_fact",
        "title": "Выбранный уровень курса достигнут",
        "body": (
            "Текущая котировка достигла уровня, который вы указали. Комиссия, "
            "лимиты и срок действия котировки показаны перед подтверждением."
        ),
    },
)


def copy_for(scenario, direction):
    return [row.copy() for row in COPY_LIBRARY
            if row["scenario"] == scenario and row["direction"] == direction]
