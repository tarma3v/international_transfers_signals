from ml.copy_library import COPY_LIBRARY, copy_for


def test_copy_library_covers_product_states():
    keys = {(row['scenario'], row['direction']) for row in COPY_LIBRARY}
    assert ('sparse_push_and_fresh_widget', 'supports_current_moment') in keys
    assert ('fresh_widget_only', 'supports_current_moment') in keys
    assert ('fresh_widget_only', 'mixed_or_neutral') in keys
    assert ('fresh_widget_only', 'weak_support_for_current_moment') in keys
    assert ('widget_historical_context_only', 'neutral') in keys
    assert ('customer_level_alert', 'level_reached') in keys
    assert len(copy_for(
        'sparse_push_and_fresh_widget', 'supports_current_moment')) >= 2


def test_customer_copy_contains_no_future_promise_or_instruction():
    forbidden = (
        'подождите', 'лучше подождать', 'курс будет', 'курс станет',
        'гарантируем', 'успейте', 'переводите сейчас', 'советуем',
    )
    for row in COPY_LIBRARY:
        text = (row['title'] + ' ' + row['body']).lower()
        assert not any(fragment in text for fragment in forbidden), row
