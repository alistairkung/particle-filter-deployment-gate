from telemetry_sufficiency.observation import Collector, SECOND


def test_completed_effects_across_first_two_boundaries():
    collector = Collector()
    end = -55 * SECOND
    collector.event("insert", 0, end - SECOND)
    collector.event("insert", 1, end - 3)
    collector.event("mark", 0, end - 1)
    collector.event("mark", 1, end)
    collector.event("insert", 2, end)
    first = collector.seal(end)
    assert first.pending_count == 1 and first.oldest_pending_age_s == 0
    assert first.publication_throughput_per_s == 0.4
    assert first.completed_delay_median_s == (SECOND + 2) / (2 * SECOND)
    collector.event("mark", 2, end + 1)
    second = collector.seal(end + 5 * SECOND)
    assert second.pending_count == 0 and second.oldest_pending_age_s == 0
    assert second.publication_throughput_per_s == 0.2
    assert second.completed_delay_median_s == 1 / SECOND
    third = collector.seal(end + 10 * SECOND)
    assert (
        third.publication_throughput_per_s == 0
        and third.completed_delay_median_s is None
    )


def test_history_changes_only_at_delivery_and_remains_immutable():
    from dataclasses import FrozenInstanceError, asdict
    import pytest
    from telemetry_sufficiency.observation import DeliveryLog, parse_history

    feed, collector = DeliveryLog(), Collector()
    reads = [collector.seal(end * SECOND) for end in (-55, -50, -45, -40)]
    feed.schedule(reads[0], available_at_ns=-42 * SECOND)
    feed.schedule(reads[1], available_at_ns=None)
    feed.schedule(reads[2], available_at_ns=-42 * SECOND)
    feed.schedule(reads[3], available_at_ns=1)
    before = feed.as_of(-42 * SECOND - 1)
    assert before == ()
    assert [r.interval_end_ns for r in feed.as_of(-42 * SECOND)] == [
        -55 * SECOND,
        -45 * SECOND,
    ]
    assert feed.as_of(0) == feed.as_of(-42 * SECOND)
    assert len(feed.as_of(1)) == 3
    assert before == feed.as_of(-42 * SECOND - 1)
    with pytest.raises(FrozenInstanceError):
        feed.as_of(0)[0].pending_count = 3
    with pytest.raises(ValueError):
        feed.schedule(reads[1], available_at_ns=5)
    with pytest.raises(ValueError):
        DeliveryLog().schedule(reads[0], available_at_ns=-56 * SECOND)
    rows = [asdict(r) for r in feed.as_of(1)]
    assert parse_history(rows, 0) == feed.as_of(0)
    for patch in (
        {"condition": "reference"},
        {"oldest_pending_age_s": float("nan")},
        {"available_at_ns": -56 * SECOND},
        {"pending_count": True},
    ):
        with pytest.raises(ValueError):
            parse_history([dict(rows[0], **patch)])
    with pytest.raises(ValueError):
        parse_history([rows[0], rows[0]])


def test_out_of_order_transit_and_lost_reading_have_same_view_before_delivery():
    from telemetry_sufficiency.observation import DeliveryLog
    import pytest

    feed, collector = DeliveryLog(), Collector()
    first, second, third = [collector.seal(end * SECOND) for end in (-55, -50, -45)]
    feed.schedule(first, available_at_ns=-44 * SECOND)
    feed.schedule(second, available_at_ns=-50 * SECOND)
    assert feed.as_of(-49 * SECOND) == (second,)
    assert [r.interval_end_ns for r in feed.as_of(-44 * SECOND)] == [
        -55 * SECOND,
        -50 * SECOND,
    ]
    with pytest.raises(ValueError, match="retroactively"):
        feed.schedule(third, available_at_ns=-45 * SECOND)
    feed.schedule(third, available_at_ns=None)
    assert feed.as_of(-49 * SECOND) == (second,)
