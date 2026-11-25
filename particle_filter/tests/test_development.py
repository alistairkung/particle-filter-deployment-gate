from particle_filter.development import development_inventory


def test_replayed_roots_match_the_recorded_development_folds():
    roots = development_inventory()
    assert len({r.key for r in roots}) == 1200
    for condition in ("reference", "contention", "publisher"):
        for fold in range(5):
            assert sum(r.condition == condition and r.fold == fold for r in roots) == 80
        assert sum(r.condition == condition and r.panel for r in roots) == 12
