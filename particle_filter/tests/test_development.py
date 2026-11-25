from particle_filter.validation.development import development_inventory


def test_replayed_roots_match_the_recorded_development_folds():
    roots = development_inventory()
    assert len({r.key for r in roots}) == 1200
    for condition in ("reference", "contention", "publisher"):
        for fold in range(5):
            assert sum(r.condition == condition and r.fold == fold for r in roots) == 80
        assert sum(r.condition == condition and r.panel for r in roots) == 12


def test_source_receipt_binds_nested_model_code_and_configuration_but_skips_results(
    tmp_path, monkeypatch
):
    from particle_filter.validation import development

    sub = tmp_path / "particle_filter" / "validation"
    sub.mkdir(parents=True)
    source = sub / "development.py"
    source.write_text("# model")
    monkeypatch.setattr(development, "__file__", str(source))
    first = development.study_source_digest()
    config = sub.parent / "configs" / "full.toml"
    config.parent.mkdir()
    config.write_text("half_life = 120")
    second = development.study_source_digest()
    assert first != second
    generated = sub.parent / "results" / "reproduction.py"
    generated.parent.mkdir()
    generated.write_text("# evidence")
    assert second == development.study_source_digest()
    source.write_text("# revised model")
    assert second != development.study_source_digest()


def test_aggregation_rejects_unbound_run_receipts_before_writing_a_report(tmp_path):
    import json
    import pytest
    from particle_filter.validation.development import physical_digest
    from particle_filter.validation.validation_report import summarise

    teacher, sequential = tmp_path / "teacher", tmp_path / "sequential"
    teacher.mkdir()
    sequential.mkdir()
    (teacher / "summary.json").write_text('{"roots":1200,"intervals":14400}')
    (sequential / "sequential.json").write_text('{"roots":36}')
    phase = tmp_path / "phase.json"
    phase.write_text(
        json.dumps({"development_roots": 1200, "physical_digest": physical_digest()})
    )
    with pytest.raises(ValueError, match="producing model"):
        summarise(teacher, sequential, phase, tmp_path / "report")
    assert not (tmp_path / "report").exists()
