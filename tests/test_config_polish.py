"""Configuration helpers preserve scalar and nested override behavior."""

from src.common.config import _coerce, load_config


def test_coerce_known_scalar_types() -> None:
    assert _coerce("TRUE") is True
    assert _coerce("false") is False
    assert _coerce("42") == 42
    assert _coerce("0.25") == 0.25
    assert _coerce("'sample name'") == "sample name"


def test_load_config_keeps_unspecified_release_defaults(tmp_path) -> None:
    source = tmp_path / "settings.yaml"
    source.write_text(
        "mode: demo\nn_rows: 100\nrelease:\n  max_ece: 0.05\n",
        encoding="utf-8",
    )
    config = load_config(source, overrides={"n_rows": 120})
    assert config["mode"] == "demo"
    assert config["n_rows"] == 120
    assert config["release"]["max_ece"] == 0.05
    assert config["release"]["min_long_tail_exposure"] == load_config()["release"]["min_long_tail_exposure"]
