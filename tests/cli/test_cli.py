import sys

import pytest

from kalbee.cli import main, build_parser, _sparkline


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "kalbee" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_demo_static_runs(capsys):
    code = main(["demo", "--steps", "20", "--filter", "kf"])
    assert code == 0
    out = capsys.readouterr().out
    assert "filtered RMSE" in out


def test_demo_rejects_incompatible_filter():
    with pytest.raises(SystemExit):
        main(["demo", "--filter", "ekf"])  # not in the allowed choices


def test_demo_unknown_signal(capsys):
    code = main(["demo", "--signal", "not-a-signal"])
    assert code == 1
    assert "Unknown signal" in capsys.readouterr().out


def test_bench_runs(capsys):
    code = main(["bench", "--duration", "1", "--dt", "0.2"])
    assert code == 0
    assert "Filter" in capsys.readouterr().out


def test_bench_falls_back_without_rich(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "rich", None)
    monkeypatch.setitem(sys.modules, "rich.console", None)
    monkeypatch.setitem(sys.modules, "rich.table", None)
    code = main(["bench", "--duration", "1", "--dt", "0.2"])
    assert code == 0
    # run_benchmark() itself already printed a plain-text table
    assert "Filter" in capsys.readouterr().out


def test_demo_live_runs(monkeypatch, capsys):
    monkeypatch.setattr("time.sleep", lambda _: None)
    code = main(["demo", "--live", "--steps", "5", "--fps", "1000"])
    assert code == 0
    out = capsys.readouterr().out
    assert "filtered" in out


def test_demo_live_falls_back_without_rich(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "rich", None)
    monkeypatch.setitem(sys.modules, "rich.console", None)
    monkeypatch.setitem(sys.modules, "rich.live", None)
    monkeypatch.setitem(sys.modules, "rich.panel", None)
    monkeypatch.setitem(sys.modules, "rich.text", None)
    code = main(["demo", "--live", "--steps", "5"])
    assert code == 0
    out = capsys.readouterr().out
    assert "pip install kalbee[cli]" in out
    assert "filtered RMSE" in out  # fell back to the static summary


def test_new_scaffolds_file(tmp_path, capsys):
    target = tmp_path / "my_script.py"
    code = main(["new", str(target)])
    assert code == 0
    assert target.exists()
    assert "KalmanFilter" in target.read_text()


def test_new_refuses_overwrite_without_force(tmp_path):
    target = tmp_path / "existing.py"
    target.write_text("# already here\n")
    code = main(["new", str(target)])
    assert code == 1
    assert target.read_text() == "# already here\n"


def test_new_force_overwrites(tmp_path):
    target = tmp_path / "existing.py"
    target.write_text("# already here\n")
    code = main(["new", str(target), "--force"])
    assert code == 0
    assert "KalmanFilter" in target.read_text()


def test_new_all_templates_are_valid_python(tmp_path):
    import ast

    from kalbee.cli import _TEMPLATES

    for name, source in _TEMPLATES.items():
        ast.parse(source)  # raises SyntaxError on failure


def test_sparkline_basic():
    assert _sparkline([]) == ""
    assert len(_sparkline([1, 2, 3, 4])) == 4
    assert (
        _sparkline([5, 5, 5]) == _sparkline([5, 5, 5])[0] * 3
    )  # flat input -> flat line


def test_build_parser_has_expected_subcommands():
    parser = build_parser()
    sub_actions = [a for a in parser._actions if a.dest == "command"]
    assert sub_actions
    assert set(sub_actions[0].choices) == {"demo", "bench", "new"}
