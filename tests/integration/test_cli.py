"""The ``parapick`` command line, end to end."""

from __future__ import annotations

import json

import pytest

import parapick as pp
from parapick.cli.commands import COMMANDS
from parapick.cli.main import build_parser, main


@pytest.fixture
def simulated(tmp_path):
    out = tmp_path / "sim"
    code = main(
        ["simulate", "-o", str(out), "--n-tilt-series", "2", "--n-particles", "15",
         "--tilt-step", "5", "--image-size", "320", "320", "--false-positives", "5", "-q"]
    )  # fmt: skip
    assert code == 0
    return out


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    text = capsys.readouterr().out
    for command in COMMANDS:
        assert command.name in text


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "parapick" in capsys.readouterr().out


def test_formats(capsys):
    assert main(["formats"]) == 0
    assert "relion5" in capsys.readouterr().out


def test_full_pipeline(tmp_path, simulated, capsys):
    tomograms = str(simulated / "tomograms.star")
    out = tmp_path / "out"
    assert main(["localize", "--tomograms", tomograms, "--picks", str(simulated / "picks.star"),
                 "-o", str(out), "--device", "cpu", "-q"]) == 0  # fmt: skip
    assert (out / "particles.star").exists() and (out / "optimisation_set.star").exists()

    report = tmp_path / "evaluation.json"
    assert main(["evaluate", str(out / "particles.star"), str(simulated / "ground_truth.star"),
                 "--cutoff", "50", "--min-score", "0.3", "--json", str(report)]) == 0  # fmt: skip
    assert "all" in capsys.readouterr().out
    assert json.loads(report.read_text())["overall"]["f1"] > 0.85

    filtered = tmp_path / "filtered.star"
    assert main(["filter", str(out / "particles.star"), "--top-n", "5", "-o", str(filtered),
                 "--tomograms", tomograms]) == 0  # fmt: skip
    assert len(pp.read_particles(filtered)) == 10

    topaz = tmp_path / "gt.txt"
    assert main(["project", str(simulated / "ground_truth.star"), "--tomograms", tomograms,
                 "-o", str(topaz), "--output-format", "topaz"]) == 0  # fmt: skip
    converted = tmp_path / "gt_picks.star"
    assert main(["convert-picks", str(topaz), "-o", str(converted),
                 "--tomograms", tomograms]) == 0  # fmt: skip
    picks = pp.read_picks(converted)
    assert set(picks.tilt_series) == {"TS_01", "TS_02"}

    coords = tmp_path / "coords" / "autopick.star"
    assert main(["convert-picks", str(converted), "-o", str(coords),
                 "--output-format", "relion-coords"]) == 0  # fmt: skip
    out2 = tmp_path / "out2"
    args = ["localize", "--tomograms", tomograms, "--picks", str(coords), "-o", str(out2),
            "--no-background", "--annealing", "0", "--merge-distance", "0", "--init", "uniform",
            "--min-score", "0.5", "-q"]  # fmt: skip
    assert main(args) == 0
    assert len(pp.read_particles(out2 / "particles.star")) > 0


def test_errors_return_non_zero(tmp_path, simulated, caplog):
    code = main(["localize", "--tomograms", str(tmp_path / "missing.star"),
                 "--picks", str(simulated / "picks.star"), "-o", str(tmp_path / "x"),
                 "--tomograms-format", "relion5"])  # fmt: skip
    assert code == 1
    with pytest.raises(SystemExit):
        main(["localize", "--picks", "x"])  # missing required arguments
    code = main(["localize", "--tomograms", str(simulated / "tomograms.star"),
                 "--picks", str(simulated / "picks.star"), "-o", str(tmp_path / "x"),
                 "--alpha", "-1"])  # fmt: skip
    assert code == 1
    assert "alpha" in caplog.text
