import json
import logging
from pathlib import Path

import pytest

from llm_basics import (
    ExperimentTracker,
    InMemoryMetricSink,
    JsonlFileMetricSink,
    LoggerMetricSink,
    SummaryWriterMetricSink,
    WandbLikeMetricSink,
)


class FakeClock:
    def __init__(self, *values: float):
        self._values = list(values)

    def __call__(self) -> float:
        if not self._values:
            raise RuntimeError("FakeClock exhausted.")
        return self._values.pop(0)


def test_experiment_tracker_records_dual_axes():
    sink = InMemoryMetricSink()
    tracker = ExperimentTracker(sinks=[sink], clock=FakeClock(100.0, 102.5))

    tracker.log_metrics({"train/loss": 1.25, "lr": 3e-4}, gradient_step=7)

    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.gradient_step == 7
    assert record.wall_time_seconds == pytest.approx(2.5)
    assert record.metrics == {"train/loss": 1.25, "lr": 3e-4}


def test_experiment_tracker_rejects_invalid_metric_values():
    tracker = ExperimentTracker(sinks=[], clock=FakeClock(0.0, 1.0))

    with pytest.raises(TypeError):
        tracker.log_metrics({"train/loss": "bad"}, gradient_step=1)


def test_jsonl_sink_writes_records(tmp_path: Path):
    sink = JsonlFileMetricSink(tmp_path / "logs" / "metrics.jsonl")
    tracker = ExperimentTracker(sinks=[sink], clock=FakeClock(5.0, 8.0))

    tracker.log_metrics({"val/loss": 2.0}, gradient_step=3)

    lines = (tmp_path / "logs" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["gradient_step"] == 3
    assert payload["wall_time_seconds"] == pytest.approx(3.0)
    assert payload["metrics"] == {"val/loss": 2.0}


def test_wandb_like_sink_logs_step_and_axes():
    calls = []

    class FakeRun:
        def log(self, data, step=None):
            calls.append((data, step))

    tracker = ExperimentTracker(sinks=[WandbLikeMetricSink(FakeRun())], clock=FakeClock(1.0, 4.5))
    tracker.log_metrics({"train/ppl": 12.0}, gradient_step=9)

    assert calls == [
        ({"train/ppl": 12.0, "gradient_step": 9.0, "wall_time_seconds": 3.5}, 9),
    ]


def test_summary_writer_sink_logs_step_and_time_axes():
    calls = []

    class FakeWriter:
        def add_scalar(self, tag, scalar_value, global_step):
            calls.append((tag, scalar_value, global_step))

    tracker = ExperimentTracker(sinks=[SummaryWriterMetricSink(FakeWriter())], clock=FakeClock(10.0, 12.2))
    tracker.log_metrics({"train/loss": 0.75}, gradient_step=5)

    assert calls == [
        ("train/loss", 0.75, 5),
        ("time/train/loss", 0.75, 2),
    ]


def test_logger_sink_emits_both_axes(caplog: pytest.LogCaptureFixture):
    logger = logging.getLogger("test.experiment_tracking")
    sink = LoggerMetricSink(logger=logger)
    tracker = ExperimentTracker(sinks=[sink], clock=FakeClock(20.0, 23.0))

    with caplog.at_level(logging.INFO, logger="test.experiment_tracking"):
        tracker.log_metrics({"grad_norm": 1.5}, gradient_step=4)

    assert "step=4" in caplog.text
    assert "time=3.000s" in caplog.text
    assert "grad_norm=1.500000" in caplog.text
