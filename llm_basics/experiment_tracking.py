from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TextIO


MetricValue = int | float


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """A single logged metric bundle with dual x-axes."""

    gradient_step: int
    wall_time_seconds: float
    metrics: dict[str, float]


class MetricSink(Protocol):
    """Protocol for pluggable experiment-tracking backends."""

    def log(self, record: MetricRecord) -> None:
        """Consume one metric record."""


@dataclass(slots=True)
class InMemoryMetricSink:
    """Stores metric records for tests or downstream inspection."""

    records: list[MetricRecord] = field(default_factory=list)

    def log(self, record: MetricRecord) -> None:
        self.records.append(record)


@dataclass(slots=True)
class LoggerMetricSink:
    """Logs metrics through the stdlib logging module."""

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    level: int = logging.INFO

    def log(self, record: MetricRecord) -> None:
        metric_items = " ".join(f"{name}={value:.6f}" for name, value in sorted(record.metrics.items()))
        self.logger.log(
            self.level,
            "step=%d time=%.3fs %s",
            record.gradient_step,
            record.wall_time_seconds,
            metric_items,
        )


@dataclass(slots=True)
class JsonlFileMetricSink:
    """Appends records to a local JSONL file without external dependencies."""

    path: str | Path

    def log(self, record: MetricRecord) -> None:
        import json

        payload = {
            "gradient_step": record.gradient_step,
            "wall_time_seconds": record.wall_time_seconds,
            "metrics": record.metrics,
        }
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


class WandbLikeRun(Protocol):
    def log(self, data: dict[str, float], step: int | None = None) -> None: ...


@dataclass(slots=True)
class WandbLikeMetricSink:
    """Adapts to objects that expose the common wandb.Run.log API."""

    run: WandbLikeRun

    def log(self, record: MetricRecord) -> None:
        payload = dict(record.metrics)
        payload["gradient_step"] = float(record.gradient_step)
        payload["wall_time_seconds"] = record.wall_time_seconds
        self.run.log(payload, step=record.gradient_step)


class SummaryWriterLike(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...


@dataclass(slots=True)
class SummaryWriterMetricSink:
    """Adapts to TensorBoard SummaryWriter-style add_scalar methods."""

    writer: SummaryWriterLike
    time_axis_prefix: str = "time"

    def log(self, record: MetricRecord) -> None:
        for metric_name, metric_value in record.metrics.items():
            self.writer.add_scalar(metric_name, metric_value, record.gradient_step)
            self.writer.add_scalar(
                f"{self.time_axis_prefix}/{metric_name}",
                metric_value,
                int(round(record.wall_time_seconds)),
            )


@dataclass(slots=True)
class CompositeMetricSink:
    """Fan-out sink that forwards each record to multiple backends."""

    sinks: list[MetricSink]

    def log(self, record: MetricRecord) -> None:
        for sink in self.sinks:
            sink.log(record)


@dataclass(slots=True)
class ExperimentTracker:
    """Tracks metrics against both gradient-step and wall-clock axes."""

    sinks: list[MetricSink] = field(default_factory=list)
    clock: callable = time.perf_counter
    start_time_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        self.start_time_seconds = float(self.clock())

    def log_metrics(self, metrics: dict[str, MetricValue], gradient_step: int) -> None:
        """
        Log a metric bundle with both gradient_step and elapsed wall-clock time.
        """
        if gradient_step < 0:
            raise ValueError("gradient_step must be non-negative.")

        normalized_metrics: dict[str, float] = {}
        for name, value in metrics.items():
            if not isinstance(value, int | float):
                raise TypeError(f"Metric {name!r} must be numeric, got {type(value)!r}.")
            normalized_metrics[name] = float(value)

        record = MetricRecord(
            gradient_step=gradient_step,
            wall_time_seconds=float(self.clock()) - self.start_time_seconds,
            metrics=normalized_metrics,
        )
        for sink in self.sinks:
            sink.log(record)

    def add_sink(self, sink: MetricSink) -> None:
        self.sinks.append(sink)


def build_default_experiment_tracker(stream: TextIO | None = None) -> ExperimentTracker:
    """Create a local-first tracker suitable for training loops."""
    logger = logging.getLogger("llm_basics.training")
    if stream is not None and not logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)
    return ExperimentTracker(sinks=[LoggerMetricSink(logger=logger)])


__all__ = [
    "CompositeMetricSink",
    "ExperimentTracker",
    "InMemoryMetricSink",
    "JsonlFileMetricSink",
    "LoggerMetricSink",
    "MetricRecord",
    "MetricSink",
    "SummaryWriterMetricSink",
    "WandbLikeMetricSink",
    "build_default_experiment_tracker",
]
