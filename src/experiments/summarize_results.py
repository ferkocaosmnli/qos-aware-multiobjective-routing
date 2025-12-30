from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any


@dataclass
class MetricStats:
    count: int
    mean: float
    std: float | None
    min: float
    max: float


def _safe_float(x: str) -> float:
    """Boş string veya None gelirse 0.0 döner, aksi halde float'a çevirir."""
    if x is None:
        return 0.0
    x = x.strip()
    if not x:
        return 0.0
    return float(x)


def summarize_results(
    input_csv: str | Path = None,
    output_csv: str | Path = None,
) -> None:
    """experiment_results.csv dosyasından istatistikleri hesaplar."""

    project_root = Path(__file__).resolve().parents[2]

    if input_csv is None:
        input_csv = project_root / "experiment_results.csv"
    else:
        input_csv = Path(input_csv)

    if output_csv is None:
        output_csv = project_root / "experiment_stats.csv"
    else:
        output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {input_csv}")

    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)

    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["algorithm"],
                int(row["scenario_index"]),
                int(row["source"]),
                int(row["dest"]),
                _safe_float(row["bandwidth_requirement"]),
            )
            groups[key].append(row)

    def compute_stats(values: List[float]) -> MetricStats:
        if not values:
            return MetricStats(0, 0.0, None, 0.0, 0.0)
        count = len(values)
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if count > 1 else 0.0
        vmin = min(values)
        vmax = max(values)
        return MetricStats(count, mean, std, vmin, vmax)

    with output_csv.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            [
                "algorithm",
                "scenario_index",
                "source",
                "dest",
                "bandwidth_requirement",
                "num_runs",
                "mean_total_cost",
                "std_total_cost",
                "min_total_cost",
                "max_total_cost",
                "mean_delay_ms",
                "std_delay_ms",
                "min_delay_ms",
                "max_delay_ms",
                "mean_reliability_cost",
                "std_reliability_cost",
                "min_reliability_cost",
                "max_reliability_cost",
                "mean_resource_cost",
                "std_resource_cost",
                "min_resource_cost",
                "max_resource_cost",
            ]
        )

        for (alg, sc_idx, s, d, B) in sorted(groups.keys(), key=lambda k: (k[0], k[1])):
            rows = groups[(alg, sc_idx, s, d, B)]

            total_cost_vals = [_safe_float(r["total_cost"]) for r in rows]
            delay_vals = [_safe_float(r["delay_ms"]) for r in rows]
            rel_vals = [_safe_float(r["reliability_cost"]) for r in rows]
            res_vals = [_safe_float(r["resource_cost"]) for r in rows]

            stats_total = compute_stats(total_cost_vals)
            stats_delay = compute_stats(delay_vals)
            stats_rel = compute_stats(rel_vals)
            stats_res = compute_stats(res_vals)

            writer.writerow(
                [
                    alg,
                    sc_idx,
                    s,
                    d,
                    B,
                    stats_total.count,
                    stats_total.mean,
                    stats_total.std,
                    stats_total.min,
                    stats_total.max,
                    stats_delay.mean,
                    stats_delay.std,
                    stats_delay.min,
                    stats_delay.max,
                    stats_rel.mean,
                    stats_rel.std,
                    stats_rel.min,
                    stats_rel.max,
                    stats_res.mean,
                    stats_res.std,
                    stats_res.min,
                    stats_res.max,
                ]
            )

    print(f"İstatistik özeti '{output_csv}' dosyasına yazıldı.")


if __name__ == "__main__":
    summarize_results()
