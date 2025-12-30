from __future__ import annotations

from dataclasses import dataclass
from typing import List
import csv
from pathlib import Path


@dataclass
class Scenario:
    """Deney senaryosu tanımı.

    - s: kaynak düğüm ID
    - d: hedef düğüm ID
    - bandwidth_requirement: minimum gerekli bant genişliği (Mbps)
    """

    s: int
    d: int
    bandwidth_requirement: float


# Örnek 20 senaryo (daha sonra isteğe göre değiştirilebilir / genişletilebilir)
SCENARIOS: List[Scenario] = [
    Scenario(0, 149, 100.0),
    Scenario(3, 87, 200.0),
    Scenario(10, 200, 150.0),
    Scenario(5, 120, 250.0),
    Scenario(12, 220, 300.0),
    Scenario(33, 44, 100.0),
    Scenario(40, 199, 400.0),
    Scenario(50, 180, 500.0),
    Scenario(60, 170, 150.0),
    Scenario(70, 160, 200.0),
    Scenario(80, 150, 250.0),
    Scenario(90, 140, 300.0),
    Scenario(100, 130, 350.0),
    Scenario(110, 120, 400.0),
    Scenario(20, 30, 100.0),
    Scenario(45, 210, 450.0),
    Scenario(77, 155, 200.0),
    Scenario(88, 166, 300.0),
    Scenario(99, 177, 350.0),
    Scenario(123, 200, 250.0),
]

def load_scenarios_from_csv(csv_path: str | Path) -> List[Scenario]:
    """DemandData.csv formatından senaryo listesi üretir.

    Beklenen kolonlar:
        src ; dst ; demand_mbps
    """
    csv_path = Path(csv_path)
    scenarios: List[Scenario] = []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            demand_str = str(row.get("demand_mbps", "")).strip()
            demand_val = float(demand_str.replace(",", ".")) if demand_str else 0.0
            scenarios.append(
                Scenario(
                    s=int(row["src"]),
                    d=int(row["dst"]),
                    bandwidth_requirement=demand_val,
                )
            )

    return scenarios
