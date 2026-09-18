"""Create or reset the local demo tenant."""

import argparse
from pathlib import Path

from app.config import get_settings
from app.domain import ClinicalEvent
from app.risk import RiskEngine
from app.store import Store
from app.synthetic import demo_patients


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="replace the local demo store")
    parser.add_argument("--path", default=None, help="override the JSON store path")
    args = parser.parse_args()
    settings = get_settings()
    path = args.path or settings.store_path
    if args.reset:
        Path(path).unlink(missing_ok=True)
    store = Store(path)
    patients, events = demo_patients()
    store.seed(patients, events)
    engine = RiskEngine(settings)
    for event in events:
        store.upsert_score(engine.score(ClinicalEvent.model_validate(event)))
    print(f"seeded {len(patients)} patients at {path}")


if __name__ == "__main__":
    main()
