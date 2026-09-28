from datetime import UTC, datetime, timedelta
from random import Random

from .domain import ClinicalEvent

FEATURES = [
    "age",
    "heart_rate",
    "respiratory_rate",
    "systolic_bp",
    "temperature",
    "spo2",
    "wbc",
    "creatinine",
    "lactate",
    "prior_admissions_30d",
]


def demo_patients(count: int = 18, seed: int = 7) -> tuple[list[dict], list[ClinicalEvent]]:
    rng = Random(seed)
    patients: list[dict] = []
    events: list[ClinicalEvent] = []
    base = datetime.now(UTC) - timedelta(hours=2)
    for index in range(1, count + 1):
        patient_id = f"pat-{index:03d}"
        encounter_id = f"enc-{index:03d}"
        high_risk = index in {2, 5, 9, 14}
        payload = {
            "age": rng.randint(22, 91),
            "heart_rate": rng.randint(105, 138) if high_risk else rng.randint(62, 102),
            "respiratory_rate": rng.randint(23, 36) if high_risk else rng.randint(12, 21),
            "systolic_bp": rng.randint(78, 101) if high_risk else rng.randint(105, 148),
            "temperature": round(rng.uniform(38.0, 39.6), 1)
            if high_risk
            else round(rng.uniform(36.2, 37.5), 1),
            "spo2": rng.randint(84, 94) if high_risk else rng.randint(95, 100),
            "wbc": round(rng.uniform(13, 23), 1) if high_risk else round(rng.uniform(4, 12), 1),
            "creatinine": round(rng.uniform(1.5, 3.5), 1)
            if high_risk
            else round(rng.uniform(0.6, 1.4), 1),
            "lactate": round(rng.uniform(2.2, 5.0), 1)
            if high_risk
            else round(rng.uniform(0.7, 1.9), 1),
            "prior_admissions_30d": rng.randint(1, 3) if high_risk else rng.randint(0, 1),
        }
        patients.append(
            {
                "patient_id": patient_id,
                "encounter_id": encounter_id,
                "display_name": f"演示患者 {index:03d}",
                "age": payload["age"],
                "sex": "F" if index % 2 else "M",
                "department": ["呼吸内科", "心内科", "综合内科"][index % 3],
                "admitted_at": (base - timedelta(hours=index * 3)).isoformat(),
                "status": "inpatient",
                "tenant_id": "group-demo",
                "facility_id": "hospital-01",
            }
        )
        events.append(
            ClinicalEvent(
                event_type="vital_signs",
                patient_id=patient_id,
                encounter_id=encounter_id,
                occurred_at=base,
                payload=payload,
            )
        )
    return patients, events


DEMO_HIGH_RISK = frozenset({2, 5, 9, 14})
DEMO_SERIAL_HOURS = range(0, 97, 6)  # 每 6h 一条生命体征，0..96h
DEMO_RRT_DELAY_HOURS = 48


def demo_patients_with_outcomes(
    count: int = 18, seed: int = 7
) -> tuple[list[dict], list[ClinicalEvent], list[dict]]:
    """演示管线专用：患者 + 串行生命体征 + 高风险患者的 RRT 结局（与真实管道同构）。"""
    patients, first_events = demo_patients(count=count, seed=seed)
    events: list[ClinicalEvent] = []
    outcomes: list[dict] = []
    for patient, first in zip(patients, first_events):
        encounter_id = patient["encounter_id"]
        index = int(encounter_id.split("-")[1])
        for hour in DEMO_SERIAL_HOURS:
            events.append(
                first.model_copy(
                    update={
                        "event_id": f"{first.event_id}-h{hour}",
                        "occurred_at": first.occurred_at + timedelta(hours=hour),
                    }
                )
            )
        if index in DEMO_HIGH_RISK:
            outcomes.append(
                {
                    "encounter_id": encounter_id,
                    "occurred_at": (
                        first.occurred_at + timedelta(hours=DEMO_RRT_DELAY_HOURS)
                    ).isoformat(),
                    "outcome_type": "rrt",
                }
            )
    return patients, events, outcomes
