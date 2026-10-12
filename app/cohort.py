"""2–3 万规模合成出院/入院队列（真实规模优化 G1；仅标准库 + 本仓库 app 依赖）。

演示路径（app/synthetic.py）完全不动；本模块只服务 scripts/generate_scale_data.py。
随机性按 (seed, encounter_id, purpose) 独立派生——同一 (scale, seed) 字节级确定，
流式按遭遇消费时与批量顺序无关。
"""

import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.domain import ClinicalEvent

SCALE_DEFAULT = 25000
SCALE_MIN, SCALE_MAX = 1000, 50000
DEFAULT_SEED = 20261012
DEFAULT_HORIZON_DAYS = 540

FACILITY_WEIGHTS = (("hospital-01", 0.45), ("hospital-02", 0.35), ("hospital-03", 0.20))
DEPARTMENTS = ("呼吸内科", "心内科", "消化内科", "神经内科", "普外科", "综合内科")
DEPT_WEIGHTS = (0.24, 0.20, 0.15, 0.13, 0.14, 0.14)

# 生命体征每 6h 一条（0..96h，17 条，与演示 DEMO_SERIAL_HOURS 同构）；
# deterioration landmark 取 6..96h 共 16 格——首格窗口 [index-24, index) 必有历史。
VITAL_HOURS = tuple(range(0, 97, 6))
LANDMARK_HOURS = tuple(range(6, 97, 6))

EVENT_RATE = 0.04  # 遭遇级 24h 恶化事件率（spec §2 设定 4%）
PATIENT_SHARE = 0.82  # 唯一患者数 ≈ scale × 0.82；多次患者各 2 次 → 22% 多次住院自洽

BASE_VITALS: dict[str, tuple[float, float]] = {
    "heart_rate": (78.0, 12.0),
    "respiratory_rate": (16.0, 3.0),
    "systolic_bp": (128.0, 15.0),
    "temperature": (36.8, 0.5),
    "spo2": (97.0, 1.5),
    "wbc": (7.5, 2.5),
    "creatinine": (1.0, 0.4),
    "lactate": (1.2, 0.5),
}
# 事件前 36h 线性起漂的每小时漂移量（阳性 landmark 落在 [event-24, event) → progress ≥ 1/3）
DRIFT_PER_HOUR: dict[str, float] = {
    "heart_rate": 2.2,
    "respiratory_rate": 0.8,
    "systolic_bp": -3.5,
    "temperature": 0.12,
    "spo2": -0.7,
    "wbc": 0.5,
    "creatinine": 0.10,
    "lactate": 0.35,
}
CLAMPS: dict[str, tuple[float, float]] = {
    "heart_rate": (35, 220),
    "respiratory_rate": (4, 60),
    "systolic_bp": (50, 250),
    "temperature": (30, 44),
    "spo2": (50, 100),
    "wbc": (0.1, 50),
    "creatinine": (0.1, 20),
    "lactate": (0.1, 30),
}
ROUND_ONE = frozenset({"temperature", "wbc", "creatinine", "lactate"})
MISSING_P0, MISSING_P1 = 0.05, 0.12  # MAR 缺失注入窗口（随风险升高，spec §2）
READMISSION_LATENT_INTERCEPT = -5.1  # 校准到 30 天再入院 ≈12%（测试窗口 9–15%）


@dataclass(frozen=True)
class ScaleEncounter:
    patient_id: str
    encounter_id: str
    admitted_at: datetime
    discharged_at: datetime
    facility_id: str
    department: str
    age: int
    sex: str
    los_days: float


def _rng(seed: int, encounter_id: str, purpose: str) -> random.Random:
    return random.Random(f"{seed}|{encounter_id}|{purpose}")


def generate_encounters(scale: int, seed: int, horizon_days: int) -> list[ScaleEncounter]:
    """遭遇元数据：患者分配（22% 各 2 次住院）+ 540 天窗内时间 + 院区/科室抽样。"""
    if not SCALE_MIN <= scale <= SCALE_MAX:
        raise ValueError(f"scale must be in [{SCALE_MIN}, {SCALE_MAX}], got {scale}")
    if horizon_days < 90:
        raise ValueError("horizon_days must be >= 90")
    rng = random.Random(f"{seed}|cohort")
    n_patients = max(4, round(scale * PATIENT_SHARE))
    n_multi = min(scale - n_patients, n_patients)  # 多次患者各恰好 2 次遭遇
    assignment = [f"pat-{index:05d}" for index in range(1, n_patients - n_multi + 1)]
    for offset in range(n_multi):
        pid = f"pat-{n_patients - n_multi + offset + 1:05d}"
        assignment.extend((pid, pid))
    rng.shuffle(assignment)
    start = datetime(2025, 4, 1, tzinfo=UTC)
    encounters: list[ScaleEncounter] = []
    for ordinal, pid in enumerate(assignment, start=1):
        los_days = round(0.5 + min(rng.expovariate(0.45), 40.0), 1)
        admitted = start + timedelta(days=rng.uniform(0, max(1.0, horizon_days - los_days)))
        facility = rng.choices(
            [name for name, _ in FACILITY_WEIGHTS], [weight for _, weight in FACILITY_WEIGHTS]
        )[0]
        encounters.append(
            ScaleEncounter(
                patient_id=pid,
                encounter_id=f"enc-{ordinal:05d}",
                admitted_at=admitted,
                discharged_at=admitted + timedelta(days=los_days),
                facility_id=facility,
                department=rng.choices(DEPARTMENTS, DEPT_WEIGHTS)[0],
                age=rng.randint(35, 94),
                sex="F" if rng.random() < 0.5 else "M",
                los_days=los_days,
            )
        )
    return encounters


def _deterioration_draw(enc: ScaleEncounter, seed: int) -> tuple[int, str] | None:
    """遭遇级 4% 抽样 → (入院后小时 6..96, 结局类型)；None = 无事件。"""
    rng = _rng(seed, enc.encounter_id, "deterioration")
    if rng.random() >= EVENT_RATE:
        return None
    hour = rng.randrange(6, 97)
    kind = rng.choices(
        ("rrt", "unplanned_icu_transfer", "cardiac_arrest", "death"),
        weights=(45, 40, 5, 10),
    )[0]
    return hour, kind


def deterioration_outcome(enc: ScaleEncounter, seed: int) -> dict | None:
    """24h 恶化结局事件（derive_labels 白名单内类型）。"""
    draw = _deterioration_draw(enc, seed)
    if draw is None:
        return None
    hour, kind = draw
    return {
        "encounter_id": enc.encounter_id,
        "occurred_at": (enc.admitted_at + timedelta(hours=hour)).isoformat(),
        "outcome_type": kind,
    }


def vital_events(enc: ScaleEncounter, seed: int) -> list[ClinicalEvent]:
    """每 6h 一条生命体征（0..96h）；事件遭遇前 36h 线性漂移 + MAR 缺失 5–12%。"""
    rng = _rng(seed, enc.encounter_id, "vitals")
    draw = _deterioration_draw(enc, seed)
    event_hour = None if draw is None else draw[0]
    prior_admissions = rng.randint(1, 3) if event_hour is not None else rng.randint(0, 1)
    events: list[ClinicalEvent] = []
    for hour in VITAL_HOURS:
        if event_hour is not None:
            progress = max(0.0, (hour - (event_hour - 36)) / 36)
        else:
            progress = 0.0
        risk = (0.65 if event_hour is not None else 0.15) * (0.4 + 0.6 * hour / 96)
        p_missing = MISSING_P0 + (MISSING_P1 - MISSING_P0) * min(1.0, risk)
        payload: dict[str, object] = {"age": enc.age, "prior_admissions_30d": prior_admissions}
        for feature, (mean, std) in BASE_VITALS.items():
            if rng.random() < p_missing:
                continue  # MAR 缺失：键不出现（build_landmarks 按缺测跳过）
            value = rng.gauss(mean, std) + DRIFT_PER_HOUR[feature] * progress
            low, high = CLAMPS[feature]
            value = min(max(value, low), high)
            payload[feature] = round(value, 1) if feature in ROUND_ONE else round(value)
        events.append(
            ClinicalEvent(
                event_type="vital_signs",
                patient_id=enc.patient_id,
                encounter_id=enc.encounter_id,
                occurred_at=enc.admitted_at + timedelta(hours=hour),
                payload=payload,
            )
        )
    return events


def readmission_snapshot(enc: ScaleEncounter, seed: int) -> tuple[dict, dict | None]:
    """出院快照 6 源字段 + 30 天非计划再入院结局（字段驱动 latent ≈12%）。

    快照不注入缺失：LACE 基线要求字段完整（门禁完整性约束，R-RS-4）。
    """
    rng = _rng(seed, enc.encounter_id, "snapshot")
    field_rng = _rng(seed, enc.encounter_id, "fields")
    comorbidity = round(min(14.0, field_rng.expovariate(0.5)), 1)
    prior_util = min(6, int(field_rng.expovariate(0.8)))
    ed_visits = min(8, int(field_rng.expovariate(1.0)))
    destination = field_rng.choices(
        ["home", "home_health", "snf", "other"], weights=[55, 20, 15, 10]
    )[0]
    code_count = int(comorbidity) + field_rng.randrange(3)
    snapshot = {
        "los_days": enc.los_days,
        "prior_utilization_12m": prior_util,
        "comorbidity_index": comorbidity,
        "ed_visits_6m": ed_visits,
        "diagnosis_codes": [f"D{i:02d}" for i in range(code_count)],
        "discharge_destination": destination,
    }
    latent = (
        READMISSION_LATENT_INTERCEPT
        + 0.10 * enc.los_days
        + 0.38 * prior_util
        + 0.26 * comorbidity
        + 0.22 * ed_visits
        + 0.30 * code_count
        + (0.0 if destination == "home" else 0.45)
        + 0.022 * max(enc.age - 50, 0)
    )
    if rng.random() >= 1 / (1 + math.exp(-latent)):
        return snapshot, None
    outcome = {
        "encounter_id": enc.encounter_id,
        # 1..29 天（24–696h）必落 [discharge, discharge+720h) 开窗内
        "occurred_at": (enc.discharged_at + timedelta(days=1 + rng.randrange(29))).isoformat(),
        "outcome_type": "unplanned_readmission",
    }
    return snapshot, outcome
