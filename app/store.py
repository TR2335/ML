import json
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

from .domain import AuditEvent, ClinicalEvent, RiskScore, TargetId


class Store:
    """Transactional local store.

    SQLite is used for the runnable development slice because it supports WAL,
    transactions, and multiple API/worker processes. Production should replace
    this adapter with PostgreSQL without changing the API contract.
    """

    def __init__(self, path: str):
        requested_path = Path(path)
        if requested_path.exists() and requested_path.is_file():
            try:
                prefix = requested_path.read_bytes()[:1]
            except OSError:
                prefix = b""
            if prefix in {b"{", b"["}:
                requested_path = requested_path.with_suffix(".db")
        self.path = requested_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI executes sync handlers in a thread pool. A single SQLite
        # connection is safe to share only when access is serialized; the
        # process lock also makes nested read/write helpers re-entrant.
        self._lock = RLock()
        self.connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _create_schema(self) -> None:
        with self._lock:
            self._migrate_legacy_scope_keys()
            self._migrate_score_targets()
            self._migrate_event_tenant_scope()
            with self.connection:
                self.connection.executescript(
                    """
                CREATE TABLE IF NOT EXISTS patients (
                    encounter_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    facility_id TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, encounter_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT NOT NULL,
                    encounter_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    facility_id TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    FOREIGN KEY (tenant_id, encounter_id)
                        REFERENCES patients(tenant_id, encounter_id),
                    PRIMARY KEY (tenant_id, event_id)
                );
                CREATE INDEX IF NOT EXISTS events_tenant_encounter_time
                    ON events(tenant_id, encounter_id, occurred_at);
                CREATE TABLE IF NOT EXISTS scores (
                    encounter_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    facility_id TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    target_id TEXT NOT NULL DEFAULT 'deterioration_24h',
                    PRIMARY KEY (tenant_id, encounter_id, target_id),
                    FOREIGN KEY (tenant_id, encounter_id)
                        REFERENCES patients(tenant_id, encounter_id)
                );
                CREATE TABLE IF NOT EXISTS audits (
                    audit_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    facility_id TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );
                """
                )

    def _migrate_legacy_scope_keys(self) -> None:
        """Upgrade the original encounter-only schema without dropping data."""
        table = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'patients'"
        ).fetchone()
        if not table:
            return
        columns = self.connection.execute("PRAGMA table_info(patients)").fetchall()
        primary_key = [row["name"] for row in columns if row["pk"]]
        if primary_key != ["encounter_id"]:
            return
        self.connection.execute("PRAGMA foreign_keys=OFF")
        try:
            with self.connection:
                self.connection.execute("DROP INDEX IF EXISTS events_encounter_time")
                self.connection.execute("ALTER TABLE patients RENAME TO patients_legacy")
                self.connection.execute("ALTER TABLE events RENAME TO events_legacy")
                self.connection.execute("ALTER TABLE scores RENAME TO scores_legacy")
                self.connection.executescript(
                    """
                    CREATE TABLE patients (
                        encounter_id TEXT NOT NULL,
                        patient_id TEXT NOT NULL,
                        tenant_id TEXT NOT NULL,
                        facility_id TEXT NOT NULL,
                        data_json TEXT NOT NULL,
                        PRIMARY KEY (tenant_id, encounter_id)
                    );
                    CREATE TABLE events (
                        event_id TEXT PRIMARY KEY,
                        encounter_id TEXT NOT NULL,
                        patient_id TEXT NOT NULL,
                        tenant_id TEXT NOT NULL,
                        facility_id TEXT NOT NULL,
                        occurred_at TEXT NOT NULL,
                        data_json TEXT NOT NULL,
                        FOREIGN KEY (tenant_id, encounter_id)
                            REFERENCES patients(tenant_id, encounter_id)
                    );
                    CREATE TABLE scores (
                        encounter_id TEXT NOT NULL,
                        tenant_id TEXT NOT NULL,
                        facility_id TEXT NOT NULL,
                        data_json TEXT NOT NULL,
                        PRIMARY KEY (tenant_id, encounter_id),
                        FOREIGN KEY (tenant_id, encounter_id)
                            REFERENCES patients(tenant_id, encounter_id)
                    );
                    INSERT INTO patients SELECT encounter_id, patient_id, tenant_id, facility_id, data_json FROM patients_legacy;
                    INSERT INTO events SELECT event_id, encounter_id, patient_id, tenant_id, facility_id, occurred_at, data_json FROM events_legacy;
                    INSERT INTO scores SELECT encounter_id, tenant_id, facility_id, data_json FROM scores_legacy;
                    DROP TABLE events_legacy;
                    DROP TABLE scores_legacy;
                    DROP TABLE patients_legacy;
                    """
                )
        finally:
            self.connection.execute("PRAGMA foreign_keys=ON")

    def _migrate_score_targets(self) -> None:
        """Add target_id to scores while preserving existing deterioration scores."""
        table = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'scores'"
        ).fetchone()
        if not table:
            return
        columns = self.connection.execute("PRAGMA table_info(scores)").fetchall()
        if any(row["name"] == "target_id" for row in columns):
            return
        self.connection.execute("PRAGMA foreign_keys=OFF")
        try:
            with self.connection:
                self.connection.execute("ALTER TABLE scores RENAME TO scores_legacy")
                self.connection.executescript(
                    """
                    CREATE TABLE scores (
                        encounter_id TEXT NOT NULL,
                        tenant_id TEXT NOT NULL,
                        facility_id TEXT NOT NULL,
                        data_json TEXT NOT NULL,
                        target_id TEXT NOT NULL DEFAULT 'deterioration_24h',
                        PRIMARY KEY (tenant_id, encounter_id, target_id),
                        FOREIGN KEY (tenant_id, encounter_id)
                            REFERENCES patients(tenant_id, encounter_id)
                    );
                    INSERT INTO scores(encounter_id, tenant_id, facility_id, data_json, target_id)
                        SELECT encounter_id, tenant_id, facility_id, data_json, 'deterioration_24h'
                        FROM scores_legacy;
                    DROP TABLE scores_legacy;
                    """
                )
        finally:
            self.connection.execute("PRAGMA foreign_keys=ON")

    def _migrate_event_tenant_scope(self) -> None:
        """Re-key events to (tenant_id, event_id) so cross-tenant ids coexist."""
        table = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'events'"
        ).fetchone()
        if not table:
            return
        columns = self.connection.execute("PRAGMA table_info(events)").fetchall()
        primary_key = [row["name"] for row in columns if row["pk"]]
        if primary_key == ["tenant_id", "event_id"]:
            return
        self.connection.execute("PRAGMA foreign_keys=OFF")
        try:
            with self.connection:
                self.connection.execute("ALTER TABLE events RENAME TO events_legacy")
                self.connection.executescript(
                    """
                    CREATE TABLE events (
                        event_id TEXT NOT NULL,
                        encounter_id TEXT NOT NULL,
                        patient_id TEXT NOT NULL,
                        tenant_id TEXT NOT NULL,
                        facility_id TEXT NOT NULL,
                        occurred_at TEXT NOT NULL,
                        data_json TEXT NOT NULL,
                        FOREIGN KEY (tenant_id, encounter_id)
                            REFERENCES patients(tenant_id, encounter_id),
                        PRIMARY KEY (tenant_id, event_id)
                    );
                    CREATE INDEX IF NOT EXISTS events_tenant_encounter_time
                        ON events(tenant_id, encounter_id, occurred_at);
                    INSERT INTO events(event_id, encounter_id, patient_id, tenant_id, facility_id, occurred_at, data_json)
                        SELECT event_id, encounter_id, patient_id, tenant_id, facility_id, occurred_at, data_json
                        FROM events_legacy;
                    DROP TABLE events_legacy;
                    """
                )
        finally:
            self.connection.execute("PRAGMA foreign_keys=ON")

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)

    @staticmethod
    def _loads(value: str) -> dict:
        return json.loads(value)

    def seed(self, patients: list[dict], events: list[ClinicalEvent]) -> None:
        with self._lock, self.connection:
            if self.connection.execute("SELECT 1 FROM patients LIMIT 1").fetchone():
                return
            for patient in patients:
                tenant_id = patient.get("tenant_id", "group-demo")
                facility_id = patient.get("facility_id", "hospital-01")
                self.connection.execute(
                    "INSERT OR IGNORE INTO patients(encounter_id, patient_id, tenant_id, facility_id, data_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        patient["encounter_id"],
                        patient["patient_id"],
                        tenant_id,
                        facility_id,
                        self._json({**patient, "tenant_id": tenant_id, "facility_id": facility_id}),
                    ),
                )
            for event in events:
                self._insert_event(event)

    def upsert_patient(self, patient: dict) -> None:
        tenant_id = patient.get("tenant_id", "group-demo")
        facility_id = patient.get("facility_id", "hospital-01")
        with self._lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO patients(encounter_id, patient_id, tenant_id, facility_id, data_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, encounter_id) DO UPDATE SET
                    patient_id=excluded.patient_id,
                    facility_id=excluded.facility_id,
                    data_json=excluded.data_json
                """,
                (
                    patient["encounter_id"],
                    patient["patient_id"],
                    tenant_id,
                    facility_id,
                    self._json({**patient, "tenant_id": tenant_id, "facility_id": facility_id}),
                ),
            )

    def _insert_event(self, event: ClinicalEvent) -> bool:
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO events(event_id, encounter_id, patient_id, tenant_id, facility_id, occurred_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.encounter_id,
                event.patient_id,
                event.tenant_id,
                event.facility_id,
                event.occurred_at.isoformat(),
                self._json(event.model_dump(mode="json")),
            ),
        )
        return cursor.rowcount == 1

    def has_encounter(
        self, encounter_id: str, patient_id: str, tenant_id: str, facility_id: str
    ) -> bool:
        with self._lock:
            row = self.connection.execute(
                "SELECT 1 FROM patients WHERE encounter_id = ? AND patient_id = ? AND tenant_id = ? AND facility_id = ?",
                (encounter_id, patient_id, tenant_id, facility_id),
            ).fetchone()
        return row is not None

    def event_exists(self, event_id: str) -> bool:
        with self._lock:
            return (
                self.connection.execute(
                    "SELECT 1 FROM events WHERE event_id = ?", (event_id,)
                ).fetchone()
                is not None
            )

    def get_event(self, event_id: str, tenant_id: str) -> ClinicalEvent | None:
        """Return a tenant-scoped event by id for safe idempotency conflict checks."""
        with self._lock:
            row = self.connection.execute(
                "SELECT data_json FROM events WHERE event_id = ? AND tenant_id = ?",
                (event_id, tenant_id),
            ).fetchone()
        return ClinicalEvent.model_validate(self._loads(row["data_json"])) if row else None

    def list_patients(
        self,
        tenant_id: str,
        facility_id: str | None = None,
        band: str | None = None,
        target_id: TargetId = "deterioration_24h",
    ) -> list[dict]:
        with self._lock:
            query = "SELECT p.data_json, s.data_json AS score_json FROM patients p LEFT JOIN scores s ON p.encounter_id = s.encounter_id AND p.tenant_id = s.tenant_id AND s.target_id = ? WHERE p.tenant_id = ?"
            params: list[Any] = [target_id, tenant_id]
            if facility_id:
                query += " AND p.facility_id = ?"
                params.append(facility_id)
            rows = []
            for row in self.connection.execute(query, params).fetchall():
                patient = self._loads(row["data_json"])
                score = self._loads(row["score_json"]) if row["score_json"] else None
                if band and (not score or score["band"] != band):
                    continue
                if score:
                    patient["risk"] = score
                rows.append(patient)
        return sorted(rows, key=lambda item: item.get("risk", {}).get("score", 0), reverse=True)

    def list_all_patients(self) -> list[dict]:
        with self._lock:
            rows = []
            for row in self.connection.execute(
                "SELECT data_json FROM patients ORDER BY encounter_id"
            ).fetchall():
                rows.append(self._loads(row["data_json"]))
        return rows

    def get_patient(
        self, encounter_id: str, tenant_id: str, facility_id: str | None = None
    ) -> dict | None:
        with self._lock:
            query = "SELECT p.data_json, s.data_json AS score_json FROM patients p LEFT JOIN scores s ON p.encounter_id = s.encounter_id AND p.tenant_id = s.tenant_id AND s.target_id = ? WHERE p.encounter_id = ? AND p.tenant_id = ?"
            params: list[Any] = ["deterioration_24h", encounter_id, tenant_id]
            if facility_id:
                query += " AND p.facility_id = ?"
                params.append(facility_id)
            row = self.connection.execute(query, params).fetchone()
        if not row:
            return None
        patient = self._loads(row["data_json"])
        if row["score_json"]:
            patient["risk"] = self._loads(row["score_json"])
        return patient

    def events_for(
        self, encounter_id: str, tenant_id: str, facility_id: str | None = None
    ) -> list[ClinicalEvent]:
        with self._lock:
            query = "SELECT data_json FROM events WHERE encounter_id = ? AND tenant_id = ?"
            params: list[Any] = [encounter_id, tenant_id]
            if facility_id:
                query += " AND facility_id = ?"
                params.append(facility_id)
            query += " ORDER BY occurred_at ASC, event_id ASC"
            return [
                ClinicalEvent.model_validate(self._loads(row["data_json"]))
                for row in self.connection.execute(query, params).fetchall()
            ]

    def feature_snapshot(
        self,
        encounter_id: str,
        tenant_id: str,
        facility_id: str | None = None,
        as_of: datetime | None = None,
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for event in self.events_for(encounter_id, tenant_id, facility_id):
            if as_of is not None and event.occurred_at > as_of:
                continue
            for key, value in event.payload.items():
                if value is not None:
                    snapshot[key] = value
        return snapshot

    def latest_event_time(
        self, encounter_id: str, tenant_id: str, facility_id: str | None = None
    ) -> datetime | None:
        """Return the latest known event time for an encounter."""
        with self._lock:
            query = "SELECT occurred_at FROM events WHERE encounter_id = ? AND tenant_id = ?"
            params: list[Any] = [encounter_id, tenant_id]
            if facility_id:
                query += " AND facility_id = ?"
                params.append(facility_id)
            query += " ORDER BY occurred_at DESC, event_id DESC LIMIT 1"
            row = self.connection.execute(query, params).fetchone()
        return datetime.fromisoformat(row["occurred_at"]) if row else None

    def add_event(self, event: ClinicalEvent) -> bool:
        with self._lock, self.connection:
            return self._insert_event(event)

    def get_score(
        self,
        encounter_id: str,
        tenant_id: str,
        facility_id: str | None = None,
        target_id: TargetId = "deterioration_24h",
    ) -> RiskScore | None:
        with self._lock:
            query = "SELECT data_json FROM scores WHERE encounter_id = ? AND tenant_id = ? AND target_id = ?"
            params: list[Any] = [encounter_id, tenant_id, target_id]
            if facility_id:
                query += " AND facility_id = ?"
                params.append(facility_id)
            row = self.connection.execute(query, params).fetchone()
        return RiskScore.model_validate(self._loads(row["data_json"])) if row else None

    def get_scores(
        self, encounter_id: str, tenant_id: str, facility_id: str | None = None
    ) -> dict[str, RiskScore]:
        with self._lock:
            query = "SELECT data_json FROM scores WHERE encounter_id = ? AND tenant_id = ?"
            params: list[Any] = [encounter_id, tenant_id]
            if facility_id:
                query += " AND facility_id = ?"
                params.append(facility_id)
            rows = self.connection.execute(query, params).fetchall()
        scores = [RiskScore.model_validate(self._loads(row["data_json"])) for row in rows]
        return {score.target_id: score for score in scores}

    def upsert_score(self, score: RiskScore, *, preserve_review: bool = True) -> str:
        with self._lock, self.connection:
            previous = self.get_score(
                score.encounter_id,
                score.tenant_id,
                score.facility_id,
                target_id=score.target_id,
            )
            if previous:
                # A legacy row without as_of has unknown staleness: never block
                # rescoring on it (preserve_review still guards human review).
                if previous.as_of is not None:
                    new_as_of = score.as_of or score.scored_at
                    if new_as_of < previous.as_of:
                        return "skipped_stale"
                score.score_id = previous.score_id
                if preserve_review and previous.status != "new":
                    score.status = previous.status
                    score.human_review = previous.human_review
            self.connection.execute(
                "INSERT INTO scores(encounter_id, tenant_id, facility_id, data_json, target_id) VALUES (?, ?, ?, ?, ?) ON CONFLICT(tenant_id, encounter_id, target_id) DO UPDATE SET facility_id=excluded.facility_id, data_json=excluded.data_json",
                (
                    score.encounter_id,
                    score.tenant_id,
                    score.facility_id,
                    self._json(score.model_dump(mode="json")),
                    score.target_id,
                ),
            )
        return "applied"

    def add_audit(self, event: AuditEvent) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO audits(audit_id, tenant_id, facility_id, occurred_at, data_json) VALUES (?, ?, ?, ?, ?)",
                (
                    event.audit_id,
                    event.tenant_id,
                    event.facility_id,
                    event.occurred_at.isoformat(),
                    self._json(event.model_dump(mode="json")),
                ),
            )

    def audits(
        self, tenant_id: str, facility_id: str | None = None, limit: int = 100
    ) -> list[AuditEvent]:
        with self._lock:
            query = "SELECT data_json FROM audits WHERE tenant_id = ?"
            params: list[Any] = [tenant_id]
            if facility_id:
                query += " AND facility_id = ?"
                params.append(facility_id)
            query += " ORDER BY occurred_at DESC LIMIT ?"
            params.append(limit)
            return [
                AuditEvent.model_validate(self._loads(row["data_json"]))
                for row in self.connection.execute(query, params).fetchall()
            ]
