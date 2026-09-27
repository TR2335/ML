"""PostgreSQL storage adapter for production deployments.

The adapter deliberately keeps the same methods and JSON contract as the
SQLite development store. PostgreSQL provides connection pooling, JSONB,
transaction isolation and durable backups for the hospital deployment.
"""

from datetime import datetime
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .domain import AuditEvent, ClinicalEvent, RiskScore, TargetId


class PostgresStore:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 10):
        self.pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
        )
        self._create_schema()

    def close(self) -> None:
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def _json(value: Any) -> Jsonb:
        return Jsonb(value)

    def _create_schema(self) -> None:
        with self.pool.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    encounter_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    facility_id TEXT NOT NULL,
                    data_json JSONB NOT NULL,
                    PRIMARY KEY (tenant_id, encounter_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT NOT NULL,
                    encounter_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    facility_id TEXT NOT NULL,
                    occurred_at TIMESTAMPTZ NOT NULL,
                    data_json JSONB NOT NULL,
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
                    data_json JSONB NOT NULL,
                    target_id TEXT NOT NULL DEFAULT 'deterioration_24h',
                    PRIMARY KEY (tenant_id, encounter_id, target_id),
                    FOREIGN KEY (tenant_id, encounter_id)
                        REFERENCES patients(tenant_id, encounter_id)
                );
                CREATE TABLE IF NOT EXISTS audits (
                    audit_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    facility_id TEXT NOT NULL,
                    occurred_at TIMESTAMPTZ NOT NULL,
                    data_json JSONB NOT NULL
                );
                """
            )
            # Re-key a pre-existing global-PK events table to (tenant_id, event_id).
            connection.execute(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM pg_constraint c
                        WHERE c.conname = 'events_pkey'
                          AND c.conrelid = 'events'::regclass
                          AND (
                              SELECT array_agg(a.attname ORDER BY k.ord)
                              FROM unnest(c.conkey) WITH ORDINALITY AS k(attnum, ord)
                              JOIN pg_attribute a
                                ON a.attrelid = c.conrelid AND a.attnum = k.attnum
                          ) <> ARRAY['tenant_id', 'event_id']
                    ) THEN
                        ALTER TABLE events DROP CONSTRAINT events_pkey;
                        ALTER TABLE events ADD PRIMARY KEY (tenant_id, event_id);
                    END IF;
                END $$;
                """
            )

    def seed(self, patients: list[dict], events: list[ClinicalEvent]) -> None:
        with self.pool.connection() as connection:
            if connection.execute("SELECT 1 FROM patients LIMIT 1").fetchone():
                return
            for patient in patients:
                tenant_id = patient.get("tenant_id", "group-demo")
                facility_id = patient.get("facility_id", "hospital-01")
                connection.execute(
                    """
                    INSERT INTO patients(encounter_id, patient_id, tenant_id, facility_id, data_json)
                    VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING
                    """,
                    (
                        patient["encounter_id"],
                        patient["patient_id"],
                        tenant_id,
                        facility_id,
                        self._json({**patient, "tenant_id": tenant_id, "facility_id": facility_id}),
                    ),
                )
            for event in events:
                self._insert_event(connection, event)

    def upsert_patient(self, patient: dict) -> None:
        tenant_id = patient.get("tenant_id", "group-demo")
        facility_id = patient.get("facility_id", "hospital-01")
        with self.pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO patients(encounter_id, patient_id, tenant_id, facility_id, data_json)
                VALUES (%s, %s, %s, %s, %s)
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

    @staticmethod
    def _insert_event(connection, event: ClinicalEvent) -> bool:
        cursor = connection.execute(
            """
            INSERT INTO events(event_id, encounter_id, patient_id, tenant_id, facility_id, occurred_at, data_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING
            """,
            (
                event.event_id,
                event.encounter_id,
                event.patient_id,
                event.tenant_id,
                event.facility_id,
                event.occurred_at,
                PostgresStore._json(event.model_dump(mode="json")),
            ),
        )
        return cursor.rowcount == 1

    def has_encounter(
        self, encounter_id: str, patient_id: str, tenant_id: str, facility_id: str
    ) -> bool:
        with self.pool.connection() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM patients WHERE encounter_id=%s AND patient_id=%s AND tenant_id=%s AND facility_id=%s",
                    (encounter_id, patient_id, tenant_id, facility_id),
                ).fetchone()
                is not None
            )

    def get_event(self, event_id: str, tenant_id: str) -> ClinicalEvent | None:
        with self.pool.connection() as connection:
            row = connection.execute(
                "SELECT data_json FROM events WHERE event_id=%s AND tenant_id=%s",
                (event_id, tenant_id),
            ).fetchone()
        return ClinicalEvent.model_validate(row["data_json"]) if row else None

    def list_patients(
        self,
        tenant_id: str,
        facility_id: str | None = None,
        band: str | None = None,
        target_id: TargetId = "deterioration_24h",
    ) -> list[dict]:
        query = """
            SELECT p.data_json, s.data_json AS score_json
            FROM patients p
            LEFT JOIN scores s ON p.encounter_id=s.encounter_id
                AND p.tenant_id=s.tenant_id AND s.target_id=%s
            WHERE p.tenant_id=%s
        """
        params: list[Any] = [target_id, tenant_id]
        if facility_id:
            query += " AND p.facility_id=%s"
            params.append(facility_id)
        with self.pool.connection() as connection:
            rows = connection.execute(query, params).fetchall()
        result = []
        for row in rows:
            patient = dict(row["data_json"])
            score = row["score_json"]
            if band and (not score or score["band"] != band):
                continue
            if score:
                patient["risk"] = score
            result.append(patient)
        return sorted(result, key=lambda item: item.get("risk", {}).get("score", 0), reverse=True)

    def list_all_patients(self) -> list[dict]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                "SELECT data_json FROM patients ORDER BY encounter_id"
            ).fetchall()
        return [dict(row["data_json"]) for row in rows]

    def get_patient(
        self, encounter_id: str, tenant_id: str, facility_id: str | None = None
    ) -> dict | None:
        query = """
            SELECT p.data_json, s.data_json AS score_json
            FROM patients p
            LEFT JOIN scores s ON p.encounter_id=s.encounter_id
                AND p.tenant_id=s.tenant_id AND s.target_id='deterioration_24h'
            WHERE p.encounter_id=%s AND p.tenant_id=%s
        """
        params: list[Any] = [encounter_id, tenant_id]
        if facility_id:
            query += " AND p.facility_id=%s"
            params.append(facility_id)
        with self.pool.connection() as connection:
            row = connection.execute(query, params).fetchone()
        if not row:
            return None
        patient = dict(row["data_json"])
        if row["score_json"]:
            patient["risk"] = dict(row["score_json"])
        return patient

    def events_for(
        self, encounter_id: str, tenant_id: str, facility_id: str | None = None
    ) -> list[ClinicalEvent]:
        query = "SELECT data_json FROM events WHERE encounter_id=%s AND tenant_id=%s"
        params: list[Any] = [encounter_id, tenant_id]
        if facility_id:
            query += " AND facility_id=%s"
            params.append(facility_id)
        query += " ORDER BY occurred_at ASC, event_id ASC"
        with self.pool.connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [ClinicalEvent.model_validate(row["data_json"]) for row in rows]

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
        query = "SELECT occurred_at FROM events WHERE encounter_id=%s AND tenant_id=%s"
        params: list[Any] = [encounter_id, tenant_id]
        if facility_id:
            query += " AND facility_id=%s"
            params.append(facility_id)
        query += " ORDER BY occurred_at DESC, event_id DESC LIMIT 1"
        with self.pool.connection() as connection:
            row = connection.execute(query, params).fetchone()
        return row["occurred_at"] if row else None

    def add_event(self, event: ClinicalEvent) -> bool:
        with self.pool.connection() as connection:
            return self._insert_event(connection, event)

    def get_score(
        self,
        encounter_id: str,
        tenant_id: str,
        facility_id: str | None = None,
        target_id: TargetId = "deterioration_24h",
    ) -> RiskScore | None:
        query = (
            "SELECT data_json FROM scores WHERE encounter_id=%s AND tenant_id=%s AND target_id=%s"
        )
        params: list[Any] = [encounter_id, tenant_id, target_id]
        if facility_id:
            query += " AND facility_id=%s"
            params.append(facility_id)
        with self.pool.connection() as connection:
            row = connection.execute(query, params).fetchone()
        return RiskScore.model_validate(row["data_json"]) if row else None

    def get_scores(
        self, encounter_id: str, tenant_id: str, facility_id: str | None = None
    ) -> dict[str, RiskScore]:
        query = "SELECT data_json FROM scores WHERE encounter_id=%s AND tenant_id=%s"
        params: list[Any] = [encounter_id, tenant_id]
        if facility_id:
            query += " AND facility_id=%s"
            params.append(facility_id)
        with self.pool.connection() as connection:
            rows = connection.execute(query, params).fetchall()
        scores = [RiskScore.model_validate(row["data_json"]) for row in rows]
        return {score.target_id: score for score in scores}

    def upsert_score(self, score: RiskScore, *, preserve_review: bool = True) -> str:
        previous = self.get_score(
            score.encounter_id, score.tenant_id, score.facility_id, score.target_id
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
        with self.pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO scores(encounter_id, tenant_id, facility_id, data_json, target_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT(tenant_id, encounter_id, target_id)
                DO UPDATE SET facility_id=excluded.facility_id, data_json=excluded.data_json
                """,
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
        with self.pool.connection() as connection:
            connection.execute(
                "INSERT INTO audits(audit_id, tenant_id, facility_id, occurred_at, data_json) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (
                    event.audit_id,
                    event.tenant_id,
                    event.facility_id,
                    event.occurred_at,
                    self._json(event.model_dump(mode="json")),
                ),
            )

    def audits(
        self, tenant_id: str, facility_id: str | None = None, limit: int = 100
    ) -> list[AuditEvent]:
        query = "SELECT data_json FROM audits WHERE tenant_id=%s"
        params: list[Any] = [tenant_id]
        if facility_id:
            query += " AND facility_id=%s"
            params.append(facility_id)
        query += " ORDER BY occurred_at DESC LIMIT %s"
        params.append(limit)
        with self.pool.connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [AuditEvent.model_validate(row["data_json"]) for row in rows]
