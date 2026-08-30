import json
import os
import re
import sqlite3
import uuid
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional


class AuditLogger:
    """SQLite-backed audit logger for ControlPlane requests."""

    SECRET_PATTERNS = [
        r"(?i)\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[bapors]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|AIza[0-9A-Za-z\-_]{20,}|sk_(?:live|test)_[A-Za-z0-9]{16,})\b",
        r"(?i)(?:api[_ -]?key|secret[_ -]?key|access[_ -]?key[_ -]?id|secret[_ -]?access[_ -]?key|token|password|private[_ -]?key|bearer)[\s:=]+['\"]?[A-Za-z0-9_\-./=+:%#@]{8,}['\"]?",
        r"(?i)-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----",
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b(?:\d[ -]*){13,16}\b",
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        r"(?<!\d)(?:\+\d{1,3}[\s.-]?)?(?:\(\?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)",
    ]

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "controlplane_audit.db"
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    use_case TEXT,
                    user_prompt TEXT,
                    llm_response TEXT,
                    source TEXT,
                    destination TEXT,
                    trust_boundary_crossed INTEGER,
                    sensitivity TEXT,
                    categories TEXT,
                    security_score REAL,
                    security_status TEXT,
                    security_decision TEXT,
                    security_findings TEXT,
                    matched_policies TEXT,
                    policy_source TEXT,
                    performance_score REAL,
                    performance_status TEXT,
                    cost_score REAL,
                    cost_status TEXT,
                    estimated_cost REAL,
                    unified_risk_score REAL,
                    final_action TEXT,
                    preflight_risk_score REAL,
                    expected_action TEXT,
                    ground_truth TEXT,
                    evaluation_result TEXT,
                    latency_ms REAL,
                    audit_record_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()

    @staticmethod
    def _ensure_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    def _redact_text(self, value: Any) -> str:
        text = self._ensure_text(value)
        if not text:
            return ""

        redacted = text
        for pattern in self.SECRET_PATTERNS:
            redacted = re.sub(pattern, "[REDACTED]", redacted)

        if "API key" in redacted or "password" in redacted.lower() or "bearer" in redacted.lower():
            redacted = re.sub(r"(?i)(api[_ -]?key|password|bearer|token|secret|private[_ -]?key)\s*[:=]?\s*[\w\-./=+:@%#\{\}\[\]\(\)\'\"]{4,}", r"\1=[REDACTED]", redacted)

        return redacted

    @staticmethod
    def _safe_json(value: Any) -> str:
        if value is None:
            return "[]"
        try:
            return json.dumps(value)
        except TypeError:
            return json.dumps(str(value))

    def _sanitize_record_for_storage(self, record: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = deepcopy(record)

        for field in ["user_prompt", "llm_response", "final_response"]:
            if field in sanitized:
                sanitized[field] = self._redact_text(sanitized[field])

        for field in ["categories", "security_findings", "matched_policies", "pattern_findings", "audit_log"]:
            if field in sanitized and sanitized[field] is not None:
                sanitized[field] = json.loads(json.dumps(sanitized[field], default=str))

        return sanitized

    def _row_to_record(self, row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        for key in ["categories", "security_findings", "matched_policies", "audit_log"]:
            value = result.get(key)
            if isinstance(value, str):
                try:
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    result[key] = value

        if "user_prompt" in result:
            result["user_prompt"] = self._redact_text(result["user_prompt"])
        if "llm_response" in result:
            result["llm_response"] = self._redact_text(result["llm_response"])
        return result

    def log_request(self, record: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = self._sanitize_record_for_storage(record)
        request_id = cleaned.get("request_id") or str(uuid.uuid4())
        cleaned["request_id"] = request_id

        if cleaned.get("audit_record_id") is None:
            cleaned["audit_record_id"] = clean = f"audit_{request_id}"

        def json_field(value: Any) -> Optional[str]:
            if value is None:
                return None
            return json.dumps(value, default=str)

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO audit_records (
                    request_id,
                    timestamp,
                    user_id,
                    use_case,
                    user_prompt,
                    llm_response,
                    source,
                    destination,
                    trust_boundary_crossed,
                    sensitivity,
                    categories,
                    security_score,
                    security_status,
                    security_decision,
                    security_findings,
                    matched_policies,
                    policy_source,
                    performance_score,
                    performance_status,
                    cost_score,
                    cost_status,
                    estimated_cost,
                    unified_risk_score,
                    final_action,
                    preflight_risk_score,
                    expected_action,
                    ground_truth,
                    evaluation_result,
                    latency_ms,
                    audit_record_id
                ) VALUES (
                    :request_id,
                    datetime('now'),
                    :user_id,
                    :use_case,
                    :user_prompt,
                    :llm_response,
                    :source,
                    :destination,
                    :trust_boundary_crossed,
                    :sensitivity,
                    :categories,
                    :security_score,
                    :security_status,
                    :security_decision,
                    :security_findings,
                    :matched_policies,
                    :policy_source,
                    :performance_score,
                    :performance_status,
                    :cost_score,
                    :cost_status,
                    :estimated_cost,
                    :unified_risk_score,
                    :final_action,
                    :preflight_risk_score,
                    :expected_action,
                    :ground_truth,
                    :evaluation_result,
                    :latency_ms,
                    :audit_record_id
                )
                ON CONFLICT(request_id)
                DO UPDATE SET
                    timestamp = excluded.timestamp,
                    user_id = excluded.user_id,
                    use_case = excluded.use_case,
                    user_prompt = excluded.user_prompt,
                    llm_response = excluded.llm_response,
                    source = excluded.source,
                    destination = excluded.destination,
                    trust_boundary_crossed = excluded.trust_boundary_crossed,
                    sensitivity = excluded.sensitivity,
                    categories = excluded.categories,
                    security_score = excluded.security_score,
                    security_status = excluded.security_status,
                    security_decision = excluded.security_decision,
                    security_findings = excluded.security_findings,
                    matched_policies = excluded.matched_policies,
                    policy_source = excluded.policy_source,
                    performance_score = excluded.performance_score,
                    performance_status = excluded.performance_status,
                    cost_score = excluded.cost_score,
                    cost_status = excluded.cost_status,
                    estimated_cost = excluded.estimated_cost,
                    unified_risk_score = excluded.unified_risk_score,
                    final_action = excluded.final_action,
                    preflight_risk_score = excluded.preflight_risk_score,
                    expected_action = excluded.expected_action,
                    ground_truth = excluded.ground_truth,
                    evaluation_result = excluded.evaluation_result,
                    latency_ms = excluded.latency_ms,
                    audit_record_id = excluded.audit_record_id
                """,
                {
                    "request_id": request_id,
                    "user_id": cleaned.get("user_id"),
                    "use_case": cleaned.get("use_case"),
                    "user_prompt": self._redact_text(cleaned.get("user_prompt")),
                    "llm_response": self._redact_text(cleaned.get("llm_response")),
                    "source": cleaned.get("source"),
                    "destination": cleaned.get("destination"),
                    "trust_boundary_crossed": 1 if cleaned.get("trust_boundary_crossed") else 0,
                    "sensitivity": cleaned.get("sensitivity"),
                    "categories": json_field(cleaned.get("categories")),
                    "security_score": cleaned.get("security_score"),
                    "security_status": cleaned.get("security_status"),
                    "security_decision": cleaned.get("security_decision"),
                    "security_findings": json_field(cleaned.get("security_findings")),
                    "matched_policies": json_field(cleaned.get("matched_policies")),
                    "policy_source": cleaned.get("policy_source"),
                    "performance_score": cleaned.get("performance_score"),
                    "performance_status": cleaned.get("performance_status"),
                    "cost_score": cleaned.get("cost_score"),
                    "cost_status": cleaned.get("cost_status"),
                    "estimated_cost": cleaned.get("estimated_cost"),
                    "unified_risk_score": cleaned.get("unified_risk_score"),
                    "final_action": cleaned.get("final_action"),
                    "preflight_risk_score": cleaned.get("preflight_risk_score"),
                    "expected_action": cleaned.get("expected_action"),
                    "ground_truth": cleaned.get("ground_truth"),
                    "evaluation_result": cleaned.get("evaluation_result"),
                    "latency_ms": cleaned.get("latency_ms"),
                    "audit_record_id": cleaned.get("audit_record_id"),
                },
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM audit_records WHERE request_id = ?",
                (request_id,),
            ).fetchone()

        return self._row_to_record(row)

    def get_recent_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_records ORDER BY timestamp DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def get_decision_distribution(self) -> Dict[str, int]:
        records = self.get_recent_records(limit=100000)
        distribution = {"ALLOW": 0, "BLOCK": 0, "ESCALATE": 0, "REWRITE": 0}
        for row in records:
            action = str(row.get("final_action") or "").upper()
            if action in distribution:
                distribution[action] += 1
        return distribution

    def get_score_summary(self) -> Dict[str, float]:
        records = self.get_recent_records(limit=100000)
        if not records:
            return {
                "average_security_score": 0.0,
                "average_performance_score": 0.0,
                "average_cost_score": 0.0,
                "average_unified_risk_score": 0.0,
            }

        def avg(field: str) -> float:
            values = [float(row.get(field) or 0.0) for row in records if row.get(field) is not None]
            if not values:
                return 0.0
            return sum(values) / len(values)

        return {
            "average_security_score": avg("security_score"),
            "average_performance_score": avg("performance_score"),
            "average_cost_score": avg("cost_score"),
            "average_unified_risk_score": avg("unified_risk_score"),
        }

    def get_security_metrics(self) -> Dict[str, float]:
        records = self.get_recent_records(limit=100000)
        total = len(records)

        if not records:
            return {
                "total_requests": 0,
                "evaluated_requests": 0,
                "allowed": 0,
                "blocked": 0,
                "escalated": 0,
                "rewritten": 0,
                "true_positives": 0,
                "true_negatives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
            }

        distribution = self.get_decision_distribution()
        evaluated = sum(
            1
            for row in records
            if (row.get("ground_truth") is not None or row.get("expected_action") is not None or row.get("evaluation_result") is not None)
        )

        tp = tn = fp = fn = 0
        for row in records:
            ground_truth = str(row.get("ground_truth") or row.get("expected_action") or "").upper()
            decision = str(row.get("security_decision") or row.get("final_action") or "").upper()
            if ground_truth not in {"ALLOW", "BLOCK"} or decision not in {"ALLOW", "BLOCK"}:
                continue

            if ground_truth == "BLOCK" and decision == "BLOCK":
                tp += 1
            elif ground_truth == "ALLOW" and decision == "ALLOW":
                tn += 1
            elif ground_truth == "ALLOW" and decision == "BLOCK":
                fp += 1
            elif ground_truth == "BLOCK" and decision == "ALLOW":
                fn += 1

        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

        return {
            "total_requests": total,
            "evaluated_requests": evaluated,
            "allowed": distribution.get("ALLOW", 0),
            "blocked": distribution.get("BLOCK", 0),
            "escalated": distribution.get("ESCALATE", 0),
            "rewritten": distribution.get("REWRITE", 0),
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
        }

    def get_metrics(self) -> Dict[str, Any]:
        security = self.get_security_metrics()
        distribution = self.get_decision_distribution()
        scores = self.get_score_summary()
        metrics = {
            "total_requests": security["total_requests"],
            "evaluated_requests": security["evaluated_requests"],
            "allowed": distribution.get("ALLOW", 0),
            "blocked": distribution.get("BLOCK", 0),
            "escalated": distribution.get("ESCALATE", 0),
            "rewritten": distribution.get("REWRITE", 0),
            "average_security_score": scores["average_security_score"],
            "average_performance_score": scores["average_performance_score"],
            "average_cost_score": scores["average_cost_score"],
            "average_unified_risk_score": scores["average_unified_risk_score"],
            "true_positives": security["true_positives"],
            "true_negatives": security["true_negatives"],
            "false_positives": security["false_positives"],
            "false_negatives": security["false_negatives"],
            "false_positive_rate": security["false_positive_rate"],
            "false_negative_rate": security["false_negative_rate"],
            "precision": security["precision"],
            "recall": security["recall"],
            "accuracy": security["accuracy"],
        }
        return metrics

    def export_csv(self, output_path: str, limit: int = 10000) -> str:
        import csv

        records = self.get_recent_records(limit=limit)
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "request_id","timestamp","user_id","use_case","source","destination","trust_boundary_crossed",
                "sensitivity","categories","security_score","security_status","security_decision","performance_score",
                "performance_status","cost_score","cost_status","estimated_cost","unified_risk_score","final_action",
                "expected_action","ground_truth","evaluation_result","latency_ms"
            ])
            writer.writeheader()
            for row in records:
                writer.writerow({
                    "request_id": row.get("request_id"),
                    "timestamp": row.get("timestamp"),
                    "user_id": row.get("user_id"),
                    "use_case": row.get("use_case"),
                    "source": row.get("source"),
                    "destination": row.get("destination"),
                    "trust_boundary_crossed": row.get("trust_boundary_crossed"),
                    "sensitivity": row.get("sensitivity"),
                    "categories": json.dumps(row.get("categories") or []),
                    "security_score": row.get("security_score"),
                    "security_status": row.get("security_status"),
                    "security_decision": row.get("security_decision"),
                    "performance_score": row.get("performance_score"),
                    "performance_status": row.get("performance_status"),
                    "cost_score": row.get("cost_score"),
                    "cost_status": row.get("cost_status"),
                    "estimated_cost": row.get("estimated_cost"),
                    "unified_risk_score": row.get("unified_risk_score"),
                    "final_action": row.get("final_action"),
                    "expected_action": row.get("expected_action"),
                    "ground_truth": row.get("ground_truth"),
                    "evaluation_result": row.get("evaluation_result"),
                    "latency_ms": row.get("latency_ms"),
                })
        return output_path
