"""Frozen, versioned entry point for the final hackathon temperature model."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ml.transfer_temperature import case_output_as_of, receipt_gated_snapshots


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "model/final_temperature_model_v1.json"
REQUIRED_HORIZON_FIELDS = (
    "probability_now_best_h", "temperature_0_100",
    "expected_future_cbr_bps_h", "last_source_at", "freshness", "phase",
    "confidence", "source_kind", "availability_evidence", "push_now",
)


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict:
    return json.loads(Path(path).read_text())


def verify_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict:
    manifest = load_manifest(path)
    expected = {
        "status": "frozen_hackathon_candidate",
        "runtime_mode": "historical_replay_artifact",
        "horizons": [1, 3, 5, 10, 20],
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"unexpected manifest {field}: {manifest.get(field)!r}")
    if sorted(manifest.get("corridors", [])) != [
            "AMD", "KGS", "KZT", "TJS", "UZS"]:
        raise ValueError("unexpected manifest corridors")
    if manifest.get("target", {}).get("name") != "future_only_send_now":
        raise ValueError("unexpected manifest target")
    if manifest.get("availability_contract", {}).get(
            "fixed_1800_boundary_forbidden") is not True:
        raise ValueError("fixed 18:00 boundary must remain forbidden")
    if "direct_pair_probability" not in manifest.get(
            "rejected_components", {}):
        raise ValueError("rejected direct-pair probability is not recorded")
    roles = set()
    for artifact in manifest["artifacts"]:
        artifact_path = ROOT / artifact["path"]
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if digest != artifact["sha256"]:
            raise ValueError(f"artifact hash mismatch: {artifact['path']}")
        if artifact["role"] in roles:
            raise ValueError(f"duplicate artifact role: {artifact['role']}")
        roles.add(artifact["role"])
    required = {
        "active_runtime_snapshots", "active_runtime_audit",
        "active_push_evidence", "active_push_audit",
        "after_receipt_probability_evidence", "after_receipt_probability_audit",
    }
    if not required <= roles:
        raise ValueError(f"missing active artifacts: {sorted(required - roles)}")
    return manifest


@lru_cache(maxsize=1)
def _frozen_snapshots() -> pd.DataFrame:
    manifest = verify_manifest()
    artifact = next(
        item for item in manifest["artifacts"]
        if item["role"] == "active_runtime_snapshots"
    )
    return pd.read_csv(ROOT / artifact["path"])


def load_frozen_snapshots() -> pd.DataFrame:
    """Return an isolated copy so callers cannot mutate the cached artifact."""
    return _frozen_snapshots().copy()


def _decorate_final(result, manifest):
    if result is None:
        return None
    missing = [field for field in REQUIRED_HORIZON_FIELDS if field not in result]
    if missing:
        raise ValueError(f"runtime output missing fields: {missing}")
    return {
        **result,
        "model_version": manifest["model_version"],
        "model_status": manifest["status"],
        "target_name": manifest["target"]["name"],
        # These are deliberately not silently promoted by this entry point.
        "shadow_components_active": False,
    }


def _score_prepared(frame, manifest, currency, as_of, horizon):
    result = case_output_as_of(frame, currency, as_of, horizon)
    return _decorate_final(result, manifest)


def score_final_as_of(currency, as_of, horizon=5, verified_receipt_at=None,
                      snapshots=None):
    """Return one selected horizon from the frozen production-style router."""
    manifest = load_manifest()
    if horizon not in manifest["horizons"]:
        raise ValueError(f"unsupported horizon: {horizon}")
    if currency not in manifest["corridors"]:
        raise ValueError(f"unsupported currency: {currency}")
    frame = load_frozen_snapshots() if snapshots is None else snapshots
    prepared = receipt_gated_snapshots(
        frame, as_of, verified_receipt_at=verified_receipt_at,
    )
    return _score_prepared(prepared, manifest, currency, as_of, horizon)


def score_final_all_horizons_as_of(currency, as_of, selected_horizon=5,
                                   verified_receipt_at=None, snapshots=None):
    """Return the selected view plus auditable outputs for all five horizons."""
    manifest = load_manifest()
    if selected_horizon not in manifest["horizons"]:
        raise ValueError(f"unsupported horizon: {selected_horizon}")
    if currency not in manifest["corridors"]:
        raise ValueError(f"unsupported currency: {currency}")
    frame = load_frozen_snapshots() if snapshots is None else snapshots
    prepared = receipt_gated_snapshots(
        frame, as_of, verified_receipt_at=verified_receipt_at,
    )
    return _all_horizons_prepared(
        prepared, manifest, currency, as_of, selected_horizon)


def score_final_table_as_of(as_of, selected_horizon=5,
                            verified_receipt_at=None, snapshots=None):
    """Return one all-horizon final-model row for every configured corridor."""
    manifest = load_manifest()
    if selected_horizon not in manifest["horizons"]:
        raise ValueError(f"unsupported horizon: {selected_horizon}")
    frame = load_frozen_snapshots() if snapshots is None else snapshots
    prepared = receipt_gated_snapshots(
        frame, as_of, verified_receipt_at=verified_receipt_at,
    )
    return [
        _all_horizons_prepared(
            prepared, manifest, currency, as_of, selected_horizon)
        for currency in manifest["corridors"]
    ]


def _all_horizons_prepared(frame, manifest, currency, as_of, selected_horizon):
    by_horizon = {
        str(h): _score_prepared(frame, manifest, currency, as_of, h)
        for h in manifest["horizons"]
    }
    selected = by_horizon[str(selected_horizon)]
    if selected is None:
        return None
    available = {
        key: value for key, value in by_horizon.items() if value is not None
    }
    selected = dict(selected)
    selected["horizon_outputs"] = {
        key: {field: value[field] for field in REQUIRED_HORIZON_FIELDS}
        for key, value in available.items()
    }
    return selected
