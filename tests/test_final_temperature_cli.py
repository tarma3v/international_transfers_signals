import json

import pytest

from run_final_temperature import build_payload, main


def test_final_cli_builds_all_corridors_and_horizons():
    payload = build_payload("2025-05-15T15:45:00+03:00")
    assert len(payload) == 5
    assert {row["corridor"] for row in payload} == {
        "AMD", "KGS", "KZT", "TJS", "UZS",
    }
    assert all(set(row["horizon_outputs"]) == {"1", "3", "5", "10", "20"}
               for row in payload)


def test_final_cli_rejects_timezone_free_query():
    with pytest.raises(ValueError, match="timezone"):
        build_payload("2025-05-15T15:45:00")


def test_final_cli_verify_only_is_machine_readable(capsys):
    main(["--verify-only"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifact_hashes_verified"] is True
    assert payload["status"] == "frozen_hackathon_candidate"
