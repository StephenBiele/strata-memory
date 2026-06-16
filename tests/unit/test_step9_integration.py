"""Step 9 suite: policy, L1.5 buffer, reflection, gateway, HTTP, CLI."""

import json
import urllib.request

import pytest

from strata.canonical.records import MemoryRecord, Relation, Sensitivity, Status, Tier
from strata.gateway.api import Strata
from strata.policy.sensitivity import SensitivityPolicy
from strata.reflection.buffer import cluster_l1, review_worthy
from strata.reflection.proposals import ProposalKind, ProposalState


# -- policy --------------------------------------------------------------------
def test_policy_filters_sensitive_from_recall():
    pol = SensitivityPolicy(recall_max=Sensitivity.PERSONAL)
    normal = MemoryRecord.create("ok", sensitivity=Sensitivity.NORMAL)
    secret = MemoryRecord.create("nope", sensitivity=Sensitivity.SECRET)
    assert pol.can_recall(normal)
    assert not pol.can_recall(secret)


def test_policy_reflection_ceiling():
    pol = SensitivityPolicy(reflection_max=Sensitivity.NORMAL)
    assert pol.can_reflect_on(MemoryRecord.create("x", sensitivity=Sensitivity.NORMAL))
    assert not pol.can_reflect_on(MemoryRecord.create("y", sensitivity=Sensitivity.SENSITIVE))


# -- L1.5 buffer ---------------------------------------------------------------
def test_l1_clustering_groups_near_duplicates():
    recs = [
        MemoryRecord.create("user loves hiking in the mountains"),
        MemoryRecord.create("user loves hiking in the mountains a lot"),
        MemoryRecord.create("user dislikes loud restaurants"),
    ]
    clusters = review_worthy(cluster_l1(recs, threshold=0.5))
    assert len(clusters) == 1
    assert len(clusters[0].record_ids) == 2


# -- gateway facade ------------------------------------------------------------
@pytest.fixture
def strata():
    s = Strata.open()
    yield s
    s.close()


def test_secret_filtered_by_policy_floor(strata):
    strata.write_memory("public preference is tea")
    strata.write_memory("home address is 12 Elm St", sensitivity="secret")
    claims = {e["claim"] for e in strata.recall("address preference")["current_beliefs"]}
    assert not any("Elm St" in c for c in claims)


def test_correction_and_deletion_via_gateway(strata):
    rec = strata.write_memory("favorite is red")
    strata.supersede_memory(rec["id"], "favorite is blue")
    assert {e["claim"] for e in strata.recall("favorite")["current_beliefs"]} == {"favorite is blue"}
    new_id = strata.recall("favorite")["current_beliefs"][0]["id"]
    job = strata.delete_memory(new_id, mode="hard")["job_id"]
    assert strata.deletion_status(job)["state"] == "verified"
    assert strata.recall("favorite")["current_beliefs"] == []


def test_explain_memory_returns_provenance(strata):
    a = strata.write_memory("base fact")
    b = strata.write_memory("derived fact")
    strata.store.add_dependency(a["id"], b["id"], Relation.SOURCE_OF)
    info = strata.explain_memory(a["id"])
    assert info["record"]["id"] == a["id"]
    assert b["id"] in info["related"]


def test_reflection_consolidates_duplicates(strata):
    strata.write_memory("user enjoys jazz music")
    strata.write_memory("user enjoys jazz music")  # exact duplicate
    strata.write_memory("user enjoys jazz music a lot")  # near duplicate
    result = strata.run_reflection("consolidate")
    assert result["count"] >= 1
    # After consolidation, recall surfaces a single current belief for the cluster.
    current = [e for e in strata.recall("jazz music")["current_beliefs"]]
    assert len(current) == 1


def test_reflection_never_resurrects_deleted(strata):
    rec = strata.write_memory("sensitive duplicate one")
    strata.write_memory("sensitive duplicate one")
    job = strata.delete_memory(rec["id"])["job_id"]
    assert strata.deletion_status(job)["verified"]
    strata.run_reflection("consolidate")
    # The tombstoned record stays gone regardless of reflection.
    assert strata.store.is_tombstoned(rec["id"])
    assert all(e["id"] != rec["id"] for e in strata.recall("sensitive duplicate")["current_beliefs"])


def test_contradiction_audit_flags_unresolved(strata):
    a = strata.write_memory("meeting is monday")
    b = strata.write_memory("meeting is tuesday")
    strata.store.add_dependency(a["id"], b["id"], Relation.CONTRADICTS)
    result = strata.run_reflection("contradiction_audit")
    assert result["count"] == 1
    prop = strata.reflection.proposals.list()[0]
    assert prop.kind is ProposalKind.CLARIFY_CONFLICT
    assert prop.state is ProposalState.USER_REVIEW_REQUIRED


# -- HTTP stub -----------------------------------------------------------------
def test_http_endpoints_roundtrip():
    from strata.gateway.http import StrataHTTPServer

    s = Strata.open()
    server = StrataHTTPServer(s, port=0)
    server.start()
    try:
        host, port = server.address
        base = f"http://{host}:{port}"

        def post(path, payload):
            req = urllib.request.Request(
                base + path, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())

        post("/write_memory", {"content": "remote fact about sailing"})
        bundle = post("/recall", {"query": "sailing"})
        assert any("sailing" in e["claim"] for e in bundle["current_beliefs"])
    finally:
        server.stop()
        s.close()


# -- CLI demo ------------------------------------------------------------------
def test_cli_demo_runs():
    from strata.cli import main
    assert main(["demo"]) == 0
