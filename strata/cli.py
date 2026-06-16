"""CLI / SDK example for companion and generic agent usage (spec §13 MVP).

Run ``python -m strata.cli demo`` for an end-to-end walkthrough: write a fact, correct it,
delete it (with verification), run reflection, and print a belief bundle — asserting that the
deleted content is unrecoverable through the public recall API.
"""

from __future__ import annotations

import argparse
import json
import sys

from strata.gateway.api import Strata


def demo() -> int:
    s = Strata.open()
    print("# write durable preferences")
    s.write_memory("user prefers tea for work")
    s.write_memory("user prefers coffee on weekends")
    secret = s.write_memory("user's home address is 12 Elm St", sensitivity="secret")

    print("# recall (secret is filtered by the policy floor)")
    bundle = s.recall("user prefers")
    print(json.dumps(bundle["current_beliefs"], indent=2))
    assert all("Elm St" not in e["claim"] for e in bundle["current_beliefs"])

    print("\n# correction: tea -> matcha")
    # Select by claim text: the deterministic hash embedder has no semantics, so similarity
    # rank is not reliable for picking a specific record in a demo.
    tea = next(e for e in s.recall("prefers tea")["current_beliefs"] if "tea" in e["claim"])
    s.supersede_memory(tea["id"], "user prefers matcha for work")
    after = {e["claim"] for e in s.recall("prefers")["current_beliefs"]}
    print(after)
    assert "user prefers matcha for work" in after
    assert "user prefers tea for work" not in after

    print("\n# right to forget: hard-delete the secret, then verify")
    job = s.delete_memory(secret["id"], mode="hard")["job_id"]
    status = s.deletion_status(job)
    print("deletion status:", status["state"])
    assert status["state"] == "verified"
    # The deleted secret is unrecoverable; the hash embedder may surface unrelated records,
    # so assert the secret specifically is gone rather than that the bundle is empty.
    surfaced = s.recall("home address")["current_beliefs"]
    assert all(e["id"] != secret["id"] and "Elm St" not in e["claim"] for e in surfaced)

    print("\n# reflection: consolidate duplicates")
    s.write_memory("user prefers matcha for work")  # duplicate of the correction
    result = s.run_reflection("consolidate")
    print("reflection proposals:", result["count"])

    print("\nOK: deletion verified, correction applied, secret unrecoverable.")
    s.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="strata")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="run the end-to-end walkthrough")
    args = parser.parse_args(argv)
    if args.cmd == "demo" or args.cmd is None:
        return demo()
    return 1


if __name__ == "__main__":
    sys.exit(main())
