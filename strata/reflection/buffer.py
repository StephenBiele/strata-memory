"""L1.5 aggregation buffer (spec §7).

An internal review tier between atomic facts (L1) and durable scenarios (L2/L3). It clusters
related/duplicate L1 records so promotion into L2/L3 is not premature and L1 stays simple.
The MVP clustering is deterministic token-Jaccard over canonical text; cluster outputs feed
the reflection engine's consolidation proposals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from strata.canonical.records import MemoryRecord
from strata.resolver.diversity import jaccard


@dataclass
class Cluster:
    record_ids: list[int] = field(default_factory=list)
    representative: str = ""


def cluster_l1(records: list[MemoryRecord], *, threshold: float = 0.6) -> list[Cluster]:
    """Greedy single-pass clustering of related L1 records by canonical-text similarity.

    A record joins the first cluster whose representative it resembles (>= threshold);
    otherwise it seeds a new cluster. Only clusters with 2+ members are review-worthy.
    """
    clusters: list[Cluster] = []
    for rec in records:
        placed = False
        for c in clusters:
            if jaccard(rec.content, c.representative) >= threshold:
                c.record_ids.append(rec.id)
                placed = True
                break
        if not placed:
            clusters.append(Cluster(record_ids=[rec.id], representative=rec.content))
    return clusters


def review_worthy(clusters: list[Cluster]) -> list[Cluster]:
    return [c for c in clusters if len(c.record_ids) >= 2]
