"""Minimal sensitivity / permission policy layer (spec §13 MVP enforcement floor).

This is intentionally small — a hard-coded set of sensitivity classes, a recall permission
filter, and a reflection restriction. It is the floor the deletion, privacy, and local-trust
guarantees depend on, so it ships in MVP even though the full configurable policy engine
(DSL-driven) is post-MVP.
"""

from __future__ import annotations

from typing import Callable

from strata.canonical.records import MemoryRecord, Sensitivity

# Ordered least → most restricted.
_ORDER = {
    Sensitivity.NORMAL: 0,
    Sensitivity.PERSONAL: 1,
    Sensitivity.SENSITIVE: 2,
    Sensitivity.SECRET: 3,
}


class SensitivityPolicy:
    def __init__(
        self,
        *,
        recall_max: Sensitivity = Sensitivity.PERSONAL,
        reflection_max: Sensitivity = Sensitivity.PERSONAL,
    ) -> None:
        # Enforcement floor: by default SENSITIVE/SECRET are NOT surfaced unless the host scope
        # explicitly raises recall_max. Reflection may consider up to reflection_max.
        self.recall_max = recall_max
        self.reflection_max = reflection_max

    def can_recall(self, record: MemoryRecord) -> bool:
        return _ORDER[record.sensitivity] <= _ORDER[self.recall_max]

    def can_reflect_on(self, record: MemoryRecord) -> bool:
        """Reflection must not consider records above the reflection ceiling (e.g. SECRET
        material never feeds background inference by default)."""
        return _ORDER[record.sensitivity] <= _ORDER[self.reflection_max]

    def recall_permit(self) -> Callable[[MemoryRecord], bool]:
        """Permit callback for the resolver's permission filter."""
        return self.can_recall
