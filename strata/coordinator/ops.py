"""Write Coordinator operation types (spec §12)."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

from strata.ids import new_operation_id


class OpType(str, enum.Enum):
    UPSERT_INDEX = "upsert_index"
    REMOVE_INDEX = "remove_index"
    TOMBSTONE = "tombstone"
    REINDEX_PROMOTE = "reindex_promote"
    HARD_PURGE = "hard_purge"


# Destructive ops are globally ordered: they act as barriers in the op log (spec §12
# "global ordering for destructive operations such as deletion, reindex promotion, hard purge").
DESTRUCTIVE = frozenset({
    OpType.REMOVE_INDEX,
    OpType.TOMBSTONE,
    OpType.REINDEX_PROMOTE,
    OpType.HARD_PURGE,
})


class OpState(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PARTIAL_FAILURE = "partial_failure"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Operation:
    op_type: OpType
    target_ids: list[int]
    target: Optional[str] = None          # handler/adapter name
    payload: Optional[dict] = None
    expected_generation: Optional[int] = None
    op_id: str = field(default_factory=new_operation_id)

    @property
    def is_destructive(self) -> bool:
        return self.op_type in DESTRUCTIVE
