#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Scheduler interaction for Sim-LLM batch deferral (Phase 3+).

When the match ratio in a batch exceeds ``deferral_ratio``, unmatched
tasks are re-queued via the vLLM scheduler so they can be batched with
more similar tasks in a subsequent scheduling cycle.  Each task can be
deferred at most ``max_deferrals`` times before being force-processed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def defer_unmatched_tasks(
    deferral_indices: set[int],
    req_ids: list[str],
    scheduler: Any | None = None,
    deferral_counts: dict[str, int] | None = None,
    max_deferrals: int = 3,
) -> dict[str, int]:
    """Flag unmatched tasks for re-queue to the scheduler.

    In Phase 3 this will call ``scheduler.add_request()`` to re-submit
    deferred tasks with incremented deferral counts.  For Phase 2 it is
    a logging-only stub.

    Parameters
    ----------
    deferral_indices:
        Batch indices of tasks to defer.
    req_ids:
        Request IDs for the batch (ordered).
    scheduler:
        vLLM ``Scheduler`` instance (available in Phase 3).
    deferral_counts:
        Per-request deferral count dict; updated in-place if provided.
    max_deferrals:
        Maximum deferral count before force-processing.

    Returns
    -------
    Updated *deferral_counts* dict (same object if passed in).
    """
    if deferral_counts is None:
        deferral_counts = {}

    deferred = 0
    for idx in deferral_indices:
        if idx >= len(req_ids):
            continue
        req_id = req_ids[idx]
        cnt = deferral_counts.get(req_id, 0)
        if cnt >= max_deferrals:
            logger.debug(
                "SimLLM: req %s reached max deferrals (%d), force-processing.",
                req_id, cnt,
            )
            continue
        deferral_counts[req_id] = cnt + 1
        deferred += 1

        # Phase 3: scheduler.add_request(deferred_request)
        # The deferred request is re-created with the same prompt_token_ids
        # and an incremented deferral_count in its metadata.

    if deferred:
        logger.debug(
            "SimLLM scheduler: flagged %d tasks for deferral.", deferred,
        )

    return deferral_counts
