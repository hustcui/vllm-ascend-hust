import unittest
from unittest.mock import patch

import torch

from vllm_ascend.eplb.core.eplb_worker import EplbWorker


class TestEplbWorkerUpdatePlanning(unittest.TestCase):
    def _build_worker(self):
        worker = EplbWorker.__new__(EplbWorker)
        worker.rank_id = 0
        return worker

    def test_compose_update_info_uses_precomputed_expert_sources(self):
        worker = self._build_worker()
        current_expert_maps = torch.tensor(
            [
                [
                    [0, 1, -1, -1],
                    [-1, 0, 1, -1],
                    [-1, -1, -1, 0],
                ]
            ],
            dtype=torch.int64,
        )
        updated_expert_maps = torch.tensor(
            [
                [
                    [0, -1, -1, 1],
                    [-1, 0, 1, -1],
                    [-1, -1, 2, -1],
                ]
            ],
            dtype=torch.int64,
        )

        update_info = list(
            worker.compose_expert_update_info_greedy(
                updated_expert_maps,
                current_expert_maps,
            )
        )

        send_info, recv_info, new_expert_map, layer_id = update_info[0]
        self.assertEqual(layer_id, 0)
        self.assertTrue(torch.equal(new_expert_map, updated_expert_maps[0]))
        self.assertEqual(send_info, {1: [(2, 2)], 2: [(0, 3)]})
        self.assertEqual(recv_info, {0: [(2, 3)], 2: [(1, 2)]})

    def test_compose_update_info_skips_extra_work_when_layer_unchanged(self):
        worker = self._build_worker()
        current_expert_maps = torch.tensor(
            [
                [
                    [0, 1],
                    [1, 0],
                ]
            ],
            dtype=torch.int64,
        )

        update_info = list(
            worker.compose_expert_update_info_greedy(
                current_expert_maps,
                current_expert_maps,
            )
        )

        self.assertEqual(len(update_info), 1)
        send_info, recv_info, new_expert_map, layer_id = update_info[0]
        self.assertEqual(layer_id, 0)
        self.assertEqual(send_info, {})
        self.assertEqual(recv_info, {})
        self.assertTrue(torch.equal(new_expert_map, current_expert_maps[0]))

    @patch("vllm_ascend.eplb.core.eplb_worker.generate_log2phy_map")
    def test_pack_update_info_batches_tensor_to_list_conversion(
        self, mock_generate_log2phy_map
    ):
        worker = self._build_worker()
        mock_generate_log2phy_map.side_effect = [
            torch.tensor([3, 4, 5], dtype=torch.int32),
            torch.tensor([6, 7, 8], dtype=torch.int32),
        ]
        update_info = [
            (
                {0: [(1, 2)]},
                {0: [(2, 1)]},
                torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32),
                0,
            ),
            (
                {0: [(1, 3)]},
                {0: [(2, 4)]},
                torch.tensor([[13, 14, 15], [23, 24, 25]], dtype=torch.int32),
                1,
            ),
        ]

        packed = worker.pack_update_info(iter(update_info))

        self.assertEqual(
            packed,
            [
                ([(1, 2)], [(2, 1)], [10, 11, 12], [3, 4, 5], 0),
                ([(1, 3)], [(2, 4)], [13, 14, 15], [6, 7, 8], 1),
            ],
        )
        self.assertEqual(mock_generate_log2phy_map.call_count, 2)


if __name__ == "__main__":
    unittest.main()
