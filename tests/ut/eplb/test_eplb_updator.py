import unittest
from multiprocessing import Manager, Process
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.eplb.eplb_updator import EplbUpdator


def read_moe_load_from_shared_dict(shared_dict, result_queue):
    result_queue.put(shared_dict["moe_load"])


class TestEplbUpdatorComputeAndSetMoeLoad(unittest.TestCase):
    def setUp(self):
        # ====================== 1. Mock environment ======================
        self.rank = 0
        self.world_size = 4
        self.device = torch.device("cpu")

        # mock dist
        p1 = patch("torch.distributed.get_rank", return_value=self.rank)
        p2 = patch("torch.distributed.get_world_size", return_value=self.world_size)
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        p1.start()
        p2.start()

        # ====================== 2. Mock comm group ======================
        self.mock_comm_group = MagicMock()

        def mock_all_gather(tensor, dim):
            gathered = torch.cat([tensor for _ in range(self.world_size)], dim=dim)
            return gathered

        self.mock_comm_group.all_gather = mock_all_gather

        p3 = patch("vllm_ascend.eplb.eplb_updator.get_dynamic_eplb_group", return_value=self.mock_comm_group)
        self.addCleanup(p3.stop)
        p3.start()

        # mock _PP in vllm.distributed.parallel_state (PP+EPLB support)
        # Patching the variable directly so that even the real get_pp_group()
        # (already imported into eplb_updator's namespace) reads a non-None _PP.
        self.mock_pp = MagicMock()
        self.mock_pp.rank_in_group = 0
        p4 = patch("vllm.distributed.parallel_state._PP", self.mock_pp)
        self.addCleanup(p4.stop)
        p4.start()

        # ====================== 3. Mock EplbUpdator ======================
        self.eplb_config = MagicMock()
        self.loader = MagicMock()
        self.eplb_process = MagicMock()
        self.process = MagicMock()
        self.eplb_process.shared_dict = {}

        self.updator = EplbUpdator(
            eplb_config=self.eplb_config, loader=self.loader, eplb_process=self.eplb_process, process=self.process
        )

        # ====================== 4. Mock adaptor ======================
        self.adaptor = MagicMock()
        self.adaptor.num_moe_layers = 4
        self.adaptor.num_dense_layers = 2
        self.mock_local_load = torch.randn(58, 100, 8, device=self.device)
        self.adaptor.get_rank_expert_workload.return_value = self.mock_local_load

        self.updator.set_adaptor(self.adaptor)

    def test_compute_and_set_moe_load_normal(self):
        self.updator.multi_stage = False

        moe_load = self.updator.compute_and_set_moe_load()

        self.assertEqual(moe_load.shape, (58, self.world_size, 100, 8))
        self.assertTrue("moe_load" in self.updator.shared_dict)
        self.assertEqual(moe_load.device.type, "cpu")
        self.assertEqual(moe_load.shape[1], self.world_size)

    def test_compute_and_set_moe_load_multi_stage(self):
        self.updator.multi_stage = True

        moe_load = self.updator.compute_and_set_moe_load()

        self.assertEqual(moe_load.shape, (100, 58, self.world_size, 8))
        self.assertTrue("moe_load" in self.updator.shared_dict)
        self.assertEqual(moe_load.device.type, "cpu")

    def test_compute_and_set_moe_load_reuses_cpu_buffer(self):
        self.updator.multi_stage = False
        self.adaptor.get_rank_expert_workload.return_value = torch.ones(
            58, 100, 8, device=self.device
        )
        first_moe_load = self.updator.compute_and_set_moe_load()

        self.adaptor.get_rank_expert_workload.return_value = torch.full(
            (58, 100, 8),
            2,
            device=self.device,
        )
        second_moe_load = self.updator.compute_and_set_moe_load()

        self.assertEqual(
            self.updator.shared_dict["moe_load"].data_ptr(),
            second_moe_load.data_ptr(),
        )
        self.assertTrue(torch.equal(second_moe_load, torch.full_like(second_moe_load, 2)))

    def test_compute_and_set_moe_load_visible_through_manager_dict(self):
        self.updator.multi_stage = False

        with Manager() as manager:
            self.updator.shared_dict = manager.dict()

            self.adaptor.get_rank_expert_workload.return_value = torch.ones(
                2,
                3,
                1,
                device=self.device,
            )
            first_moe_load = self.updator.compute_and_set_moe_load()
            first_visible = self._read_moe_load_from_process(manager)

            self.adaptor.get_rank_expert_workload.return_value = torch.full(
                (2, 3, 1),
                2,
                device=self.device,
            )
            second_moe_load = self.updator.compute_and_set_moe_load()
            second_visible = self._read_moe_load_from_process(manager)

        self.assertEqual(first_visible.shape, (2, self.world_size, 3, 1))
        self.assertEqual(second_visible.shape, first_visible.shape)
        self.assertTrue(torch.equal(first_visible, torch.ones_like(first_visible)))
        self.assertTrue(torch.equal(second_visible, torch.full_like(second_visible, 2)))

    def _read_moe_load_from_process(self, manager):
        result_queue = manager.Queue()
        process = Process(
            target=read_moe_load_from_shared_dict,
            args=(self.updator.shared_dict, result_queue),
        )
        process.start()
        process.join(timeout=10)

        if process.is_alive():
            process.terminate()
            process.join()
            self.fail("Timed out reading moe_load from Manager dict")

        self.assertEqual(process.exitcode, 0)
        return result_queue.get(timeout=10)


if __name__ == "__main__":
    unittest.main()
