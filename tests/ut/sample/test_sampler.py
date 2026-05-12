from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample import sampler as sampler_module
from vllm_ascend.sample.sampler import AscendSampler, AscendTopKTopPSampler


class TestAscendSampler(TestBase):
    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, "topk_topp_sampler"))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)

    def test_apply_top_k_top_p_falls_back_without_custom_op(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        k = torch.tensor([2], dtype=torch.int32)

        expected = sampler_module._apply_top_k_top_p_pytorch(logits.clone(), k, None)

        with patch.object(
            sampler_module,
            "get_ascend_device_type",
            return_value=sampler_module.AscendDeviceType.A2,
        ), patch.object(
            sampler_module.torch.ops,
            "_C_ascend",
            SimpleNamespace(),
            create=True,
        ):
            sampler_module._MISSING_TOP_K_TOP_P_OP_WARNED = False
            actual = sampler_module.apply_top_k_top_p(logits.clone(), k, None)

        torch.testing.assert_close(actual, expected)
