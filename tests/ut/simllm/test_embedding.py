#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Unit tests for embedding extraction — pooling modes and shape handling."""

from __future__ import annotations

import pytest
import torch

from vllm_ascend.simllm.embedding import extract_embedding

# ---------------------------------------------------------------------------
# Model-based semantic tests — verify that real transformer hidden states
# produce meaningful embeddings (identical text → cos ≈ 1.0, dissimilar
# text → cos < 0.5).
# Skip with:  pytest -m "not model" tests/ut/simllm/
# ---------------------------------------------------------------------------

_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def tinyllm_tokenizer():
    """Load TinyLlama tokenizer once per test module."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(_MODEL_ID)


@pytest.fixture(scope="module")
def tinyllm_model():
    """Load TinyLlama-1.1B once per test module (CPU, float32).

    This is ~2.2 GiB and takes a few seconds to load — module-scoped so
    all semantic tests share one instance.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(_MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    return model


def _get_last_hidden(
    model, tokenizer, texts: list[str]
) -> torch.Tensor:
    """Run a forward pass and return the last-layer hidden states ``[B, S, D]``."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # outputs.hidden_states[-1]: [B, S, D]
    return outputs.hidden_states[-1]


@pytest.mark.model
class TestSemanticEmbedding:
    """Semantic quality checks using a real TinyLlama-1.1B model."""

    def test_identical_text_similarity(self, tinyllm_model, tinyllm_tokenizer):
        """Same text twice → cosine similarity ≈ 1.0."""
        text = "Machine learning is a subset of artificial intelligence."
        hs = _get_last_hidden(tinyllm_model, tinyllm_tokenizer, [text, text])
        emb = extract_embedding(hs, pooling="mean")  # [2, D]

        sim = (emb[0] * emb[1]).sum()
        assert sim.item() > 0.99, f"Expected cos ≈ 1.0, got {sim.item():.4f}"

    def test_semantically_similar_texts(self, tinyllm_model, tinyllm_tokenizer):
        """Paraphrased texts → cosine similarity should be high (> 0.7)."""
        text_a = "What is the capital of France?"
        text_b = "Can you tell me the capital city of France?"
        hs = _get_last_hidden(tinyllm_model, tinyllm_tokenizer, [text_a, text_b])
        emb = extract_embedding(hs, pooling="mean")

        sim = (emb[0] * emb[1]).sum()
        assert sim.item() > 0.7, (
            f"Paraphrases should have high similarity, got {sim.item():.4f}"
        )

    def test_dissimilar_texts_low_similarity(self, tinyllm_model, tinyllm_tokenizer):
        """Unrelated texts → cosine similarity should be low (< 0.5)."""
        text_a = "Explain the backpropagation algorithm in detail."
        text_b = "What is the best recipe for chocolate chip cookies?"
        hs = _get_last_hidden(tinyllm_model, tinyllm_tokenizer, [text_a, text_b])
        emb = extract_embedding(hs, pooling="mean")

        sim = (emb[0] * emb[1]).sum()
        assert sim.item() < 0.5, (
            f"Unrelated texts should have low similarity, got {sim.item():.4f}"
        )

    def test_pooling_mode_consistency(self, tinyllm_model, tinyllm_tokenizer):
        """All three pooling modes produce L2-normalized output."""
        text = "Artificial intelligence is transforming industries worldwide."
        hs = _get_last_hidden(tinyllm_model, tinyllm_tokenizer, [text])

        for mode in ("mean", "last", "cls"):
            emb = extract_embedding(hs, pooling=mode)
            norm = emb.norm(p=2, dim=-1)
            assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6), (
                f"Mode {mode}: expected unit norm, got {norm.item():.4f}"
            )


class TestExtractEmbedding:
    """Pooling modes: mean, last, cls."""

    def test_mean_pooling_2d_input(self):
        hs = torch.randn(16, 128)  # [S, D]
        emb = extract_embedding(hs, pooling="mean")
        assert emb.shape == (1, 128)
        # L2-normed
        assert torch.allclose(emb.norm(dim=-1), torch.tensor([1.0]), atol=1e-6)

    def test_mean_pooling_3d_input(self):
        hs = torch.randn(4, 16, 128)  # [B, S, D]
        emb = extract_embedding(hs, pooling="mean")
        assert emb.shape == (4, 128)

    def test_last_pooling(self):
        hs = torch.randn(2, 10, 64)  # [B, S, D]
        emb = extract_embedding(hs, pooling="last")
        expected = hs[:, -1, :]  # last token of each sequence
        expected_norm = torch.nn.functional.normalize(expected, dim=-1)
        assert torch.allclose(emb, expected_norm, atol=1e-6)

    def test_cls_pooling(self):
        hs = torch.randn(3, 8, 32)  # [B, S, D]
        emb = extract_embedding(hs, pooling="cls")
        expected = hs[:, 0, :]  # first token
        expected_norm = torch.nn.functional.normalize(expected, dim=-1)
        assert torch.allclose(emb, expected_norm, atol=1e-6)

    def test_identical_inputs_near_1_similarity(self):
        hs = torch.randn(2, 10, 128)
        emb1 = extract_embedding(hs, pooling="mean")
        emb2 = extract_embedding(hs, pooling="mean")
        # L2-normalized => cosine = dot product
        sim = (emb1 * emb2).sum(dim=-1)
        assert torch.allclose(sim, torch.tensor([1.0, 1.0]), atol=1e-6)

    def test_dissimilar_inputs_low_similarity(self):
        """Different inputs should have cosine similarity < 0.5 with high prob."""
        hs1 = torch.randn(2, 10, 128)
        hs2 = torch.randn(2, 10, 128)
        emb1 = extract_embedding(hs1, pooling="mean")
        emb2 = extract_embedding(hs2, pooling="mean")
        sim = (emb1 * emb2).sum(dim=-1)
        assert (sim < 0.5).all(), f"Expected low similarity, got {sim}"

    def test_invalid_pooling_raises(self):
        hs = torch.randn(4, 8, 32)
        with pytest.raises(ValueError, match="Unknown pooling mode"):
            extract_embedding(hs, pooling="max")

    def test_normalized_output(self):
        hs = torch.randn(5, 12, 96)
        for mode in ("mean", "last", "cls"):
            emb = extract_embedding(hs, pooling=mode)
            norms = emb.norm(p=2, dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
