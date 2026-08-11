r"""Verify the CA engine against a model whose gradients are exact by hand.

Construct a denoiser whose prediction is one matrix-vector product plus a term
that does not involve the parameter:

.. math:: p = W u + x_t

where ``u`` is the prompt embedding and ``W`` is the model's single attention
projection. For any loss ``L = mean((p - z)^2)`` with ``z`` detached,

.. math:: \nabla_W L = \frac{2}{m}(p - z) \otimes u

so the statistics have closed forms:

.. math::
    \mathrm{CA} = \frac{\lVert p - p_{aug}\rVert}{\lVert p - \epsilon\rVert},
    \qquad
    \cos(\hat g_a, \hat g_b) = \cos(p - p_{aug,a},\; p - p_{aug,b})

(the second because the shared ``⊗ u`` factor cancels out of the cosine). The
``+ x_t`` term keeps the closed form intact — it is constant in ``W`` — while
making the denominator genuinely timestep-dependent, so these tests also cover
the noise schedule and the seeding contract, not just the ratio arithmetic.
"""

from __future__ import annotations

import torch
from pytest import approx
from torch import nn

from calora.ca.measure import measure_concept_awareness, prepare_samples
from calora.data.dummy import DummyConceptDataset
from calora.models.adapter import Conditioning, DenoiserAdapter
from calora.models.projections import ProjectionTable
from calora.utils.seeding import derive_generator, stable_hash

DIM = 6
AXES = ["style", "viewpoint"]


class _AnalyticAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)


class _AnalyticUNet(nn.Module):
    def __init__(self, dim: int, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.attn1 = _AnalyticAttention(dim)
        with torch.no_grad():
            self.attn1.to_q.weight.copy_(torch.randn(dim, dim, generator=generator))

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.attn1.to_q(encoder_hidden_states) + sample


class _AnalyticAdapter(DenoiserAdapter):
    num_train_timesteps = 1000
    prediction_type = "epsilon"

    def __init__(self, dim: int = DIM, seed: int = 0):
        self.dim = dim
        self._unet = _AnalyticUNet(dim, seed)
        betas = torch.linspace(1e-4, 0.02, self.num_train_timesteps)
        self._alphas = torch.cumprod(1.0 - betas, dim=0)

    @property
    def unet(self):
        return self._unet

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    @property
    def alphas_cumprod(self):
        return self._alphas

    def encode_images(self, images):
        flat = images.reshape(images.shape[0], -1)
        return flat[:, : self.dim].contiguous()

    def encode_prompts(self, prompts):
        embeds = [
            torch.randn(
                self.dim,
                generator=torch.Generator().manual_seed(stable_hash("p", p) % (2**31 - 1)),
            )
            for p in prompts
        ]
        return {"prompt_embeds": torch.stack(embeds)}

    def predict(self, noisy_latents, timesteps, cond: Conditioning):
        return self._unet(noisy_latents, timesteps, cond["prompt_embeds"])

    def latent_shape(self, batch_size: int, resolution: int):
        return (batch_size, self.dim)


def _closed_form(adapter: _AnalyticAdapter, item: dict, timestep: int, index: int, seed: int = 0):
    """Recompute p, p_aug and eps exactly the way the measurement loop does.

    ``augmented`` comes back as ``{axis: [p_aug per slot]}``, one entry per
    ``augs_per_sample``.
    """
    weight = adapter.unet.attn1.to_q.weight.detach()
    latents = adapter.encode_images(item["image"].unsqueeze(0))
    noise = torch.randn(
        latents.shape, generator=derive_generator(seed, "noise", index), dtype=torch.float32
    )
    timesteps = torch.full((1,), timestep, dtype=torch.long)
    noisy = adapter.add_noise(latents, noise, timesteps)

    prediction = adapter.encode_prompts([item["caption"]])["prompt_embeds"] @ weight.T + noisy
    augmented = {
        axis: [
            adapter.encode_prompts([prompt])["prompt_embeds"] @ weight.T + noisy
            for prompt in item["augmented"][axis]
        ]
        for axis in AXES
    }
    return prediction, augmented, noise


def _measure(adapter, table, dataset, timestep: int, **kwargs):
    samples = prepare_samples(adapter, dataset, axes=AXES)
    return measure_concept_awareness(
        adapter, table, samples, timestep=timestep, axes=AXES, seed=0, **kwargs
    )


def test_projection_table_finds_the_single_projection():
    adapter = _AnalyticAdapter()
    table = ProjectionTable.from_model(adapter.unet)
    assert table.names == ["attn1.to_q"]
    assert table.total_params == DIM * DIM


def test_ca_matches_closed_form(augmenter):
    adapter = _AnalyticAdapter()
    table = ProjectionTable.from_model(adapter.unet)
    dataset = DummyConceptDataset(augmenter, num_items=1, resolution=64, seed=0)
    result = _measure(adapter, table, dataset, 81)

    prediction, augmented, noise = _closed_form(adapter, dataset[0], 81, 0)
    denominator = (prediction - noise).reshape(-1).norm()

    for axis in AXES:
        numerator = (prediction - augmented[axis][0]).reshape(-1).norm()
        expected = float(numerator / denominator)
        assert abs(float(result.ca(axis)[0]) - expected) < 1e-5, (
            f"{axis}: got {float(result.ca(axis)[0])}, expected {expected}"
        )


def test_cross_axis_cosine_matches_closed_form(augmenter):
    adapter = _AnalyticAdapter()
    table = ProjectionTable.from_model(adapter.unet)
    dataset = DummyConceptDataset(augmenter, num_items=1, resolution=64, seed=0)
    result = _measure(adapter, table, dataset, 81)

    prediction, augmented, _ = _closed_form(adapter, dataset[0], 81, 0)
    residual_a = (prediction - augmented["style"][0]).reshape(-1)
    residual_b = (prediction - augmented["viewpoint"][0]).reshape(-1)
    expected = float(torch.dot(residual_a, residual_b) / (residual_a.norm() * residual_b.norm()))

    # With one sample, cos-of-means and mean-of-cos necessarily coincide.
    assert abs(float(result.similarity(*AXES, source="cos_of_means")[0]) - expected) < 1e-5
    assert abs(float(result.similarity(*AXES, source="mean_of_cos")[0]) - expected) < 1e-5


def test_ca_averages_per_sample_ratios(augmenter):
    """CA is E[ratio], not a ratio of means; check mean and std over 5 samples."""
    adapter = _AnalyticAdapter()
    table = ProjectionTable.from_model(adapter.unet)
    dataset = DummyConceptDataset(augmenter, num_items=5, resolution=64, seed=0)
    result = _measure(adapter, table, dataset, 81)

    for axis in AXES:
        ratios = []
        for index in range(len(dataset)):
            prediction, augmented, noise = _closed_form(adapter, dataset[index], 81, index)
            numerator = (prediction - augmented[axis][0]).reshape(-1).norm()
            denominator = (prediction - noise).reshape(-1).norm()
            ratios.append(float(numerator / denominator))

        mean = sum(ratios) / len(ratios)
        std = (sum((r - mean) ** 2 for r in ratios) / len(ratios)) ** 0.5
        assert abs(float(result.ca(axis)[0]) - mean) < 1e-5
        assert abs(float(result.per_axis[axis].ca_std[0]) - std) < 1e-4


def test_ca_averages_over_augmentation_slots(augmenter):
    """With m>1 the estimand is the mean ratio over every (sample, slot) pair.

    The denominator is computed once per sample and shared by all its slots, so a
    slot must not re-count it — which is exactly what would happen if the
    accumulator treated each ``add_axis`` as a fresh sample.
    """
    adapter = _AnalyticAdapter()
    table = ProjectionTable.from_model(adapter.unet)
    dataset = DummyConceptDataset(augmenter, num_items=3, resolution=64, augs_per_sample=4, seed=0)
    result = _measure(adapter, table, dataset, 81)

    assert result.num_items == 3
    assert result.num_samples == 12
    for axis in AXES:
        ratios = []
        for index in range(len(dataset)):
            prediction, augmented, noise = _closed_form(adapter, dataset[index], 81, index)
            denominator = (prediction - noise).reshape(-1).norm()
            for slot in augmented[axis]:
                ratios.append(float((prediction - slot).reshape(-1).norm() / denominator))

        assert len(ratios) == 12
        assert abs(float(result.ca(axis)[0]) - sum(ratios) / len(ratios)) < 1e-5


def test_direction_and_paired_cosine_respect_the_slot_pairing(augmenter):
    """The direction mean sums every slot; the cosine pairs slot j with slot j."""
    adapter = _AnalyticAdapter()
    table = ProjectionTable.from_model(adapter.unet)
    dataset = DummyConceptDataset(augmenter, num_items=2, resolution=64, augs_per_sample=3, seed=0)
    result = _measure(adapter, table, dataset, 81)

    # g_c = (2/m)(p - p_aug) (x) u, so g_c / ||g_d|| is the residual scaled by a
    # per-sample constant; the mean direction is the mean of those outer products.
    expected_direction = {axis: torch.zeros(DIM, DIM) for axis in AXES}
    expected_cos = 0.0
    for index in range(len(dataset)):
        item = dataset[index]
        prediction, augmented, noise = _closed_form(adapter, item, 81, index)
        embedding = adapter.encode_prompts([item["caption"]])["prompt_embeds"].reshape(-1)
        scale = 2.0 / DIM
        denominator = (scale * (prediction - noise).reshape(-1).norm() * embedding.norm()).item()
        for axis in AXES:
            for slot in augmented[axis]:
                residual = scale * (prediction - slot).reshape(-1)
                expected_direction[axis] += torch.outer(residual, embedding) / denominator
        for slot_a, slot_b in zip(augmented["style"], augmented["viewpoint"]):
            a = (prediction - slot_a).reshape(-1)
            b = (prediction - slot_b).reshape(-1)
            expected_cos += float(torch.dot(a, b) / (a.norm() * b.norm()))

    for axis in AXES:
        measured = result.per_axis[axis].direction_mean
        assert measured is not None
        assert torch.allclose(
            measured["attn1.to_q"].reshape(DIM, DIM),
            expected_direction[axis] / 6,  # 2 samples * 3 slots
            atol=1e-5,
        )
    assert float(result.similarity(*AXES, source="mean_of_cos")[0]) == approx(expected_cos / 6)


def test_split_half_merges_back_to_the_undivided_measurement(augmenter):
    adapter = _AnalyticAdapter()
    table = ProjectionTable.from_model(adapter.unet)
    dataset = DummyConceptDataset(augmenter, num_items=6, resolution=64, augs_per_sample=2, seed=0)

    full = _measure(adapter, table, dataset, 81)
    split = _measure(adapter, table, dataset, 81, split_half=True)

    assert [half.num_items for half in split.halves] == [3, 3]
    for axis in AXES:
        assert torch.allclose(split.ca(axis), full.ca(axis), rtol=1e-10, atol=1e-7)
        # ...and each half really is a different estimate, or the check is vacuous.
        assert not torch.allclose(
            split.halves[0].ca(axis), split.halves[1].ca(axis), atol=1e-6
        )


def test_timestep_affects_ca_and_cos_of_means_but_not_mean_of_cos(augmenter):
    """Pins down how the two similarity statistics differ.

    In this model ``x_t`` cancels out of ``p - p_aug``, so every sample's
    concept-gradient *direction* is timestep-independent and ``mean_of_cos``
    must be exactly invariant. ``cos_of_means`` is not: averaging
    ``g_c / ||g_d||`` weights each sample by ``1/||g_d||``, which does depend on
    ``t``. That weighting is a real property of the proposal's statistic, not a
    bug — worth knowing when reading S(t) curves.
    """
    adapter = _AnalyticAdapter()
    table = ProjectionTable.from_model(adapter.unet)
    dataset = DummyConceptDataset(augmenter, num_items=3, resolution=64, seed=0)

    early = _measure(adapter, table, dataset, 1)
    late = _measure(adapter, table, dataset, 481)

    assert not torch.allclose(early.ca("style"), late.ca("style"))
    assert torch.allclose(
        early.similarity(*AXES, source="mean_of_cos"),
        late.similarity(*AXES, source="mean_of_cos"),
        atol=1e-6,
    )
    assert not torch.allclose(
        early.similarity(*AXES, source="cos_of_means"),
        late.similarity(*AXES, source="cos_of_means"),
        atol=1e-6,
    )


def test_elementwise_granularity_differs_from_matrix(augmenter):
    """The two readings of the paper's wording must not silently coincide."""
    adapter = _AnalyticAdapter()
    table = ProjectionTable.from_model(adapter.unet)
    dataset = DummyConceptDataset(augmenter, num_items=3, resolution=64, seed=0)

    matrix = _measure(adapter, table, dataset, 81, granularity="matrix")
    elementwise = _measure(adapter, table, dataset, 81, granularity="elementwise")

    assert matrix.granularity == "matrix"
    assert elementwise.granularity == "elementwise"
    assert not torch.allclose(matrix.ca("style"), elementwise.ca("style"))
    assert float(elementwise.ca("style")[0]) > 0.0
