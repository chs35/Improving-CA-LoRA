"""Selection budgets, multi-timestep aggregation and timestep scoring."""

from __future__ import annotations

import pytest
import torch

from calora.ca.select import (
    Selection,
    aggregate_across_timesteps,
    axis_scores,
    normalize_scores,
    pooled_similarity_thresholds,
    rank_timesteps,
    score_timestep,
    select_all,
    select_random,
    select_topk,
    timestep_rank_agreement,
)
from calora.ca.stats import AxisStats, MeasurementResult
from calora.models.projections import ProjectionSpec, ProjectionTable

AXES = ("style", "viewpoint")
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5)


def _table(numels: list[int]) -> ProjectionTable:
    kinds = ("q", "k", "v", "out")
    return ProjectionTable(
        [
            ProjectionSpec(
                name=f"blk{i}.attn1.to_{kinds[i % 4]}",
                kind=kinds[i % 4],
                attn_index=1,
                block="down",
                block_index=0,
                shape=(n,),
                numel=n,
            )
            for i, n in enumerate(numels)
        ]
    )


def _result(timestep: int, ca_style, ca_view, similarity, table: ProjectionTable):
    names = tuple(table.names)
    zeros = torch.zeros(len(names))
    return MeasurementResult(
        timestep=timestep,
        num_samples=10,
        num_items=10,
        axes=AXES,
        names=names,
        per_axis={
            "style": AxisStats(torch.tensor(ca_style), zeros, zeros, None),
            "viewpoint": AxisStats(torch.tensor(ca_view), zeros, zeros, None),
        },
        gd_norm_mean=zeros,
        valid_count=torch.full((len(names),), 10.0),
        cos_of_means={"style|viewpoint": torch.tensor(similarity)},
        mean_of_cos={"style|viewpoint": torch.tensor(similarity)},
        table_records=table.to_records(),
    )


# -- normalisation ----------------------------------------------------------------------


def test_sum_normalisation_makes_a_distribution():
    normalized = normalize_scores(torch.tensor([1.0, 3.0]), "sum")
    assert float(normalized.sum()) == pytest.approx(1.0)


def test_rank_normalisation_is_scale_free():
    a = normalize_scores(torch.tensor([1.0, 2.0, 10.0]), "rank")
    b = normalize_scores(torch.tensor([100.0, 200.0, 1000.0]), "rank")
    assert torch.allclose(a, b)
    assert a.tolist() == [0.0, 0.5, 1.0]


def test_unknown_normalisation_is_rejected():
    with pytest.raises(ValueError, match="normalization"):
        normalize_scores(torch.tensor([1.0]), "nope")


# -- selection --------------------------------------------------------------------------


def test_topk_respects_the_parameter_budget():
    table = _table([100] * 10)  # 1000 params total
    scores = torch.arange(10, dtype=torch.float64)
    selection = select_topk(scores, table, fraction=0.3, budget_basis="attn_params")

    assert selection.params_selected <= 300
    assert selection.fraction_achieved == pytest.approx(0.3)
    # Highest scores first: indices 9, 8, 7.
    assert set(selection.names) == {table[i].name for i in (7, 8, 9)}


def test_topk_skips_oversized_projections_to_fill_the_budget():
    """A wide projection that would blow the budget must not block smaller ones."""
    table = _table([1000, 10, 10])
    scores = torch.tensor([10.0, 5.0, 1.0])  # the huge one ranks first
    selection = select_topk(scores, table, fraction=0.02)  # budget = 0.02 * 1020 ~ 20

    assert table[0].name not in selection.names
    assert set(selection.names) == {table[1].name, table[2].name}
    assert selection.params_selected == 20


def test_topk_always_returns_at_least_one():
    table = _table([1000, 1000])
    selection = select_topk(torch.tensor([1.0, 2.0]), table, fraction=0.0001)
    assert len(selection) == 1
    assert selection.names == [table[1].name]


def test_projection_count_basis_counts_layers_not_params():
    table = _table([1, 10, 100, 1000])
    selection = select_topk(
        torch.tensor([4.0, 3.0, 2.0, 1.0]), table, fraction=0.5, budget_basis="projection_count"
    )
    assert len(selection) == 2


def test_random_baseline_matches_the_budget():
    table = _table([100] * 10)
    ca = select_topk(torch.arange(10, dtype=torch.float64), table, fraction=0.3)
    random = select_random(table, fraction=0.3, seed=1)
    assert random.params_selected == ca.params_selected
    assert random.strategy == "random"


def test_select_all_takes_everything():
    table = _table([100] * 4)
    selection = select_all(table)
    assert len(selection) == 4
    assert selection.fraction_achieved == 1.0


def test_jaccard_and_round_trip():
    table = _table([100] * 10)
    a = select_topk(torch.arange(10, dtype=torch.float64), table, fraction=0.3)
    b = select_topk(torch.arange(10, dtype=torch.float64).flip(0), table, fraction=0.3)
    assert a.jaccard(a) == 1.0
    assert a.jaccard(b) == 0.0

    restored = Selection.from_dict(a.to_dict(), table)
    assert restored.names == a.names
    assert restored.indices == a.indices


def test_selection_rejects_bad_input():
    table = _table([100] * 3)
    with pytest.raises(ValueError, match="scores has"):
        select_topk(torch.zeros(2), table)
    with pytest.raises(ValueError, match="fraction"):
        select_topk(torch.zeros(3), table, fraction=1.5)
    with pytest.raises(ValueError, match="budget_basis"):
        select_topk(torch.zeros(3), table, budget_basis="nope")


# -- aggregation ------------------------------------------------------------------------


def test_axis_combiners():
    table = _table([10] * 3)
    result = _result(81, [1.0, 5.0, 3.0], [3.0, 1.0, 3.0], [0.0] * 3, table)
    assert axis_scores(result, "style").tolist() == [1.0, 5.0, 3.0]
    assert axis_scores(result, "min").tolist() == [1.0, 1.0, 3.0]
    assert axis_scores(result, "max").tolist() == [3.0, 5.0, 3.0]
    assert axis_scores(result, "mean").tolist() == [2.0, 3.0, 3.0]
    with pytest.raises(KeyError, match="Unknown selection axis"):
        axis_scores(result, "nope")


def test_aggregation_normalises_before_averaging():
    """One timestep with a huge CA scale must not dominate the aggregate."""
    table = _table([10] * 3)
    results = {
        81: _result(81, [1.0, 2.0, 3.0], [0.0] * 3, [0.0] * 3, table),
        201: _result(201, [3000.0, 2000.0, 1000.0], [0.0] * 3, [0.0] * 3, table),
    }
    aggregated = aggregate_across_timesteps(results, axis="style", method="normalized_mean")
    # Opposite orderings of equal weight cancel to a flat score.
    assert aggregated.tolist() == pytest.approx([1 / 3] * 3, abs=1e-9)


def test_rank_mean_aggregation_ignores_magnitude():
    table = _table([10] * 3)
    results = {
        81: _result(81, [1.0, 2.0, 3.0], [0.0] * 3, [0.0] * 3, table),
        201: _result(201, [10.0, 20.0, 30.0], [0.0] * 3, [0.0] * 3, table),
    }
    aggregated = aggregate_across_timesteps(results, axis="style", method="rank_mean")
    assert aggregated.tolist() == [0.0, 0.5, 1.0]


def test_aggregation_rejects_unknown_timesteps():
    table = _table([10] * 3)
    results = {81: _result(81, [1.0] * 3, [1.0] * 3, [0.0] * 3, table)}
    with pytest.raises(KeyError, match="No measurement"):
        aggregate_across_timesteps(results, timesteps=[999])


# -- timestep scoring -------------------------------------------------------------------


def _score(result, table, thresholds=None, **kwargs):
    """score_timestep with a one-entry quantile sweep, for the single-t tests."""
    thresholds = {0.3: 0.0} if thresholds is None else thresholds
    return score_timestep(
        result, table, axis_pair=AXES, sim_thresholds=thresholds, primary_quantile=0.3, **kwargs
    )


def test_score_rewards_high_ca_with_low_similarity():
    table = _table([10] * 4)
    # Same CA profile; the only difference is how much the axes disagree.
    agreeing = _result(1, [1.0, 1.0, 1.0, 1.0], [1.0] * 4, [0.9, 0.9, 0.9, 0.9], table)
    conflicting = _result(81, [1.0, 1.0, 1.0, 1.0], [1.0] * 4, [-0.5, -0.5, -0.5, -0.5], table)

    low = _score(agreeing, table)
    high = _score(conflicting, table)
    assert high.score > low.score
    assert high.selected_mean_similarity < low.selected_mean_similarity
    assert high.expanded_weighted_similarity < low.expanded_weighted_similarity


def test_score_weights_disagreement_by_ca():
    """Disagreement on a low-CA projection should count for less."""
    table = _table([10] * 2)
    ca_on_conflict = _result(1, [9.0, 1.0], [9.0, 1.0], [-1.0, 1.0], table)
    ca_on_agreement = _result(2, [1.0, 9.0], [1.0, 9.0], [-1.0, 1.0], table)

    a = _score(ca_on_conflict, table)
    b = _score(ca_on_agreement, table)
    assert a.score > b.score


def test_conflict_threshold_is_pooled_across_timesteps():
    """A per-timestep quantile would make conflict_count constant; it must not be."""
    table = _table([10] * 4)
    # Identical CA profiles, so only the similarities can move conflict_count.
    results = {
        1: _result(1, [4.0, 3.0, 2.0, 1.0], [1.0] * 4, [0.9, 0.8, 0.7, 0.6], table),
        81: _result(81, [4.0, 3.0, 2.0, 1.0], [1.0] * 4, [-0.9, -0.8, 0.7, 0.6], table),
    }
    thresholds = pooled_similarity_thresholds(results, AXES, quantiles=[0.25, 0.5])
    assert -0.8 < thresholds[0.25] < 0.6

    # fraction 0.5 of 40 params keeps the two highest-CA projections at each t.
    scores = {
        s.timestep: s
        for s in rank_timesteps(
            results,
            table,
            fraction=0.5,
            conflict_quantiles=[0.25, 0.5],
            primary_quantile=0.25,
        )
    }
    assert scores[1].conflict_count == 0
    assert scores[81].conflict_count == 2


def test_rank_timesteps_orders_by_the_requested_key():
    table = _table([10] * 4)
    results = {
        1: _result(1, [4.0, 3.0, 2.0, 1.0], [1.0] * 4, [0.9, 0.8, 0.7, 0.6], table),
        81: _result(81, [4.0, 3.0, 2.0, 1.0], [1.0] * 4, [-0.9, -0.8, 0.7, 0.6], table),
    }
    by_score = rank_timesteps(results, table, fraction=0.5, rank_by="score")
    assert by_score[0].timestep == 81

    for key in ("selected_mean_similarity", "expanded_weighted_similarity", "conflict_count"):
        assert rank_timesteps(results, table, fraction=0.5, rank_by=key)[0].timestep == 81

    with pytest.raises(ValueError, match="rank_by"):
        rank_timesteps(results, table, rank_by="nope")


# -- top-n*k weighted similarity ----------------------------------------------------------


def test_expanded_weighted_similarity_matches_a_hand_calculation():
    """Weights are the selection axis's CA, renormalised inside the expanded set."""
    table = _table([10] * 4)  # 40 params; fraction 0.25 -> top-1, x2 -> top-2
    result = _result(81, [4.0, 3.0, 2.0, 1.0], [0.0] * 4, [-1.0, 0.5, 0.9, 0.9], table)

    score = _score(result, table, fraction=0.25, selection_multiplier=2)

    assert score.num_selected == 1
    assert score.num_expanded == 2
    # top-k is projection 0 alone.
    assert score.selected_mean_similarity == pytest.approx(-1.0)
    # top-n*k is projections 0 and 1: unweighted (-1 + 0.5) / 2.
    assert score.expanded_mean_similarity == pytest.approx(-0.25)
    # ...weighted by CA 4 and 3, renormalised to 4/7 and 3/7.
    expected = (4.0 * -1.0 + 3.0 * 0.5) / 7.0
    assert score.expanded_weighted_similarity == pytest.approx(expected)


def test_expanded_selection_is_capped_at_the_whole_table():
    table = _table([10] * 4)
    result = _result(81, [4.0, 3.0, 2.0, 1.0], [0.0] * 4, [0.1, 0.2, 0.3, 0.4], table)
    # fraction * multiplier = 2.0 must clamp to 1.0 rather than raising.
    score = _score(result, table, fraction=0.5, selection_multiplier=4)
    assert score.num_expanded == 4
    assert score.expanded_mean_similarity == pytest.approx(0.25)


def test_score_timestep_rejects_a_primary_quantile_outside_the_sweep():
    table = _table([10] * 2)
    result = _result(81, [1.0, 2.0], [1.0, 2.0], [0.0, 0.0], table)
    with pytest.raises(KeyError, match="primary quantile"):
        score_timestep(
            result, table, axis_pair=AXES, sim_thresholds={0.1: 0.0}, primary_quantile=0.3
        )


# -- quantile sweep ------------------------------------------------------------------------


def test_thresholds_and_conflict_counts_are_monotone_in_the_quantile():
    table = _table([10] * 4)
    results = {
        1: _result(1, [4.0, 3.0, 2.0, 1.0], [1.0] * 4, [-0.9, -0.2, 0.4, 0.8], table),
        81: _result(81, [4.0, 3.0, 2.0, 1.0], [1.0] * 4, [-0.5, 0.1, 0.6, 0.95], table),
    }
    thresholds = pooled_similarity_thresholds(results, AXES, quantiles=QUANTILES)
    assert sorted(thresholds) == list(QUANTILES)
    values = [thresholds[q] for q in QUANTILES]
    assert values == sorted(values), "a higher quantile must not lower tau"

    scores = rank_timesteps(
        results, table, fraction=0.5, conflict_quantiles=QUANTILES, primary_quantile=0.3
    )
    for score in scores:
        counts = [score.conflict_counts[q] for q in QUANTILES]
        assert counts == sorted(counts), "conflict_count must be non-decreasing in the quantile"
        assert score.sim_threshold == thresholds[0.3]
        assert score.conflict_count == score.conflict_counts[0.3]
        assert score.conflict_fraction == pytest.approx(
            score.conflict_count / score.num_selected
        )


def test_score_dict_stringifies_quantile_keys_for_json():
    table = _table([10] * 2)
    result = _result(81, [2.0, 1.0], [2.0, 1.0], [0.0, 0.0], table)
    payload = _score(result, table, thresholds={0.1: -0.5, 0.3: 0.5})
    assert set(payload.to_dict()["conflict_counts"]) == {"0.1", "0.3"}
    assert payload.to_dict()["primary_quantile"] == 0.3


# -- split-half rank agreement -------------------------------------------------------------


def _ranked(values: dict[int, float]):
    """Scored timesteps that differ only in expanded_weighted_similarity."""
    table = _table([10] * 2)
    results = {
        t: _result(t, [2.0, 1.0], [2.0, 1.0], [v, v], table) for t, v in values.items()
    }
    return rank_timesteps(
        results, table, fraction=0.5, rank_by="expanded_weighted_similarity"
    )


def test_agreement_is_perfect_against_itself():
    scores = _ranked({1: -0.5, 81: 0.0, 201: 0.5, 481: 0.9})
    stats = timestep_rank_agreement(
        scores, scores, rank_by="expanded_weighted_similarity", top_n=2
    )
    assert stats["num_common"] == 4
    assert stats["spearman"] == pytest.approx(1.0)
    assert stats["top_overlap"] == pytest.approx(1.0)


def test_agreement_is_minus_one_when_the_halves_rank_oppositely():
    a = _ranked({1: -0.5, 81: 0.0, 201: 0.5, 481: 0.9})
    b = _ranked({1: 0.9, 81: 0.5, 201: 0.0, 481: -0.5})
    stats = timestep_rank_agreement(a, b, rank_by="expanded_weighted_similarity", top_n=2)
    assert stats["spearman"] == pytest.approx(-1.0)
    assert stats["top_overlap"] == 0.0
    assert stats["top_a"] == [1, 81]
    assert stats["top_b"] == [481, 201]


def test_agreement_uses_only_the_common_timesteps():
    a = _ranked({1: -0.5, 81: 0.0, 201: 0.5})
    b = _ranked({81: 0.0, 201: 0.5, 481: 0.9})
    stats = timestep_rank_agreement(a, b, rank_by="expanded_weighted_similarity", top_n=5)
    assert stats["timesteps"] == [81, 201]
    assert stats["num_common"] == 2
