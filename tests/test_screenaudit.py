import numpy as np

from screenaudit import sim


def test_top_mask_size():
    assert sim.top_mask(np.random.default_rng(0).normal(size=1000)).sum() == 50


def test_pool_is_reproducible_and_skill_independent_of_group():
    a, b = sim.make_pool(3), sim.make_pool(3)
    assert np.array_equal(a["skill"], b["skill"])
    assert abs(a["skill"][a["g"]].mean() - a["skill"][~a["g"]].mean()) < 0.15


def test_posterior_mean_shrinks_toward_the_prior_for_noisy_evidence():
    p = sim.make_pool(4)
    pm, ev = sim.posterior_mean(p), sim.evidence_else_proxy(p)
    has = ~np.isnan(p["evidence"])
    assert np.var(pm[has]) < np.var(ev[has])


def test_evidence_based_methods_find_more_true_top_people_than_prestige():
    r = [(sim.metrics(p, sim.top_mask(sim.posterior_mean(p)))["recall_of_true_top"], sim.metrics(p, sim.top_mask(sim.prestige_heavy(p)))["recall_of_true_top"])
         for p in (sim.make_pool(s) for s in range(8))]
    assert np.mean([a for a, _ in r]) > np.mean([b for _, b in r])
