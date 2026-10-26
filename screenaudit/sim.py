"""Candidates with a known true skill, observable proxies, and four ways to pick the top 5%.

True skill is independent of group. Proxies: a prestige flag (more common in group A, only weakly related to skill),
years of experience, a text/keyword score that follows prestige, and an evidence score (work samples, public code) that is
noisy but unbiased and only exists for some people (less often for group B).
"""
import numpy as np

N, TOP = 4000, 0.05


def make_pool(seed, held_out=False):
    rng = np.random.default_rng(seed)
    g = rng.random(N) < 0.5                                   # True = group A
    skill = rng.normal(0, 1, N)
    p_a, p_b = (0.55, 0.12) if held_out else (0.45, 0.20)
    slope = np.where(g, 0.12, 0.0) if held_out else 0.05   # prestige tracks skill a little; in held-out only for group A
    prestige = (rng.random(N) < np.clip(np.where(g, p_a, p_b) + slope * skill, 0, 1)).astype(float)
    years = np.clip(rng.normal(5, 3, N) + 0.8 * skill, 0, None)
    kw = 0.7 * prestige + rng.normal(0, 0.5, N) + 0.1 * skill
    has_ev = rng.random(N) < np.where(g, 0.55, 0.30 if not held_out else 0.20)
    evidence = np.where(has_ev, skill + rng.normal(0, 0.6, N), np.nan)
    return {"g": g, "skill": skill, "prestige": prestige, "years": years, "kw": kw, "evidence": evidence, "rng": rng}


def top_mask(score):
    k = int(TOP * len(score))
    m = np.zeros(len(score), bool)
    m[np.argsort(-score)[:k]] = True
    return m


def prestige_heavy(p):
    z = lambda x: (x - x.mean()) / x.std()
    return 0.6 * z(p["prestige"]) + 0.2 * z(p["years"]) + 0.2 * z(p["kw"])


def _prior(p, ev):
    """Regression of the evidence score on the cheap proxies, fitted on people who have evidence; applied to everyone."""
    X = np.column_stack([np.ones(N), p["prestige"], p["years"], p["kw"]])
    ok = ~np.isnan(ev)
    beta, *_ = np.linalg.lstsq(X[ok], ev[ok], rcond=None)
    resid = ev[ok] - X[ok] @ beta
    return X @ beta, float(resid.var())


def evidence_else_proxy(p, ev=None):
    ev = p["evidence"] if ev is None else ev
    prior, _ = _prior(p, ev)
    return np.where(np.isnan(ev), prior, ev)


def posterior_mean(p, ev=None, noise_var=0.36):
    ev = p["evidence"] if ev is None else ev
    prior, pv = _prior(p, ev)
    pv = max(pv - noise_var, 0.05)                      # prior variance of skill given the proxies
    w = pv / (pv + noise_var)
    return np.where(np.isnan(ev), prior, prior + w * (ev - prior))


def with_probes(p, budget=0.10):
    """Spend a work-sample budget on people without evidence who look best on the proxies, then use the posterior mean."""
    ev = p["evidence"].copy()
    prior, _ = _prior(p, ev)
    cand = np.where(np.isnan(ev))[0]
    k = int(budget * N)
    pick = cand[np.argsort(-prior[cand])[:k]]
    ev[pick] = p["skill"][pick] + p["rng"].normal(0, 0.6, len(pick))
    return posterior_mean(p, ev)


METHODS = {"prestige-heavy resume score": prestige_heavy, "evidence if present, else proxies": evidence_else_proxy,
           "posterior mean (evidence + proxies)": posterior_mean, "posterior mean + 10% work-sample probes": with_probes}


def metrics(p, sel):
    true_top = top_mask(p["skill"])
    a, b = sel[p["g"]].mean(), sel[~p["g"]].mean()
    return {"recall_of_true_top": float((sel & true_top).sum() / true_top.sum()), "mean_skill_selected": float(p["skill"][sel].mean()),
            "rate_A": float(a), "rate_B": float(b), "impact_ratio": float(min(a, b) / max(a, b)) if max(a, b) > 0 else 0.0,
            "share_B": float((sel & ~p["g"]).sum() / sel.sum())}
