# screenaudit: how a "top 5% get represented" screen behaves when evidence is missing for some people

Standout is an AI talent agent that scouts roles for people who are not looking; its homepage says "The top 5% get represented. We're selective." (https://standout.work). Any selective screen over many candidates has two measurable properties: how many of the truly strongest people it finds, and whether who it finds depends on what is easy to observe (a prestigious name) instead of what is true (skill). This is a synthetic candidate pool with known skill to measure both. It says nothing about how Standout actually ranks people.

## The setup
`screenaudit/sim.py`: 4,000 candidates in two groups (A, B) with the same skill distribution. Cheap proxies: a prestige flag (more common in A, weakly tied to skill), years of experience, a keyword score that follows prestige. Evidence (work samples, public code): an unbiased but noisy skill measure that exists for 55% of A and 30% of B (dev). Four rules pick the top 5%:
- **prestige-heavy resume score:** 0.6 prestige + 0.2 years + 0.2 keywords.
- **evidence if present, else proxies:** the evidence score when it exists, else a regression prediction from the proxies.
- **posterior mean:** proxies as a prior, evidence updates it (precision-weighted).
- **posterior mean + 10% probes:** spend a work-sample budget (10% of the pool) on people with no evidence who look best on the proxies, then use the posterior mean.
Metrics: share of the true top 5% found, mean true skill of those picked, selection rate per group and the ratio (min / max).

## Results (60 pools per split)
Dev: seeds 0-59. Held-out (seeds 1000-1059, fixed in git before the first run): prestige 55% vs 12% by group, prestige tracks skill only in group A, evidence exists for 55% of A and 20% of B.

| Held-out | true top 5% found | mean skill picked | rate A / rate B | selection-rate ratio | share of picks from B |
|---|---|---|---|---|---|
| prestige-heavy resume score | 14.9% | 0.66 | 8.4% / 1.6% | 0.19 | 16% |
| evidence if present, else proxies | 32.8% | 1.39 | 7.4% / 2.6% | 0.36 | 26% |
| posterior mean | 33.5% | 1.40 | 7.6% / 2.5% | 0.33 | 25% |
| posterior mean + 10% probes | **46.6%** | **1.62** | 6.8% / 3.3% | **0.49** | 33% |

Dev: found 14.2% / 35.7% / 36.1% / 48.0%; ratio 0.48 / 0.53 / 0.52 / 0.72. Half of the true top 5% belong to group B in both.

- Using evidence roughly doubles the share of the truly strongest people found (14.9% to 33%), and the people picked are about 0.7 sd stronger.
- The probe budget is the largest single gain (to 46.6%), because it fixes the gap where evidence is missing.
- Even the best rule here picks group B at half the rate of group A (ratio 0.49; the 4/5 rule of thumb would be 0.8) because B has less evidence and less prestige, and the proxies carry no information for B in held-out. The posterior-mean rule is slightly worse than "evidence if present" on this ratio, because it shrinks noisy evidence.

## Read this before believing the table
Skill, proxies, groups and selection rules are my own construction; the numbers say how this synthetic screen behaves, not how Standout or any real screen does. Real skill is not one number, evidence is not unbiased, and groups are not two bins. Nothing here uses Standout data or systems.

## Run it (3 commands)
```
pip install numpy pytest
python -m pytest -q tests
python -m screenaudit.run
```
numpy only. 4 tests.

Unofficial prototype by Kushagra Pratap Singh. Not affiliated with Standout.
