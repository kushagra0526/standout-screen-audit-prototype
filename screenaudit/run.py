import json

import numpy as np

from . import sim


def score(seeds, held):
    acc = {m: {} for m in sim.METHODS}
    truth_share_b = []
    for s in seeds:
        p = sim.make_pool(s, held)
        truth_share_b.append(float((sim.top_mask(p["skill"]) & ~p["g"]).sum() / sim.top_mask(p["skill"]).sum()))
        for m, fn in sim.METHODS.items():
            for k, v in sim.metrics(p, sim.top_mask(fn(p))).items():
                acc[m].setdefault(k, []).append(v)
    out = {m: {k: round(float(np.mean(v)), 3) for k, v in d.items()} for m, d in acc.items()}
    out["_share_of_true_top_in_group_B"] = round(float(np.mean(truth_share_b)), 3)
    return out


def main():
    res = {"dev": score(range(0, 60), False), "held_out": score(range(1000, 1060), True)}
    json.dump(res, open("out/result.json", "w"), indent=1)
    for k, v in res.items():
        print(k, "true top 5% share in group B:", v.pop("_share_of_true_top_in_group_B"))
        for m, d in v.items():
            print(f"  {m:44}", d)


if __name__ == "__main__":
    main()
