"""Recompute the ten-account top-25 comparisons using saved rankings."""
import itertools
import json
from pathlib import Path
from statistics import mean

RUN = Path(__file__).resolve().parents[1] / "experiments" / "mega_retraining_validation10_v1"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def replacement(left, right):
    return 100 * (1 - len(set(left) & set(right)) / 25)


def main():
    nodes = load(RUN / "cohort.json")["validation_nodes"]
    print("Checkpoint           Clean seeds        GX    Random")
    for arm in ("baseline", "clean", "random", "adversarial"):
        rows = []
        for node in nodes:
            def ranking(variant, seed):
                record = load(RUN / "explanations" / f"{arm}_{node}_{variant}_{seed}.json")
                assert record["eligible"] and len(set(record["top25"])) == 25
                return record["top25"]

            clean = [ranking("clean", seed) for seed in range(3)]
            rows.append((
                mean(replacement(a, b) for a, b in itertools.combinations(clean, 2)),
                replacement(clean[0], ranking("attack", 0)),
                replacement(clean[0], ranking("random", 0)),
            ))
        print(f"{arm:18s}" + "".join(f"{mean(values):10.2f}%" for values in zip(*rows)))


if __name__ == "__main__":
    main()
