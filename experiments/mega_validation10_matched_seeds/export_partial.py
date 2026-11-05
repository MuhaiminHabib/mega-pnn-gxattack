import hashlib
import itertools
import json
from pathlib import Path
from statistics import mean

source = Path("work/mega-validation10-matched-seeds")
dest = Path("work/github-reviewed/experiments/mega_validation10_matched_seeds")
(dest / "explanations").mkdir(parents=True, exist_ok=True)
protocol = json.loads((source / "protocol.json").read_text())
nodes = protocol["cohort"]["validation_nodes"]
arms = protocol["arms"]
records = {}
invalid = []
for path in (source / "explanations").glob("*.json"):
    if path.name.endswith("_graphs.json"):
        continue
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        key = (row["arm"], row["node"], row["variant"], row["repeat"])
        assert key[0] in arms and key[1] in nodes and key[2] in ("clean", "attack", "random") and key[3] in range(3)
        if row["eligible"]:
            assert row["seed"] == 20261017 + row["node"] + row["repeat"] * 100000
            raw = path.with_suffix(".npz")
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == row["raw_sha256"]
        allowed = ("arm","node","variant","repeat","eligible","reason","seed","top5","top25","raw_sha256","state_sha256","protocol_sha256","p_graph","p_soft","p_delete","p_keep_only_top5","drop_pp","steps","seconds")
        exported = {key: row[key] for key in allowed if key in row}
        (dest / "explanations" / path.name).write_text(json.dumps(exported, indent=2))
        records[key] = row
    except Exception as error:
        invalid.append({"file":path.name,"error":str(error)})

def complete(node):
    return all((arm,node,variant,seed) in records and records[arm,node,variant,seed]["eligible"] for arm,variant,seed in itertools.product(arms,("clean","attack","random"),range(3)))

common_nodes = [node for node in nodes if complete(node)]
table = []
for arm in arms:
    row = {"arm": arm, "completed_fits": sum(key[0] == arm for key in records), "common_accounts": len(common_nodes)}
    for k in (5,25):
        for variant in ("attack","random"):
            accounts = []
            for node in common_nodes:
                values = []
                for seed in range(3):
                    a = set(records[arm,node,"clean",seed][f"top{k}"])
                    b = set(records[arm,node,variant,seed][f"top{k}"])
                    values.append(100 * (1-len(a & b)/k))
                accounts.append(mean(values))
            row[f"{variant}_top{k}_replacement_pct"] = mean(accounts) if accounts else None
    table.append(row)
summary = {"state":"stopped_after_repeated_pc_restarts","planned_fits":360,"verified_saved_fits":len(records),"invalid_records":invalid,"common_complete_accounts":common_nodes,"rows":table,"seed_formula":"20261017 + node + repeat * 100000","steps_per_fit":4500,"note":"Partial snapshot. Means use only accounts with all 36 fits complete and eligible across all four arms. Seed effects are paired within each account."}
(dest / "summary.json").write_text(json.dumps(summary,indent=2))
(dest / "cohort.json").write_text(json.dumps({"validation_nodes":nodes,"arms":arms},indent=2))
lines = ["# Partial matched-seed GNNExplainer rerun", "", f"Verified {len(records)} of 360 saved fits. {len(invalid)} records failed validation. The run is stopped following repeated PC restarts.", "", "Each account has four checkpoints, three saved graph variants and three matching seeds. Each fit uses 4500 steps.", "", f"Common complete accounts across all four checkpoints: {common_nodes}.", "", "| Checkpoint | Saved fits | Common accounts | GX top-25 replacement | Random top-25 replacement |","|---|---:|---:|---:|---:|"]
for row in table:
    lines.append(f'| {row["arm"]} | {row["completed_fits"]} | {row["common_accounts"]} | {row["attack_top25_replacement_pct"]:.2f}% | {row["random_top25_replacement_pct"]:.2f}% |')
lines += ["", "These are interim results on the common completed accounts. Three paired seed differences are averaged within each account, then across accounts. Full-cohort conclusions require completion.", "", "Ranking JSONs include seed, transaction identifiers, fidelity measurements and original raw-mask SHA256 hashes. Local raw mask files were checked against those hashes before export.", "", "The current status file contains null bytes following a restart and cannot identify the latest interrupted step reliably. Historical complete records were validated individually.", ""]
(dest / "README.md").write_text(chr(10).join(lines))
print(json.dumps(summary,indent=2))

