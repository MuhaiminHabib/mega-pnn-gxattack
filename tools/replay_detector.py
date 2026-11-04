"""Replay the original detector from this checkout against local AMLSim CSVs."""
import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_dir = Path(__file__).resolve().parents[1] / "model"
    sys.path.insert(0, str(model_dir))
    import mega_amlsim_adapter as adapter
    import run_amlsim_audit as audit

    original = audit.OUT
    protocol = json.loads((original / "protocol.json").read_text())
    for name, expected in protocol["source_hashes"].items():
        if audit.sha(model_dir / name) != expected:
            raise ValueError(f"Archived model source differs from the protocol: {name}")

    raw = args.raw.resolve(strict=True)
    audit.OUT = args.output.resolve()
    audit.OUT.mkdir(parents=True, exist_ok=False)
    for name in ("best.pt", "predictions.npz"):
        shutil.copyfile(original / name, audit.OUT / name)

    audit.torch.set_num_threads(protocol["threads"])
    audit.torch.set_num_interop_threads(1)
    audit.torch.manual_seed(audit.SEED)
    audit.np.random.seed(audit.SEED)
    random.seed(audit.SEED)
    graph, masks, ids, tx_ids, meta = adapter.prepare(raw)
    if meta != protocol["dataset"]:
        raise ValueError("AMLSim inputs differ from the archived protocol")
    model = adapter.make_model(graph)
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    audit.evaluate(model, graph, masks, ids, tx_ids, meta, digest, True)


if __name__ == "__main__":
    main()

