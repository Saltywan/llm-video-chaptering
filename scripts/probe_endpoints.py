#!/usr/bin/env python
"""Sequential smoke only. Auth from environment or private JSON on stdin."""

import argparse
import json
import os
from pathlib import Path
import sys

from chaptering.inference import endpoint_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models.json")
    parser.add_argument("--aliases", nargs="+", default=["M1", "M2"])
    parser.add_argument("--records", help="Fixed canonical train/validation JSONL; defaults to synthetic")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, help="Explicit smoke-only total generation cap")
    parser.add_argument("--auth-stdin", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.max_tokens is not None:
        if args.max_tokens < 1:
            raise ValueError("--max-tokens must be positive")
        config["smoke"]["max_tokens"] = args.max_tokens
    auth = json.load(sys.stdin) if args.auth_stdin else {}
    if args.records:
        records = [json.loads(line) for line in Path(args.records).read_text().splitlines() if line.strip()]
        if any(row.get("split") not in {"train", "validation"} for row in records):
            raise ValueError("Smoke inputs must belong to train/validation")
    else:
        records = [{"video_id": "synthetic-c-smoke", "sentences": [
            "To bake bread, mix flour with water and yeast.",
            "Knead the dough, let it rise, then bake it in a hot oven.",
            "Now let us discuss basic bicycle maintenance.",
            "Check tire pressure and lubricate the chain before a long ride."]}]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        for alias in args.aliases:
            model = config["models"][alias]
            key = auth.get(model["api_key_env"]) or os.environ.get(model["api_key_env"])
            if not key:
                raise ValueError(f"Missing private authentication for {alias}")
            for row in records[:args.limit]:
                result = endpoint_prediction(row, model, config["smoke"], key, "endpoint-smoke-" + alias)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps({"alias": alias, "video_id": row["video_id"],
                                  "status": result["status"], "wall_seconds": result["wall_seconds"],
                                  "error": result["error"], "usage": result["usage"],
                                  "reasoning_source": result.get("reasoning_source")}), flush=True)


if __name__ == "__main__":
    main()
