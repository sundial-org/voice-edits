"""Evaluate a model (base or checkpoint) on the voice-edit test set.

Usage: run_eval.py [--split test] [--mode voice|text] [--model-path tinker://...] [--limit N] [--out name]
"""
import argparse
import json

import common

common.load_env()
import tinker  # noqa: E402
from tinker import types  # noqa: E402
from tml_renderers import chat  # noqa: E402
from tml_renderers.tinker import token_spans_to_tinker_model_input  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", default="voice", choices=["voice", "text"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    examples = [json.loads(l) for l in (common.DATA / "examples.jsonl").read_text().splitlines()]
    examples = [e for e in examples if e["split"] == args.split][: args.limit]
    print(f"{len(examples)} examples ({args.split}, {args.mode})")

    renderer = common.get_renderer()
    sc = tinker.ServiceClient()
    if args.model_path:
        sampler = sc.create_sampling_client(model_path=args.model_path)
    else:
        sampler = sc.create_sampling_client(base_model=common.MODEL)

    futures = []
    for ex in examples:
        if args.mode == "voice":
            msgs = common.build_messages(ex["doc"], wav_path=common.DATA / "wav" / ex["wav"])
        else:
            msgs = common.build_messages(ex["doc"], instruction=ex["instruction"])
        spans, parser = renderer.render_for_completion(msgs)
        fut = sampler.sample(
            prompt=token_spans_to_tinker_model_input(spans),
            sampling_params=types.SamplingParams(
                max_tokens=1200, temperature=0.0, stop=renderer.stop()),
            num_samples=1)
        futures.append((ex, parser, fut))

    rows, agg = [], {"parsed": 0, "applied": 0, "exact": 0, "failed": 0}
    for ex, parser, fut in futures:
        try:
            seq = fut.result().sequences[0]
            text = "\n".join(m.content.text for m in parser.parse_tokens(seq.tokens)
                             if isinstance(m.content, chat.Text))
        except Exception as e:
            agg["failed"] += 1
            rows.append({"id": ex["id"], "ops": ex["ops"], "error": f"sample: {e}"})
            continue
        s = common.score(ex["doc"], text, ex["target"])
        for k in ("parsed", "applied", "exact"):
            agg[k] += s[k]
        rows.append({"id": ex["id"], "ops": ex["ops"], "voice": ex.get("voice"),
                     "completion_tokens": len(seq.tokens), "response": text, **s})

    n = len(examples)
    name = args.out or f"eval_{args.split}_{args.mode}_{'tuned' if args.model_path else 'base'}"
    out = common.PROJ / "out" / f"{name}.json"
    out.write_text(json.dumps({"n": n, "model_path": args.model_path,
                               "mode": args.mode, "agg": agg, "rows": rows}, indent=1))
    print(f"n={n} parsed={agg['parsed']} applied={agg['applied']} "
          f"exact={agg['exact']} ({100 * agg['exact'] / max(n, 1):.1f}%) failed={agg['failed']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
