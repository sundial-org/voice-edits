"""Ping Tinker sampling until it responds quickly; print one line per state change."""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import common

common.load_env()
import tinker  # noqa: E402
from tinker import types  # noqa: E402
from tml_renderers import chat  # noqa: E402
from tml_renderers.tinker import token_spans_to_tinker_model_input  # noqa: E402

renderer = common.get_renderer()
spans, _ = renderer.render_for_completion([chat.Message(
    content=chat.Text("hi"), author=chat.Author(chat.AuthorKind.User))])
mi = token_spans_to_tinker_model_input(spans)
params = types.SamplingParams(max_tokens=4, temperature=0.0, stop=renderer.stop())
sampler = tinker.ServiceClient().create_sampling_client(base_model=common.MODEL)

last = None
while True:
    t0, out = time.time(), {}

    def ping():
        try:
            sampler.sample(prompt=mi, sampling_params=params, num_samples=1).result()
            out["ok"] = True
        except Exception as e:
            out["err"] = str(e)[:80]

    th = threading.Thread(target=ping, daemon=True)
    th.start()
    th.join(60)
    state = f"HEALTHY {time.time() - t0:.1f}s" if out.get("ok") else (
        f"ERROR {out['err']}" if "err" in out else "STALLED >60s")
    if state.split()[0] != (last.split()[0] if last else None):
        print(state, flush=True)
    last = state
    if out.get("ok"):
        break
    time.sleep(120)
