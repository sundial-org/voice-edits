"""Voice-edit demo: race base Inkling vs the fine-tuned checkpoint.

Run: python demo/server.py  ->  http://localhost:8322
"""
import base64
import difflib
import io
import json
import sys
import time
import wave
from pathlib import Path

DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO.parent / "scripts"))
import common  # noqa: E402

common.load_env()
import tinker  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from tinker import types  # noqa: E402
from tml_renderers import chat  # noqa: E402
from tml_renderers.tinker import token_spans_to_tinker_model_input  # noqa: E402

renderer = common.get_renderer()
sc = tinker.ServiceClient()
ckpt = [l.split("\t") for l in (common.PROJ / "out" / "checkpoints.txt").read_text().splitlines()][-1][1]
CLIENTS = {
    "base": sc.create_sampling_client(base_model=common.MODEL),
    "tuned": sc.create_sampling_client(model_path=ckpt),
}
print(f"tuned checkpoint: {ckpt}")


def _warmup() -> None:
    """First request against a checkpoint pays a cold-start; do it at boot."""
    spans, _ = renderer.render_for_completion([chat.Message(
        content=chat.Text("hi"), author=chat.Author(chat.AuthorKind.User))])
    mi = token_spans_to_tinker_model_input(spans)
    p = types.SamplingParams(max_tokens=8, temperature=0.0, stop=renderer.stop())
    for name, c in CLIENTS.items():
        try:
            c.sample(prompt=mi, sampling_params=p, num_samples=1).result()
            print(f"warmed up: {name}")
        except Exception as e:
            print(f"warmup {name} failed: {e}")


import threading  # noqa: E402

threading.Thread(target=_warmup, daemon=True).start()

app = FastAPI()


class EditRequest(BaseModel):
    doc: str
    model: str  # "base" | "tuned"
    audio_b64: str  # 16kHz mono PCM wav


def diff_lines(old: str, new: str) -> list[dict]:
    out = []
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=1000):
        if line[:3] in ("---", "+++") or line.startswith("@@"):
            continue
        t = {"+": "add", "-": "del"}.get(line[:1], "same")
        out.append({"t": t, "s": line[1:]})
    return out


@app.get("/")
def index():
    return HTMLResponse((DEMO / "index.html").read_text())


V2 = DEMO.parent / "demo2"


@app.get("/v2")
def v2_index():
    return HTMLResponse((V2 / "index.html").read_text())


@app.get("/v2/candidates")
def v2_candidates():
    return json.loads((V2 / "candidates.json").read_text())


@app.get("/v2/audio/{name}")
def v2_audio(name: str):
    for sub in ("takes", "tts"):
        p = (V2 / sub / name).resolve()
        if p.is_relative_to(V2) and p.suffix == ".wav" and p.exists():
            return FileResponse(p, media_type="audio/wav")
    return HTMLResponse(status_code=404, content="not found")


@app.get("/api/samples")
def samples():
    return json.loads((DEMO / "samples.json").read_text())


@app.get("/api/samples/{name}")
def sample_wav(name: str):
    p = (DEMO / "samples" / name).resolve()
    assert p.is_relative_to(DEMO / "samples") and p.suffix == ".wav"
    return FileResponse(p, media_type="audio/wav")


@app.post("/api/edit")
def edit(req: EditRequest):
    raw = base64.b64decode(req.audio_b64)
    with wave.open(io.BytesIO(raw)) as w:
        nf, sr = w.getnframes(), w.getframerate()
    tmp = DEMO / "last_input.wav"
    tmp.write_bytes(raw)

    msgs = common.build_messages(req.doc, wav_path=tmp)
    spans, parser = renderer.render_for_completion(msgs)
    t0 = time.time()
    out = CLIENTS[req.model].sample(
        prompt=token_spans_to_tinker_model_input(spans),
        sampling_params=types.SamplingParams(max_tokens=1200, temperature=0.0,
                                             stop=renderer.stop()),
        num_samples=1).result()
    ms = int((time.time() - t0) * 1000)
    seq = out.sequences[0]
    thinking, text = [], []
    for m in parser.parse_tokens(seq.tokens):
        if isinstance(m.content, chat.Thinking):
            thinking.append(m.content.text)
        elif isinstance(m.content, chat.Text):
            text.append(m.content.text)
    response = "\n".join(text)
    edits = common.parse_edits(response)
    applied, err = common.apply_edits(req.doc, edits)
    return {
        "model": req.model, "ms": ms, "tokens": len(seq.tokens),
        "thinking": "\n".join(thinking), "response": response, "error": err,
        "applied_doc": None if err else applied,
        "diff": None if err else diff_lines(req.doc, applied),
        "audio_seconds": round(nf / sr, 1),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8322)
