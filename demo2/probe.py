"""Probe demo2 candidates through the demo server: base vs tuned, 2 voices, 2 trials."""
import base64
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import common  # noqa: E402

common.load_env()
import os  # noqa: E402

HERE = Path(__file__).resolve().parent
WAVS = HERE / "tts"
WAVS.mkdir(exist_ok=True)


def openai_tts(text: str, voice: str, path: Path) -> None:
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/speech",
        data=json.dumps({"model": "tts-1", "input": text, "voice": voice,
                         "response_format": "wav"}).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                 "Content-Type": "application/json"})
    raw = path.with_suffix(".raw.wav")
    raw.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    str(raw), str(path)], check=True, capture_output=True)
    raw.unlink()


def norm(s: str) -> str:
    s = re.sub(r"[ \t]+$", "", s, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def call(doc: str, model: str, wav: Path) -> dict:
    audio = base64.b64encode(wav.read_bytes()).decode()
    req = urllib.request.Request(
        "http://localhost:8322/api/edit",
        data=json.dumps({"doc": doc, "model": model, "audio_b64": audio}).encode(),
        headers={"Content-Type": "application/json"})
    for attempt in range(8):
        try:
            return json.load(urllib.request.urlopen(req, timeout=600))
        except Exception as e:
            print(f"  ({model} attempt {attempt + 1} failed: {e})", flush=True)
            time.sleep(120)
    return {"error": "request failed", "tokens": 0, "ms": 0}


def main() -> None:
    cands = json.load(open(HERE / "candidates.json"))
    for c in cands:
        take = HERE / "takes" / f"{c['id']}.wav"
        variants = [("take", take)] if take.exists() else [("say", None), ("nova", "nova")]
        for vname, src in variants:
            if vname == "take":
                wav = src
            else:
                wav = WAVS / f"{c['id']}_{vname}.wav"
                if not wav.exists():
                    if src:
                        openai_tts(c["instruction"], src, wav)
                    else:
                        common.tts_wav(c["instruction"], "Samantha", wav, 185)
            res = []
            for model in ("base", "tuned"):
                for _ in range(2):
                    r = call(c["doc"], model, wav)
                    ok = (not r.get("error")) and norm(r["applied_doc"]) == norm(c["target"])
                    res.append(f"{model[0]}:{'OK' if ok else 'X'}/{r['tokens']}t/{r['ms'] // 1000}s")
            print(f"{c['id']:9s} {vname:4s} | {' '.join(res)}", flush=True)


if __name__ == "__main__":
    main()
