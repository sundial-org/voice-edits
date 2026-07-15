"""Re-synthesize a share of train/val and ALL test wavs with OpenAI TTS.

Train/val use {alloy, echo, fable, onyx}; test uses held-out {nova, shimmer}.
Runs after tts_batch.py; overwrites selected wavs and updates examples.jsonl.
"""
import concurrent.futures as cf
import json
import os
import random
import subprocess
import urllib.request
from pathlib import Path

import common

TRAIN_VOICES = ["alloy", "echo", "fable", "onyx"]
TEST_VOICES = ["nova", "shimmer"]
TRAIN_SHARE = 0.4  # fraction of train/val re-done with OpenAI voices


def api_key() -> str:
    common.load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("set OPENAI_API_KEY (see .env.example)")
    return key


def synth(key: str, text: str, voice: str, speed: float, wav: Path) -> None:
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/speech",
        data=json.dumps({"model": "tts-1", "input": text, "voice": voice,
                         "speed": speed, "response_format": "wav"}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    raw = wav.with_suffix(".raw.wav")
    with urllib.request.urlopen(req, timeout=120) as r:
        raw.write_bytes(r.read())
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(raw), str(wav)],
        check=True, capture_output=True)
    raw.unlink()


def main() -> None:
    rng = random.Random(4)
    key = api_key()
    path = common.DATA / "examples.jsonl"
    examples = [json.loads(l) for l in path.read_text().splitlines()]

    jobs = []
    for ex in examples:
        if ex["split"] == "test":
            voice = rng.choice(TEST_VOICES)
        elif rng.random() < TRAIN_SHARE:
            voice = rng.choice(TRAIN_VOICES)
        else:
            continue
        speed = round(rng.uniform(0.85, 1.2), 2)
        ex["voice"], ex["rate"] = f"openai:{voice}", speed
        jobs.append((ex, voice, speed))

    print(f"{len(jobs)} wavs via OpenAI TTS")
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(synth, key, ex["instruction"], v, s,
                            common.DATA / "wav" / ex["wav"]): ex for ex, v, s in jobs}
        done = 0
        for fut in cf.as_completed(futs):
            fut.result()
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(jobs)}")

    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print("done")


if __name__ == "__main__":
    main()
