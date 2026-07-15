"""TTS every example's instruction. Train and test use disjoint voices."""
import json
import random

import common

TRAIN_VOICES = ["Samantha", "Daniel", "Karen", "Rishi", "Tessa", "Aman",
                "Flo (English (US))", "Reed (English (UK))", "Sandy (English (US))"]
TEST_VOICES = ["Moira", "Tara", "Shelley (English (UK))", "Rocko (English (US))",
               "Grandma (English (US))"]


def main() -> None:
    rng = random.Random(2)
    wav_dir = common.DATA / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    path = common.DATA / "examples.jsonl"
    examples = [json.loads(l) for l in path.read_text().splitlines()]
    for i, ex in enumerate(examples):
        voices = TEST_VOICES if ex["split"] == "test" else TRAIN_VOICES
        voice = rng.choice(voices)
        rate = rng.randint(150, 215)
        wav = wav_dir / f"{ex['id']}.wav"
        if not wav.exists():
            common.tts_wav(ex["instruction"], voice, wav, rate)
        ex["wav"], ex["voice"], ex["rate"] = wav.name, voice, rate
        if i % 100 == 0:
            print(f"{i}/{len(examples)}")
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"done: {len(examples)} wavs in {wav_dir}")


if __name__ == "__main__":
    main()
