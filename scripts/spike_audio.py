"""Feasibility spike: voice instruction + markdown doc -> model edit, end to end."""
import subprocess
import sys
import wave
from pathlib import Path

import common

common.load_env()
import tinker  # noqa: E402
from tinker import types  # noqa: E402
from tml_renderers import chat, tokenizers, v0  # noqa: E402
from tml_renderers.tinker import token_spans_to_tinker_model_input  # noqa: E402

OUT = common.PROJ / "out"
OUT.mkdir(parents=True, exist_ok=True)

INSTRUCTION = (
    "Hey, in the roadmap doc, can you rename the section called Future Work "
    "to just Roadmap, and delete the sentence about mobile support? Thanks."
)

DOC = """# Sundial Sync Engine

## Overview
The sync engine keeps local and cloud replicas consistent using CRDTs.

## Future Work
We plan to add offline conflict visualization. We are also exploring mobile support for the editor.
"""

SYSTEM = """You are a voice-driven document editor. You receive a markdown document and a spoken instruction as audio. Reply ONLY with edit operations in this exact format, one or more blocks:

<edit>
<<<<<<< SEARCH
(exact text from the document)
=======
(replacement text)
>>>>>>> REPLACE
</edit>

Use an empty replacement to delete text. The SEARCH text must match the document exactly."""


def tts_wav(text: str, voice: str, path: Path, rate: int = 180) -> tuple[int, int]:
    aiff = path.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(path)],
        check=True,
    )
    aiff.unlink()
    with wave.open(str(path)) as w:
        return w.getnframes(), w.getframerate()


def main() -> None:
    wav = OUT / "spike_instruction.wav"
    num_frames, sample_rate = tts_wav(INSTRUCTION, "Samantha", wav)
    print(f"TTS ok: {wav.name} {num_frames} frames @ {sample_rate} Hz")

    messages = [
        chat.Message(content=chat.Text(SYSTEM), author=chat.Author(chat.AuthorKind.System)),
        chat.Message(
            content=chat.Text(f"Document:\n\n{DOC}"),
            author=chat.Author(chat.AuthorKind.User),
        ),
        chat.Message(
            content=chat.AudioPointer(
                location=str(wav), format=chat.AudioFormat.Wav,
                num_frames=num_frames, sample_rate=sample_rate,
            ),
            author=chat.Author(chat.AuthorKind.User),
        ),
    ]

    renderer = v0.Renderer(tokenizers.o200k_base_chat())
    spans, parser = renderer.render_for_completion(messages)
    model_input = token_spans_to_tinker_model_input(spans)
    print(f"Rendered prompt: {model_input.length} tokens (audio included)")

    sampler = tinker.ServiceClient().create_sampling_client(
        base_model=common.MODEL
    )
    out = sampler.sample(
        prompt=model_input,
        sampling_params=types.SamplingParams(
            max_tokens=600, temperature=0.0, stop=renderer.stop()
        ),
        num_samples=1,
    ).result()
    seq = out.sequences[0]
    print(f"Sampled {len(seq.tokens)} tokens")
    for m in parser.parse_tokens(seq.tokens):
        c = m.content
        if isinstance(c, chat.Thinking):
            print(f"--- thinking ---\n{c.text}")
        elif isinstance(c, chat.Text):
            print(f"--- text ---\n{c.text}")


if __name__ == "__main__":
    sys.exit(main())
