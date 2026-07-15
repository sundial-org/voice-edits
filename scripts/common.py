"""Shared: env, renderer, edit-block format, apply/score, TTS."""
import os
import re
import subprocess
import wave
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]  # repo root
DATA = PROJ / "data"
MODEL = "thinkingmachines/Inkling"

SYSTEM = """You are a voice-driven document editor. You receive a markdown document and a spoken instruction as audio. Reply ONLY with edit operations in this exact format, one or more blocks:

<edit>
<<<<<<< SEARCH
(exact text from the document)
=======
(replacement text)
>>>>>>> REPLACE
</edit>

Use an empty replacement to delete text. The SEARCH text must match the document exactly and unambiguously."""


def load_env() -> None:
    for env in (PROJ / ".env", PROJ.parent / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip() and not line.startswith("#"):
                    k, _, val = line.partition("=")
                    os.environ.setdefault(k.strip(), val.strip())
    if os.environ.get("TML_KEY"):
        os.environ.setdefault("TINKER_API_KEY", os.environ["TML_KEY"])


def get_renderer():
    from tml_renderers import tokenizers, v0

    return v0.Renderer(tokenizers.o200k_base_chat())


EDIT_RE = re.compile(
    r"<edit>\s*<<<<<<< SEARCH\n(.*?)\n?=======\n?(.*?)>>>>>>> REPLACE\s*</edit>",
    re.DOTALL,
)


def format_edits(edits: list[tuple[str, str]]) -> str:
    blocks = []
    for search, replace in edits:
        rep = replace + "\n" if replace else ""
        blocks.append(
            f"<edit>\n<<<<<<< SEARCH\n{search}\n=======\n{rep}>>>>>>> REPLACE\n</edit>"
        )
    return "\n\n".join(blocks)


def parse_edits(text: str) -> list[tuple[str, str]]:
    return [(s, r.rstrip("\n")) for s, r in EDIT_RE.findall(text)]


def apply_edits(doc: str, edits: list[tuple[str, str]]) -> tuple[str, str | None]:
    """Apply blocks in order. Returns (new_doc, error). SEARCH must match exactly once."""
    if not edits:
        return doc, "no edit blocks parsed"
    for search, replace in edits:
        n = doc.count(search)
        if n == 0:
            return doc, f"SEARCH not found: {search[:60]!r}"
        if n > 1:
            return doc, f"SEARCH ambiguous ({n} matches): {search[:60]!r}"
        doc = doc.replace(search, replace, 1)
    return doc, None


def normalize(doc: str) -> str:
    doc = re.sub(r"[ \t]+$", "", doc, flags=re.MULTILINE)
    doc = re.sub(r"\n{3,}", "\n\n", doc)
    return doc.strip() + "\n"


def score(doc: str, response_text: str, target_doc: str) -> dict:
    edits = parse_edits(response_text)
    new_doc, err = apply_edits(doc, edits)
    applied = err is None
    exact = applied and normalize(new_doc) == normalize(target_doc)
    return {"parsed": bool(edits), "applied": applied, "exact": exact, "error": err}


def tts_wav(text: str, voice: str, path: Path, rate: int = 180) -> tuple[int, int]:
    """Synthesize speech to 16kHz mono wav. Returns (num_frames, sample_rate)."""
    aiff = path.with_suffix(".aiff")
    for attempt in range(3):
        try:
            subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text],
                           check=True, timeout=30)
            break
        except subprocess.TimeoutExpired:
            if attempt == 2:
                raise
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(path)],
        check=True, capture_output=True,
    )
    aiff.unlink()
    with wave.open(str(path)) as w:
        return w.getnframes(), w.getframerate()


def build_messages(doc: str, wav_path: Path | None = None, instruction: str | None = None):
    """Prompt messages: system + doc + instruction (audio or text)."""
    from tml_renderers import chat

    msgs = [
        chat.Message(content=chat.Text(SYSTEM), author=chat.Author(chat.AuthorKind.System)),
        chat.Message(
            content=chat.Text(f"Document:\n\n{doc}"),
            author=chat.Author(chat.AuthorKind.User),
        ),
    ]
    if wav_path is not None:
        with wave.open(str(wav_path)) as w:
            nf, sr = w.getnframes(), w.getframerate()
        msgs.append(chat.Message(
            content=chat.AudioPointer(
                location=str(wav_path), format=chat.AudioFormat.Wav,
                num_frames=nf, sample_rate=sr),
            author=chat.Author(chat.AuthorKind.User),
        ))
    else:
        assert instruction is not None
        msgs.append(chat.Message(
            content=chat.Text(instruction), author=chat.Author(chat.AuthorKind.User)))
    return msgs
