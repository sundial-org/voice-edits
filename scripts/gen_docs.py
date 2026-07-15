"""Generate a diverse markdown doc corpus with the model itself."""
import random
import re
import sys

import common

common.load_env()
import tinker  # noqa: E402
from tinker import types  # noqa: E402
from tml_renderers import chat  # noqa: E402
from tml_renderers.tinker import token_spans_to_tinker_model_input  # noqa: E402

GENRES = [
    "product requirements document", "project README", "team meeting notes",
    "technical design doc", "blog post draft", "research notes",
    "incident postmortem", "onboarding guide", "internal FAQ", "changelog",
    "weekly status update", "grant proposal", "event plan", "lecture notes",
]
DOMAINS = [
    "a developer-tools startup", "a climate-tech company", "a biotech lab",
    "an e-commerce platform", "a robotics team", "a fintech app",
    "an edtech nonprofit", "an indie game studio", "a hospital IT group",
    "a logistics company", "a music-production collective", "an astronomy club",
    "a coffee roastery", "a city transit agency", "a cybersecurity vendor",
    "a solar installer",
]

PROMPT = """Write a realistic markdown document: a {genre} for {domain}.
Requirements:
- 250-500 words
- a single `#` title, then 3-6 `##` sections
- at least one bulleted list with 3-6 items
- include at least three specific numbers, dates, or names
- plain markdown only, no code fences around the document
Output only the document."""


def main(n_docs: int = 300) -> None:
    rng = random.Random(0)
    renderer = common.get_renderer()
    sampler = tinker.ServiceClient().create_sampling_client(base_model=common.MODEL)
    docs_dir = common.DATA / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    for i in range(n_docs):
        genre, domain = rng.choice(GENRES), rng.choice(DOMAINS)
        msgs = [chat.Message(
            content=chat.Text(PROMPT.format(genre=genre, domain=domain)),
            author=chat.Author(chat.AuthorKind.User))]
        spans, parser = renderer.render_for_completion_with_effort(msgs, 0.1)
        fut = sampler.sample(
            prompt=token_spans_to_tinker_model_input(spans),
            sampling_params=types.SamplingParams(
                max_tokens=1600, temperature=1.0, stop=renderer.stop()),
            num_samples=1,
        )
        futures.append((i, parser, fut))

    kept = 0
    for i, parser, fut in futures:
        try:
            seq = fut.result().sequences[0]
        except Exception as e:
            print(f"doc {i}: sample failed: {e}")
            continue
        text = "\n".join(
            m.content.text for m in parser.parse_tokens(seq.tokens)
            if isinstance(m.content, chat.Text))
        text = common.normalize(text)
        # quality gate: needs headings and a bullet list
        if len(re.findall(r"^## ", text, re.M)) < 3 or len(re.findall(r"^- ", text, re.M)) < 3:
            print(f"doc {i}: rejected (structure)")
            continue
        (docs_dir / f"doc_{i:04d}.md").write_text(text)
        kept += 1
    print(f"kept {kept}/{n_docs} docs in {docs_dir}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
