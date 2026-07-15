"""Programmatic (instruction, ground-truth edit) pairs from the doc corpus."""
import json
import random
import re
import sys

import common

WORDS = re.compile(r"[A-Za-z][A-Za-z'-]{3,}")
TITLES = ["Roadmap", "Next Steps", "Open Questions", "Key Risks", "Background",
          "Action Items", "Highlights", "Priorities", "Summary", "Details"]
NAMES = ["Jordan", "Priya", "Marcus", "Elena", "Sam", "Yuki", "Dana", "Omar"]
TYPOS = {"the": "teh", "with": "wtih", "receive": "recieve", "their": "thier",
         "schedule": "schedual", "separate": "seperate", "which": "wich"}


def sections(doc: str) -> list[dict]:
    lines = doc.splitlines()
    idxs = [i for i, ln in enumerate(lines) if ln.startswith("## ")]
    out = []
    for j, i in enumerate(idxs):
        end = idxs[j + 1] if j + 1 < len(idxs) else len(lines)
        out.append({"title": lines[i][3:].strip(), "head": lines[i],
                    "body": lines[i + 1:end]})
    return out


def bullets(sec: dict) -> list[str]:
    return [ln for ln in sec["body"] if ln.startswith("- ")]


def block_sentences(sec: dict) -> list[str]:
    text = " ".join(ln for ln in sec["body"] if ln and not ln.startswith(("- ", "#")))
    parts = re.split(r"(?<=[.!?]) +", text)
    return [p for p in parts if len(p.split()) >= 6 and p.endswith(".")]


def unique(doc: str, s: str) -> bool:
    return bool(s) and doc.count(s) == 1


def keyword(sentence: str) -> str:
    ws = sorted(WORDS.findall(sentence), key=len, reverse=True)
    return ws[0].lower() if ws else ""


ORD = ["first", "second", "third", "fourth", "fifth", "sixth"]

# --- edit ops: each returns (edits, train_phrase, test_phrase) or None ---


def op_rename_heading(doc, secs, rng):
    sec = rng.choice(secs)
    new = rng.choice([t for t in TITLES if t.lower() != sec["title"].lower()])
    if not unique(doc, sec["head"]):
        return None
    edits = [(sec["head"], f"## {new}")]
    t = sec["title"]
    train = rng.choice([
        f"Can you rename the {t} section to {new}?",
        f"Change the heading {t} to {new}.",
        f"Let's call the {t} section {new} instead.",
        f"Retitle {t} as {new}.",
    ])
    test = rng.choice([
        f"The section named {t} should be titled {new} now.",
        f"Swap out the {t} header for {new}.",
        f"I want {t} renamed to {new}.",
    ])
    return edits, train, test


def op_delete_sentence(doc, secs, rng):
    cands = [(sec, s) for sec in secs for s in block_sentences(sec) if unique(doc, s)]
    if not cands:
        return None
    sec, s = rng.choice(cands)
    pre = " " + s if unique(doc, " " + s) else s
    edits = [(pre, "")]
    kw, t, start = keyword(s), sec["title"], " ".join(s.split()[:4])
    train = rng.choice([
        f"In the {t} section, delete the sentence about {kw}.",
        f"Remove the sentence mentioning {kw} from {t}.",
        f"Take out the sentence that starts with {start}.",
        f"Drop the {kw} sentence in {t}.",
    ])
    test = rng.choice([
        f"There's a sentence about {kw} in {t}, get rid of it.",
        f"Cut the sentence beginning {start}.",
        f"We don't need the sentence on {kw} in {t}, remove it please.",
    ])
    return edits, train, test


def op_delete_bullet(doc, secs, rng):
    cands = [(sec, i, b) for sec in secs for i, b in enumerate(bullets(sec))
             if unique(doc, b + "\n")]
    if not cands:
        return None
    sec, i, b = rng.choice(cands)
    if i >= len(ORD):
        return None
    edits = [(b + "\n", "")]
    t, kw = sec["title"], keyword(b)
    train = rng.choice([
        f"Delete the {ORD[i]} bullet in the {t} list.",
        f"Remove the bullet point about {kw} under {t}.",
        f"Take the {ORD[i]} item out of the {t} list.",
    ])
    test = rng.choice([
        f"The {ORD[i]} bullet under {t} can go.",
        f"Scrap the list item about {kw} in {t}.",
    ])
    return edits, train, test


def op_delete_section(doc, secs, rng):
    if len(secs) < 4:
        return None
    sec = rng.choice(secs[1:])
    block = sec["head"] + "\n" + "\n".join(sec["body"])
    if not unique(doc, block):
        return None
    t = sec["title"]
    edits = [(block, "")]
    train = rng.choice([
        f"Delete the whole {t} section.",
        f"Remove the {t} section entirely.",
        f"We're cutting {t}, take that section out.",
    ])
    test = rng.choice([
        f"Kill the {t} section altogether.",
        f"The entire {t} section should be removed.",
    ])
    return edits, train, test


def _mid_sentence_cap(doc: str, w: str) -> bool:
    """True if w's single occurrence is a standalone word not opening a sentence/line."""
    if doc.count(w) != 1 or len(re.findall(rf"\b{w}\b", doc)) != 1:
        return False
    i = doc.index(w)
    return i > 2 and doc[i - 1] == " " and doc[i - 2] not in ".!?:-"


def op_rename_entity(doc, secs, rng):
    caps = [w for w in set(re.findall(r"\b[A-Z][a-z]{3,}\b", doc))
            if w not in TITLES and _mid_sentence_cap(doc, w)]
    if not caps:
        return None
    old = rng.choice(caps)
    new = rng.choice([n for n in NAMES if n != old])
    edits = [(old, new)]
    train = rng.choice([
        f"Replace {old} with {new} in the doc.",
        f"Change {old} to {new}.",
        f"It's not {old} anymore, it's {new}. Update that.",
    ])
    test = rng.choice([
        f"Swap {old} out for {new}.",
        f"Everywhere it says {old}, it should say {new}.",
    ])
    return edits, train, test


def op_change_number(doc, secs, rng):
    nums = [n for n in set(re.findall(r"\b\d{1,4}\b", doc))
            if doc.count(n) == 1 and len(re.findall(rf"\b{n}\b", doc)) == 1]
    if not nums:
        return None
    old = rng.choice(nums)
    new = str(int(old) + rng.choice([1, 2, 5, 10]))
    edits = [(old, new)]
    train = rng.choice([
        f"Change the number {old} to {new}.",
        f"That figure {old} is outdated, make it {new}.",
        f"Update {old} to {new} please.",
    ])
    test = rng.choice([
        f"Wherever it says {old}, correct it to {new}.",
        f"The {old} should actually be {new}.",
    ])
    return edits, train, test


def op_insert_bullet(doc, secs, rng):
    cands = [sec for sec in secs if bullets(sec)]
    if not cands:
        return None
    sec = rng.choice(cands)
    last = bullets(sec)[-1]
    if not unique(doc, last):
        return None
    content = rng.choice([
        "circulate the summary to stakeholders",
        "confirm the budget with finance",
        "schedule a follow-up review",
        "document the rollback procedure",
        "collect feedback from the pilot group",
    ])
    edits = [(last, f"{last}\n- {rng.choice([content.capitalize(), content])}")]
    t = sec["title"]
    train = rng.choice([
        f"Add a bullet to the {t} list that says {content}.",
        f"In {t}, append a new bullet: {content}.",
        f"Put another item in the {t} list, {content}.",
    ])
    test = rng.choice([
        f"One more bullet for {t}: {content}.",
        f"Tack on a bullet under {t} saying {content}.",
    ])
    return edits, train, test


def op_move_bullet(doc, secs, rng):
    cands = [sec for sec in secs if len(bullets(sec)) >= 3]
    if not cands:
        return None
    sec = rng.choice(cands)
    bl = bullets(sec)
    last, first = bl[-1], bl[0]
    if not (unique(doc, last + "\n") or unique(doc, "\n" + last)) or not unique(doc, first):
        return None
    if not unique(doc, "\n" + last):
        return None
    edits = [("\n" + last, ""), (first, last + "\n" + first)]
    t = sec["title"]
    train = rng.choice([
        f"Move the last bullet in {t} to the top of the list.",
        f"In the {t} list, make the final bullet the first one.",
    ])
    test = rng.choice([
        f"The last item under {t} should lead the list, move it up.",
    ])
    return edits, train, test


def op_fix_typo(doc, secs, rng):
    words = [(good, bad) for good, bad in TYPOS.items()
             if doc.count(f" {good} ") == 1 and bad not in doc]
    if not words:
        return None
    good, bad = rng.choice(words)
    planted = doc.replace(f" {good} ", f" {bad} ", 1)
    edits = [(f" {bad} ", f" {good} ")]
    train = rng.choice([
        f"There's a typo, {bad} should be {good}. Fix it.",
        f"Fix the misspelling of {good}.",
        "Someone made a spelling mistake in there, find it and fix it.",
    ])
    test = rng.choice([
        f"Correct the typo where it says {bad}.",
        "There's one misspelled word in the doc, please fix it.",
    ])
    return (edits, train, test), planted


OPS = [op_rename_heading, op_delete_sentence, op_delete_bullet, op_delete_section,
       op_rename_entity, op_change_number, op_insert_bullet, op_move_bullet,
       op_fix_typo]

PRE_TRAIN = ["", "Hey, ", "Okay so, ", "Quick one: ", "Um, ", "Alright, "]
PRE_TEST = ["", "Hi, ", "So, ", "One thing: ", "Right, "]
POST_TRAIN = ["", " Thanks!", " That's it.", " Cheers."]
POST_TEST = ["", " Thank you.", " Appreciate it."]


def make_example(doc, rng, n_ops):
    secs = sections(doc)
    if len(secs) < 3:
        return None
    chosen = rng.sample(OPS, k=min(n_ops * 3, len(OPS)))
    edits, trains, tests, ops = [], [], [], []
    for op in chosen:
        if len(ops) == n_ops:
            break
        res = op(doc, secs, rng)
        if res is None:
            continue
        if op is op_fix_typo:
            res, doc = res
            secs = sections(doc)
        e, tr, te = res
        # all SEARCH strings must still be unique in current doc and non-overlapping
        if any(not unique(doc, s) for s, _ in e if s):
            continue
        edits += e
        trains.append(tr)
        tests.append(te)
        ops.append(op.__name__[3:])
    if len(ops) < n_ops:
        return None
    new_doc, err = common.apply_edits(doc, edits)
    if err:
        return None
    joiner = rng.choice([" And also, ", " Oh and ", " Then "])
    return {
        "doc": doc,
        "target": common.normalize(new_doc),
        "edits": edits,
        "ops": ops,
        "instr_train": (rng.choice(PRE_TRAIN) + joiner.join(trains) + rng.choice(POST_TRAIN)).strip(),
        "instr_test": (rng.choice(PRE_TEST) + joiner.join(tests) + rng.choice(POST_TEST)).strip(),
    }


def main(per_doc: int = 4) -> None:
    rng = random.Random(1)
    docs = sorted((common.DATA / "docs").glob("doc_*.md"))
    print(f"{len(docs)} docs")
    rng.shuffle(docs)
    n_test = max(1, len(docs) // 10)
    split_of = {}
    for i, p in enumerate(docs):
        split_of[p.name] = "test" if i < n_test else ("val" if i < 2 * n_test else "train")

    examples = []
    for p in docs:
        doc = common.normalize(p.read_text())
        split = split_of[p.name]
        made = 0
        for k in range(per_doc * 2):
            if made == per_doc:
                break
            # 25% multi-edit examples
            n_ops = 2 if rng.random() < 0.25 else 1
            ex = make_example(doc, rng, n_ops)
            if ex is None:
                continue
            ex.update(id=f"{p.stem}_e{made}", doc_file=p.name, split=split,
                      instruction=ex.pop("instr_test" if split == "test" else "instr_train"))
            ex.pop("instr_train", None)
            ex.pop("instr_test", None)
            examples.append(ex)
            made += 1

    out = common.DATA / "examples.jsonl"
    with out.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    from collections import Counter
    print(f"{len(examples)} examples -> {out}")
    print("splits:", Counter(e["split"] for e in examples))
    print("ops:", Counter(o for e in examples for o in e["ops"]))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
