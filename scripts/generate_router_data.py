"""Generate a labelled question set for training the router classifier.

Labels are reliable *by construction*: a factual template only ever produces a factual
question, and so on. That side-steps the need to hand-label a pile of un-categorised
questions — but it is not free of bias, so:

  - Each category uses several templates with varied phrasing, and a block of hand-written
    questions that deliberately DROP the obvious keyword (a comparative without "compare", a
    multi-hop without "and how long"), so the classifier learns intent, not surface cues.
  - The domain mirrors the app's own examples (healthcare / insurance policy) plus some
    general-knowledge questions, so it isn't overfit to one vocabulary.

Reproducible: fixed seed, deterministic output. Re-run to regenerate byte-for-byte.
A human spot-check before citing accuracy is still worth doing — see data/README.

Usage:  python scripts/generate_router_data.py   ->   data/router_questions.jsonl
"""

import json
import random
from pathlib import Path

SEED = 20260720
TARGET_PER_CATEGORY = 85

DRUGS = ["Humira", "Ozempic", "buprenorphine", "Drug A", "Drug B", "metformin",
         "atorvastatin", "Keytruda", "insulin glargine", "Wegovy", "Adderall", "Xarelto"]
PLANS = ["the PPO plan", "the HMO plan", "the Silver plan", "the Gold plan",
         "the high-deductible plan", "Plan A", "Plan B", "the family plan"]
CONDITIONS = ["type 2 diabetes", "rheumatoid arthritis", "opioid use disorder",
              "hypertension", "high cholesterol", "obesity", "atrial fibrillation"]
SERVICES = ["an MRI", "a specialist visit", "physical therapy", "an emergency room visit",
            "a annual physical", "outpatient surgery", "a telehealth consult"]
ATTRS = ["deductible", "copay", "coinsurance", "out-of-pocket maximum", "monthly premium",
         "coverage limit", "prior authorization requirement"]
COMPANIES = ["Pfizer", "Novo Nordisk", "AbbVie", "Merck", "the manufacturer"]
TOPICS = ["prior authorization", "step therapy", "the appeals process", "network coverage",
          "the formulary tier", "specialty pharmacy requirements"]

# General-knowledge slots, so the set isn't purely insurance vocabulary.
PEOPLE = ["Christopher Nolan", "Marie Curie", "Alan Turing", "Ada Lovelace"]
WORKS = ["Inception", "the theory of relativity", "the first computer program"]
PLACES = ["France", "the headquarters", "the head office"]


def _u(xs):  # unique-preserving sample helper
    return random.choice(xs)


def gen_factual():
    templates = [
        lambda: f"What is the {_u(ATTRS)} for {_u(PLANS)}?",
        lambda: f"What ICD-10 code is used for {_u(CONDITIONS)}?",
        lambda: f"When does prior authorization for {_u(DRUGS)} expire?",
        lambda: f"Who manufactures {_u(DRUGS)}?",
        lambda: f"How many days does authorization for {_u(DRUGS)} last?",
        lambda: f"Is {_u(DRUGS)} covered under {_u(PLANS)}?",
        lambda: f"What is the dosage of {_u(DRUGS)}?",
        lambda: f"What is the copay for {_u(SERVICES)}?",
        lambda: f"Where is {_u(PEOPLE)} based?",
        lambda: f"Who directed {_u(WORKS)}?",
        lambda: f"What does {_u(PLANS)} say about {_u(TOPICS)}?",
        lambda: f"What year did {_u(PEOPLE)} publish {_u(WORKS)}?",
    ]
    return templates


def gen_comparative():
    templates = [
        lambda: f"Compare coverage for {_u(DRUGS)} and {_u(DRUGS)}.",
        lambda: f"What is the difference between {_u(PLANS)} and {_u(PLANS)}?",
        lambda: f"Which has a higher {_u(ATTRS)}, {_u(PLANS)} or {_u(PLANS)}?",
        lambda: f"How does {_u(DRUGS)} differ from {_u(DRUGS)} in cost?",
        lambda: f"Contrast the authorization requirements for {_u(DRUGS)} versus {_u(DRUGS)}.",
        lambda: f"Between {_u(PLANS)} and {_u(PLANS)}, which covers {_u(SERVICES)}?",
        # keyword-free comparatives (no "compare"/"difference"/"versus")
        lambda: f"Is {_u(DRUGS)} or {_u(DRUGS)} cheaper?",
        lambda: f"Does {_u(PLANS)} or {_u(PLANS)} have better coverage for {_u(SERVICES)}?",
        lambda: f"Which is more restrictive about {_u(TOPICS)}, {_u(PLANS)} or {_u(PLANS)}?",
    ]
    return templates


def gen_multihop():
    templates = [
        lambda: f"Which drug requires a test dose, and what ICD-10 code applies to its indication?",
        lambda: f"What condition does {_u(DRUGS)} treat, and what is the copay for that condition's specialist?",
        lambda: f"Who manufactures {_u(DRUGS)}, and what other drugs does that company make?",
        lambda: f"What plan covers {_u(SERVICES)}, and what is that plan's {_u(ATTRS)}?",
        lambda: f"If a patient has never taken {_u(DRUGS)}, what steps must they complete, and how long does authorization then last?",
        lambda: f"Which section covers {_u(TOPICS)}, and what does it require for {_u(DRUGS)}?",
        lambda: f"What is the formulary tier for {_u(DRUGS)}, and what copay does that tier carry?",
        # keyword-free multi-hops (no "and how long")
        lambda: f"After finding which plan covers {_u(SERVICES)}, tell me its {_u(ATTRS)}.",
        lambda: f"Identify the drug that treats {_u(CONDITIONS)}, then give its manufacturer.",
    ]
    return templates


def build(category, template_fn, target):
    seen, out = set(), []
    templates = template_fn()
    guard = 0
    while len(out) < target and guard < target * 60:
        guard += 1
        q = random.choice(templates)().replace(" a a", " an").replace("a annual", "an annual")
        if q not in seen:
            seen.add(q)
            out.append({"question": q, "label": category})
    return out


def main():
    random.seed(SEED)
    rows = []
    rows += build("factual", gen_factual, TARGET_PER_CATEGORY)
    rows += build("comparative", gen_comparative, TARGET_PER_CATEGORY)
    rows += build("multihop", gen_multihop, TARGET_PER_CATEGORY)
    random.shuffle(rows)

    out_path = Path(__file__).resolve().parent.parent / "data" / "router_questions.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = {c: sum(1 for r in rows if r["label"] == c) for c in ("factual", "comparative", "multihop")}
    print(f"wrote {len(rows)} questions to {out_path}")
    print("per category:", counts)


if __name__ == "__main__":
    main()
