"""Generate the native rollback benchmark cases (50 contradictions + 50 clean).

Output: data/rollback_cases_100.tsv
Columns: id, type, template, source, y1, y2, e1, e2
No third-party dependencies; runs on Windows Python.
"""
import random

TEMPLATES = [
    dict(key="born_graduation", e1="the person was born", e2="graduated from high school",
         f1="{name} was born in {y1}.", f2="{name} graduated from high school in {y2}.",
         y1=(1870, 1970), cd=(2, 10), kd=(18, 25), rel="gt",
         k1="born", k2="graduat"),
    dict(key="founded_retired", e1="the person founded the company", e2="retired from the company",
         f1="{name} founded the company in {y1}.", f2="{name} retired from the company in {y2}.",
         y1=(1930, 1985), cd=(2, 15), kd=(10, 35), rel="gt",
         k1="found", k2="retir"),
    dict(key="prize_death", e1="the person won the national prize", e2="died",
         f1="{name} won the national prize in {y1}.", f2="{name} died in {y2}.",
         y1=(1940, 2005), cd=(1, 8), kd=(1, 15), rel="gt",
         k1="prize", k2="died"),
    dict(key="book_death", e1="the person published the book", e2="died",
         f1="{name} published the book in {y1}.", f2="{name} died in {y2}.",
         y1=(1940, 1990), cd=(1, 8), kd=(1, 15), rel="gt",
         k1="publish", k2="died"),
    dict(key="married_birth", e1="the person married", e2="was born",
         f1="{name} married in {y1}.", f2="{name} was born in {y2}.",
         y1=(1930, 1975), cd=(18, 35), kd=(18, 35), rel="lt",
         k1="marri", k2="born"),
    dict(key="school_graduation", e1="the person started school", e2="graduated",
         f1="{name} started school in {y1}.", f2="{name} graduated in {y2}.",
         y1=(1930, 1980), cd=(2, 12), kd=(4, 12), rel="gt",
         k1="school", k2="graduat"),
    dict(key="factory_sold", e1="the person built the factory", e2="sold it",
         f1="{name} built the factory in {y1}.", f2="{name} sold it in {y2}.",
         y1=(1900, 1970), cd=(2, 15), kd=(5, 30), rel="gt",
         k1="built", k2="sold"),
    dict(key="joined_left", e1="the person joined the company", e2="left the company",
         f1="{name} joined the company in {y1}.", f2="{name} left the company in {y2}.",
         y1=(1940, 1985), cd=(2, 15), kd=(5, 30), rel="gt",
         k1="join", k2="left"),
    dict(key="wrote_death", e1="the person wrote the book", e2="died",
         f1="{name} wrote the book in {y1}.", f2="{name} died in {y2}.",
         y1=(1930, 1980), cd=(1, 10), kd=(1, 20), rel="gt",
         k1="wrote", k2="died"),
    dict(key="award_birth", e1="the person received the award", e2="was born",
         f1="{name} received the award in {y1}.", f2="{name} was born in {y2}.",
         y1=(1940, 1985), cd=(1, 25), kd=(1, 25), rel="lt",
         k1="award", k2="born"),
]

FIRST = ["Aaron", "Bella", "Carlos", "Diana", "Elias", "Fiona", "Gabriel", "Hana",
         "Ibrahim", "Julia", "Kai", "Lena", "Mateo", "Nadia", "Omar", "Priya",
         "Quentin", "Rosa", "Samir", "Tara", "Umar", "Vera", "Wei", "Xenia",
         "Yusuf", "Zara", "Adrian", "Beatriz", "Cyrus", "Delia", "Emil", "Farah",
         "Gustav", "Helena", "Ivan", "Jasmine", "Karim", "Laura", "Mikhail", "Nora",
         "Oscar", "Paloma", "Rafael", "Sofia", "Tomas", "Uma", "Victor", "Wendy",
         "Xavier", "Yara"]
LAST = ["Achterberg", "Bianchi", "Chen", "Dubois", "Eriksen", "Fischer", "Garcia",
        "Haddad", "Ivanov", "Jensen", "Kowalski", "Lopez", "Moreau", "Nakamura",
        "Okafor", "Petrov", "Quinn", "Rossi", "Silva", "Tanaka"]
THIRD = ["The rest of the record describes a long and varied career.",
         "Contemporary newspapers covered the later years in some detail.",
         "The remaining documents are preserved in the local archive.",
         "Friends described the later period as quiet and productive.",
         "The later part of the story is less well documented."]


def build(seed=1234, n_contra=50, n_clean=50):
    rng = random.Random(seed)
    names = [f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(names)
    rows = []
    idx = 0
    for kind, n in (("contradiction", n_contra), ("clean", n_clean)):
        for i in range(n):
            tpl = TEMPLATES[i % len(TEMPLATES)]
            name = names[idx]
            idx += 1
            for _ in range(200):
                y1 = rng.randint(*tpl["y1"])
                d = rng.randint(*(tpl["cd"] if kind == "contradiction" else tpl["kd"]))
                if tpl["rel"] == "gt":
                    y2 = y1 - d if kind == "contradiction" else y1 + d
                else:
                    y2 = y1 + d if kind == "contradiction" else y1 - d
                if 1000 <= y2 <= 2025 and y1 != y2:
                    break
            source = (tpl["f1"].format(name=name, y1=y1, y2=y2) + " "
                      + tpl["f2"].format(name=name, y1=y1, y2=y2) + " "
                      + rng.choice(THIRD))
            rows.append((f"{'c' if kind == 'contradiction' else 'k'}{i + 1:03d}",
                         kind, tpl["key"], source, str(y1), str(y2),
                         tpl["e1"], tpl["e2"]))
    return rows


if __name__ == "__main__":
    rows = build()
    with open("data/rollback_cases_100.tsv", "w", encoding="utf-8", newline="") as f:
        f.write("id\ttype\ttemplate\tsource\ty1\ty2\te1\te2\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print("wrote", len(rows), "cases -> data/rollback_cases_100.tsv")
