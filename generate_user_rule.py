import itertools
import json
import random

rules = [
    "An expert repeatedly asks you questions",
    "The question is unrelated to the current topic",
    "It interrupts your expression",
    "The question is impolite or overly complicated",
    "The timing is inappropriate or conflicts with the context",
    "The question is too complex to answer",
    "The question involves private or sensitive information",
    "You don't trust the other party yet and don't want to answer",
]


def format_combination(combo):
    lines = []
    n = len(combo)
    for i, item in enumerate(combo):
        if i < n - 1:
            lines.append(f"   - {item};\n")
        else:
            lines.append(f"   - {item}.\n")
    return "".join(lines)


groups = {}
for k in range(1, 8):
    combs = [format_combination(c) for c in itertools.combinations(rules, k)]
    groups[k] = combs

random.seed(42)
for k in groups:
    random.shuffle(groups[k])

train = []
eval_ = []
leftovers = []

for k in range(1, 8):
    items = groups[k]
    n = len(items)
    half = n // 2
    train.extend(items[:half])
    eval_.extend(items[half : half * 2])
    if n % 2 == 1:
        leftovers.append(items[half * 2])

for item in leftovers:
    if len(train) <= len(eval_):
        train.append(item)
    else:
        eval_.append(item)

assert len(train) + len(eval_) == 254, f"The total should be 254, current total={len(train) + len(eval_)}"
assert len(train) == 127 and len(eval_) == 127, f"Train and eval should both be 127, current train={len(train)}, eval={len(eval_)}"

output = {"train": [], "eval": []}
for it in train:
    output["train"].append(it)
for it in eval_:
    output["eval"].append(it)

with open("rules_split.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Total combinations: {len(train) + len(eval_)}")
print(f"Train: {len(train)}, Eval: {len(eval_)}")
