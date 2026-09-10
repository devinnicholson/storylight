"""V5 training-only authoring. Never opens independent development or test files."""

import copy
import hashlib
import importlib.metadata
import json
import random
import re
from collections import Counter
from pathlib import Path

import xgrammar as xgr

from storylight.scene_facts import SceneFactsV2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUPPORT = HERE / "support"
PROMPT_SHA = "759a9002a7c6209e377f3ea67a7f706c0055de7ea59de19396d068f5eee47f41"
GRAMMAR_SHA = "622bc2bf36a2182c21a7e8a889f292f9b10a91dd149ee0aa913dd70ad20404df"
ANIMALS = (
    "cat",
    "dog",
    "fox",
    "rabbit",
    "goat",
    "horse",
    "cow",
    "bird",
    "pony",
    "sheep",
    "hen",
    "duck",
    "pig",
    "deer",
    "squirrel",
    "otter",
    "zebra",
    "turtle",
    "owl",
    "seal",
    "penguin",
    "raccoon",
    "hedgehog",
    "swan",
    "lizard",
)
COLORS = (
    "blue",
    "red",
    "green",
    "yellow",
    "white",
    "black",
    "brown",
    "gray",
    "orange",
    "purple",
    "pink",
    "silver",
)
ITEMS = (
    "lantern",
    "ball",
    "box",
    "bell",
    "ribbon",
    "cup",
    "wheel",
    "umbrella",
    "tray",
    "bag",
    "jar",
    "book",
    "coin",
    "vase",
    "bottle",
    "cart",
    "balloon",
    "cushion",
    "cymbal",
    "banjo",
    "harp",
    "pot",
    "stool",
    "plate",
)
PLACES = (
    "forest",
    "field",
    "pasture",
    "yard",
    "marsh",
    "hill",
    "beach",
    "canyon",
    "grove",
    "plaza",
    "desert",
    "cave",
    "tundra",
    "savanna",
    "farm",
    "pond",
    "prairie",
    "glade",
)
RESERVED = [
    "badger",
    "beetle",
    "buffalo",
    "canary",
    "chipmunk",
    "crane",
    "ferret",
    "gazelle",
    "goose",
    "hare",
    "ibis",
    "lemur",
    "lynx",
    "mole",
    "newt",
    "oryx",
    "compass",
    "hammer",
    "basket",
    "teacup",
    "feather",
    "drum",
    "flag",
    "button",
    "rope",
    "acorn",
    "orchard",
    "courtyard",
    "meadow",
    "harbor",
    "clearing",
    "greenhouse",
    "aardvark",
    "alpaca",
    "armadillo",
    "beaver",
    "bison",
    "camel",
    "cheetah",
    "chinchilla",
    "condor",
    "coyote",
    "donkey",
    "egret",
    "flamingo",
    "gecko",
    "gopher",
    "hamster",
    "kite",
    "ladle",
    "scarf",
    "bucket",
    "helmet",
    "notebook",
    "spoon",
    "anchor",
    "rake",
    "whistle",
    "garden",
    "jetty",
    "woodland",
    "terrace",
    "riverbank",
    "workshop",
]
CONTRASTS = (
    "named_actor",
    "named_place",
    "unbound_target",
    "unfinished_correction",
    "clipped_predicate",
    "indistinguishable_roles",
)
SEMANTIC = (
    "temporal_same_actor",
    "temporal_distinct_actors",
    "repeated_actor_actions",
    "without_action",
    "without_object",
    "static_no_action",
    "explicit_target",
    "intransitive_clause",
    "counted_shared_actions",
    "object_attachment",
    "composition",
    "positive_negative_scope",
)
SOURCES = (
    "src/storylight/scene_facts.py",
    "src/storylight/domain.py",
    "src/storylight/privacy_policy.py",
    "src/storylight/semantic_text.py",
)
NUMBER = ("one", "two", "three", "four")


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def plural(word):
    irregular = {
        "pony": "ponies",
        "sheep": "sheep",
        "deer": "deer",
        "fox": "foxes",
        "box": "boxes",
        "goose": "geese",
        "ibis": "ibises",
        "lynx": "lynxes",
    }
    return irregular.get(word, word + "s")


def sentence(text):
    text = re.sub(r"\s+([,;.!?])", r"\1", text)
    return text[0].upper() + text[1:] + ("" if text[-1] in ".!?" else ".")


def entity(ref, label, color=None, count=1, actions=None, attributes=()):
    item = dict(ref=ref, label=label, color=color, count=count, attributes=list(attributes))
    if actions is not None:
        item["actions"] = list(actions)
    return item


def relation(source, kind, target):
    return dict(source=source, relation=kind, target=target)


def context(combo, index):
    animal, other, color, shade, item, place = combo
    count, second_count = index % 4 + 1, (index // 4) % 3 + 1
    label = animal if count == 1 else plural(animal)
    other_label = other if second_count == 1 else plural(other)
    a = f"{NUMBER[count - 1]} {color} {label}"
    b = f"{NUMBER[second_count - 1]} {shade} {other_label}"
    obj = f"one {shade} {item}"
    one = entity("s1", label, color, count, actions=[])
    two = entity("s2", other_label, shade, second_count, actions=[])
    return dict(
        animal=animal,
        other=other,
        color=color,
        shade=shade,
        item=item,
        place=place,
        count=count,
        second_count=second_count,
        label=label,
        other_label=other_label,
        a=a,
        b=b,
        da=f"the {color} {label}",
        db=f"the {shade} {other_label}",
        obj=obj,
        one=one,
        two=two,
        be="is" if count == 1 else "are",
        other_be="is" if second_count == 1 else "are",
        facts={"subjects": [one], "setting": {"label": "unspecified"}},
    )


def verb(base, count):
    return base if count != 1 else {"watch": "watches", "fly": "flies"}.get(base, base + "s")


def positive(family, combo, index, variant):
    c = context(combo, index)
    a, b, da, db, obj = (c[k] for k in ("a", "b", "da", "db", "obj"))
    one, two, facts = c["one"], c["two"], c["facts"]
    count, nc = c["count"], c["second_count"]
    mode = index % 8
    rest = verb("rest", count)
    if family in {"temporal_same_actor", "temporal_distinct_actors"}:
        first_base, last_base = (
            ("walk", "rest"),
            ("run", "sit"),
            ("jump", "sleep"),
            ("stand", "walk"),
            ("swim", "rest"),
            ("crawl", "wait"),
            ("sit", "stand"),
            ("rest", "run"),
        )[mode]
        first = verb(first_base, count)
        same = family == "temporal_same_actor"
        last = verb(last_base, count if same else nc)
        one["actions"] = [first, last] if same else [first]
        target = "s1" if same else "s2"
        if not same:
            two["actions"] = [last]
            facts["subjects"].append(two)
        facts["events"] = [
            dict(ref="e1", source="s1", action=first),
            dict(ref="e2", source=target, action=last),
        ]
        facts["temporal_order"] = [dict(before="e1", after="e2")]
        second = da if same else b
        connectors = (
            ("before", ", then"),
            (", then", "and afterward"),
            ("and then", ", and only afterward"),
            ("before", ", and afterward"),
        )
        link = connectors[mode % 4][variant]
        text = f"{a} {first} {link} {second} {last}"
    elif family == "repeated_actor_actions":
        shared = ("wait", "rest", "stand", "sit")[mode % 4]
        later = verb(("walk", "run", "jump", "sleep")[mode % 4], count if mode < 4 else nc)
        separated = index // 8 % 2 == 1
        first_a = verb(shared, count) if separated else shared
        first_b = verb(shared, nc) if separated else shared
        one["actions"] = [first_a] + ([later] if mode < 4 else [])
        two["actions"] = [first_b] + ([later] if mode >= 4 else [])
        facts["subjects"].append(two)
        later_subject = da if mode < 4 else db
        text = (
            f"{a} and {b} {shared}. {later_subject} {later}"
            if variant == 0
            else f"{b} and {a} {shared}. {later_subject} {later}"
        )
        if separated:
            text = (
                f"{a} {first_a}. {b} {first_b}. {later_subject} {later}"
                if variant == 0
                else f"{b} {first_b}. {a} {first_a}. {later_subject} {later}"
            )
    elif family == "without_action":
        action, negative = (
            ("walk", "running"),
            ("swim", "jumping"),
            ("sit", "dancing"),
            ("stand", "sleeping"),
            ("rest", "walking"),
            ("crawl", "digging"),
            ("run", "jumping"),
            ("wait", "singing"),
        )[mode]
        act = verb(action, count)
        one["actions"] = [act]
        facts["negatives"] = [dict(kind="action", target="s1", value=negative)]
        text = (
            f"{a} {act} without {negative}"
            if variant == 0
            else f"{a} {act}. {da} {c['be']} not {negative}"
        )
        if variant and index // 8 % 2:
            text = f"{a} {c['be']} not {negative}. {da} {act}"
    elif family == "without_object":
        one["actions"] = [rest]
        missing = plural(c["item"])
        facts["negatives"] = [dict(kind="additional_object", value=missing)]
        absent = (
            f"there are no {missing}",
            f"no {missing} are present",
            f"there are not any {missing}",
            f"no {missing}",
        )[mode % 4]
        if variant == 0:
            text = f"{a} {rest} without any {missing}"
        else:
            text = f"{absent}. {a} {rest}" if index // 8 % 2 else f"{a} {rest}. {absent}"
    elif family == "static_no_action":
        material = {
            "lantern": "metal",
            "ball": "rubber",
            "box": "wooden",
            "bell": "brass",
            "ribbon": "silk",
            "cup": "clay",
            "wheel": "wooden",
            "umbrella": "fabric",
            "tray": "wooden",
            "bag": "leather",
            "jar": "glass",
            "book": "paper",
            "coin": "metal",
            "vase": "ceramic",
            "bottle": "glass",
            "cart": "wooden",
            "balloon": "rubber",
            "cushion": "cotton",
            "cymbal": "metal",
            "banjo": "wooden",
            "harp": "wooden",
            "pot": "clay",
            "stool": "wooden",
            "plate": "ceramic",
        }[c["item"]]
        attribute = ("small", "large", "shiny", "rough", "smooth", "tiny", "plain", "bright")[
            index // 8 % 8
        ]
        label = f"{material} {c['item'] if count == 1 else plural(c['item'])}"
        one.update(label=label, attributes=[attribute], actions=[])
        facts["setting"] = {"label": c["place"]}
        noun = f"{NUMBER[count - 1]} {attribute} {c['color']} {label}"
        prep = "inside" if mode % 2 else "in"
        text = f"{noun} {prep} a {c['place']}" if variant == 0 else f"{prep} a {c['place']}, {noun}"
        if index // 8 % 2:
            existential = "there is" if count == 1 else "there are"
            text = (
                f"{existential} {noun} {prep} a {c['place']}"
                if variant == 0
                else f"{prep} a {c['place']}, {existential} {noun}"
            )
    elif family == "explicit_target":
        kind = ("carries", "holds", "looks_at", "beside", "behind", "on", "under", "next_to")[mode]
        if kind == "carries":
            act = f"{c['be']} carrying {obj}"
        elif kind == "holds":
            act = f"{c['be']} holding {obj}"
        elif kind == "looks_at":
            act = f"{verb('look', count)} at {obj}"
        else:
            act = f"{verb('stand', count)} {kind.replace('_', ' ')} {obj}"
        one["actions"] = [act]
        facts["objects"] = [entity("o1", c["item"], c["shade"])]
        facts["relationships"] = [relation("s1", kind, "o1")]
        facts["setting"] = {"label": c["place"]}
        text = f"{a} {act} in a {c['place']}" if variant == 0 else f"in a {c['place']}, {a} {act}"
    elif family == "intransitive_clause":
        first, last = (
            ("watch", "dig"),
            ("wait", "walk"),
            ("stand", "rest"),
            ("look", "sit"),
            ("rest", "run"),
            ("sit", "jump"),
            ("walk", "sleep"),
            ("run", "crawl"),
        )[mode]
        one["actions"] = [verb(first, count)]
        two["actions"] = [verb(last, nc)]
        facts["subjects"].append(two)
        text = (
            f"{a} {one['actions'][0]} while {b} {two['actions'][0]}"
            if variant == 0
            else f"{b} {two['actions'][0]}. {a} {one['actions'][0]}"
        )
        if mode in (0, 3) and variant == 1:
            # A stated target differs from an intransitive neighboring clause.
            # This pair changes gold as well as wording; it is not a paraphrase.
            direct = f"{verb(first, count)}{' at' if first == 'look' else ''} {b}"
            one["actions"] = [direct]
            facts["relationships"] = [relation("s1", "looks_at", "s2")]
            text = f"{a} {direct}. {db} {two['actions'][0]}"
        elif mode not in (0, 3) and variant == 0:
            connector = ("while", "whereas", "but", "and")[index // 8 % 4]
            text = f"{a} {one['actions'][0]} {connector} {b} {two['actions'][0]}"
    elif family == "counted_shared_actions":
        action = (
            "are sleeping",
            "are resting",
            "are walking",
            "are standing",
            "are sitting",
            "are running",
            "are jumping",
            "are waiting",
        )[mode]
        one["actions"] = [action]
        two["actions"] = [action]
        facts["subjects"].append(two)
        text = f"{a} and {b} {action}" if variant == 0 else f"{b} and {a} {action}"
    elif family == "object_attachment":
        relation_kind = ("inside", "on", "under", "behind")[mode % 4]
        act = f"{c['be']} {'holding' if mode < 4 else 'carrying'} {obj}"
        one["actions"] = [act]
        container = "crate"
        facts["objects"] = [entity("o1", c["item"], c["shade"]), entity("o2", container)]
        facts["relationships"] = [
            relation("s1", "holds" if mode < 4 else "carries", "o1"),
            relation("o1", relation_kind, "o2"),
        ]
        tail = f"the {c['shade']} {c['item']} is {relation_kind} one {container}"
        text = f"{a} {act}. {tail}" if variant == 0 else f"{tail}. {a} {act}"
    elif family == "composition":
        layer = "foreground" if mode < 4 else "background"
        base = ("rest", "sit", "stand", "wait")[mode % 4]
        act = f"{verb(base, count)} in the {layer}"
        one["actions"] = [act]
        facts["salience"] = [dict(source="s1", layer=layer)]
        facts["setting"] = {"label": c["place"]}
        text = f"{a} {act} in a {c['place']}" if variant == 0 else f"in a {c['place']}, {a} {act}"
    elif family == "positive_negative_scope":
        act = verb(("walk", "rest", "stand", "sit")[mode % 4], count)
        neg = ("run", "jump", "sleep", "dance")[mode % 4]
        one["actions"] = [act]
        two["actions"] = [verb(neg, nc)]
        facts["subjects"].append(two)
        facts["negatives"] = [dict(kind="action", target="s1", value=neg)]
        auxiliary = "does" if count == 1 else "do"
        head = f"{a} {act}. {da} {auxiliary} not {neg}"
        tail = f"{b} {two['actions'][0]}"
        text = f"{head}. {tail}" if variant == 0 else f"{tail}. {head}"
    else:
        raise ValueError(family)
    return sentence(text), facts, f"{family}-{mode}-{variant}"


def contrast(family, combo, index, variant):
    c = context(combo, index)
    a, b, da, db = (c[k] for k in ("a", "b", "da", "db"))
    one, facts = c["one"], c["facts"]
    rest = verb(
        ("rest", "sleep", "stand", "sit", "walk", "run", "wait", "jump")[index % 8], c["count"]
    )
    one["actions"] = [rest]
    mode = index % 8
    name = (
        "Beatrice",
        "Vincent",
        "Harriet",
        "Bernard",
        "Matilda",
        "Dorothy",
        "Ernest",
        "Clementine",
    )[mode]
    if family in {"named_actor", "named_place"}:
        facts["setting"] = {"label": c["place"]}
        good = f"{a} {rest} in a {c['place']}" if variant == 0 else f"in a {c['place']}, {a} {rest}"
        if family == "named_actor":
            marker = ("named", "called", "known as", "named")[mode % 4]
            marked = f"{a} {marker} {name}"
            bad = (
                f"{marked} {rest} in a {c['place']}"
                if variant == 0
                else f"in a {c['place']}, {marked} {rest}"
            )
        else:
            location = (
                "Lisbon",
                "Prague",
                "Vienna",
                "Madrid",
                "Venice",
                "Oslo",
                "Naples",
                "Dublin",
            )[mode]
            bad = f"{a} {rest} in {location}" if variant == 0 else f"in {location}, {a} {rest}"
    elif family == "unbound_target":
        act = f"{c['be']} {'carrying' if mode % 2 else 'holding'} {c['obj']}"
        one["actions"] = [act]
        facts["objects"] = [entity("o1", c["item"], c["shade"])]
        facts["relationships"] = [relation("s1", "carries" if mode % 2 else "holds", "o1")]
        facts["setting"] = {"label": c["place"]}
        good = f"{a} {act} in a {c['place']}" if variant == 0 else f"in a {c['place']}, {a} {act}"
        target = ("it", "that", "the other one", "that one")[mode % 4]
        unclear = act.replace(c["obj"], target)
        bad = (
            f"{a} {unclear} in a {c['place']}"
            if variant == 0
            else f"in a {c['place']}, {a} {unclear}"
        )
    elif family == "unfinished_correction":
        good_scene = f"{a} {rest}"
        prior = f"{b} {verb('sleep', c['second_count'])}"
        good = (
            f"{prior}. Replace the whole scene with: {good_scene}"
            if variant == 0
            else f"{prior}. Discard that entire scene. The replacement scene is: {good_scene}"
        )
        suffix = (
            "No!",
            "Actually, no.",
            "No, instead",
            "Replace the whole scene with:",
            "No, wait",
            "Change that to",
            "Instead, a",
            "No, replace that with",
        )[mode]
        bad = f"{good_scene}. {suffix}" if variant == 0 else f"{good_scene}! {suffix}"
    elif family == "clipped_predicate":
        act = f"{c['be']} {'holding' if mode % 2 else 'carrying'} {c['obj']}"
        one["actions"] = [act]
        facts["objects"] = [entity("o1", c["item"], c["shade"])]
        facts["relationships"] = [relation("s1", "holds" if mode % 2 else "carries", "o1")]
        facts["setting"] = {"label": c["place"]}
        good = f"{a} {act} in a {c['place']}" if variant == 0 else f"in a {c['place']}, {a} {act}"
        fragment = ("carry-", "hold-", "walk-", "rest-", "jump-", "stand-", "sit-", "run-")[mode]
        bad = f"in a {c['place']}, {a} {c['be']} {fragment}" if variant == 0 else f"{a} {c['be']}"
    elif family == "indistinguishable_roles":
        animal, shade, color = c["animal"], c["shade"], c["color"]
        label = plural(animal)
        first = ("walks", "swims", "jumps", "sits", "runs", "stands", "sleeps", "rests")[mode]
        last = ("rests", "sleeps", "stands", "walks", "sits", "jumps", "runs", "swims")[mode]
        shared = "wait"
        facts["subjects"] = [
            entity("s1", animal, color, 1, actions=[shared, first]),
            entity("s2", animal, shade, 1, actions=[shared, last]),
        ]
        one_np, two_np = f"one {color} {animal}", f"one {shade} {animal}"
        d1, d2 = f"the {color} {animal}", f"the {shade} {animal}"
        if variant == 0:
            good = f"{one_np} and {two_np} {shared}. {d1} {first}. {d2} {last}"
            bad = f"two {color} {label} {shared}. One {first} and the other {last}"
        else:
            good = f"{two_np} and {one_np} {shared}. {d2} {last}. {d1} {first}"
            bad = f"two {color} {label} {shared}. One of them {first}. The other {last}"
    else:
        raise ValueError(family)
    return (
        sentence(good),
        facts,
        bad if family in {"clipped_predicate", "unfinished_correction"} else sentence(bad),
        f"{family}-{mode}-{variant}",
    )


def candidate_combos():
    rng = random.Random(202609095)
    seen = set()
    while True:
        a, b = rng.sample(ANIMALS, 2)
        c, d = rng.sample(COLORS, 2)
        combo = (a, b, c, d, rng.choice(ITEMS), rng.choice(PLACES))
        if combo not in seen:
            seen.add(combo)
            yield combo


def build():
    if importlib.metadata.version("xgrammar") != "0.2.6":
        raise ValueError("requires pinned XGrammar")
    grammar_data = (SUPPORT / "grammar.ebnf").read_bytes()
    if sha(grammar_data) != GRAMMAR_SHA:
        raise ValueError("grammar changed")
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo([])).compile_grammar(grammar_data.decode())
    combos = candidate_combos()
    rows = []
    used_sources = set()
    used_targets = set()
    target_counts = Counter()
    for family in (*CONTRASTS, *SEMANTIC):
        accepted = 0
        while accepted < 100:
            combo = next(combos)
            candidates = []
            for variant in range(2):
                if family in CONTRASTS:
                    source, payload, bad, form = contrast(family, combo, accepted, variant)
                    candidates.extend(
                        [
                            (family + "_valid", variant, source, payload, form),
                            (family, variant, bad, None, form),
                        ]
                    )
                else:
                    source, payload, form = positive(family, combo, accepted, variant)
                    candidates.append((family, variant, source, payload, form))
            source_keys = [" ".join(source.casefold().split()) for _, _, source, _, _ in candidates]
            if len(set(source_keys)) != len(source_keys) or used_sources.intersection(source_keys):
                continue
            targets = {
                SceneFactsV2.model_validate(payload).to_wire()
                for _, _, _, payload, _ in candidates
                if payload
            }
            if used_targets.intersection(targets):
                continue
            group = f"train-{family}-{accepted:03}"
            for kind, variant, source, payload, form in candidates:
                facts = SceneFactsV2.model_validate(copy.deepcopy(payload)) if payload else None
                if facts:
                    try:
                        facts.to_renderer_prompt(source_text=source, visual_style="rich watercolor")
                    except ValueError as error:
                        raise ValueError(
                            f"gold boundary {kind}/{accepted}/{variant}: {source}"
                        ) from error
                    target = facts.to_wire()
                    if SceneFactsV2.from_wire(target) != facts:
                        raise ValueError("wire roundtrip")
                    target_counts[target] += 1
                else:
                    target = "REFUSE"
                matcher = xgr.GrammarMatcher(compiled)
                if not matcher.accept_string(target) or not matcher.is_completed():
                    raise ValueError("outside fixed grammar")
                words = set(re.findall(r"[a-z]+", source.casefold()))
                if words.intersection({f for w in RESERVED for f in (w, plural(w))}):
                    raise ValueError("reserved independent lexicon")
                rows.append(
                    dict(
                        id=f"train-{kind}-{accepted:03}-{variant}",
                        group_id=group,
                        family=kind,
                        form=form,
                        split="train",
                        tuple=combo,
                        source=source,
                        facts=facts.model_dump(mode="json") if facts else None,
                        target=target,
                    )
                )
            used_sources.update(source_keys)
            used_targets.update(targets)
            accepted += 1
    if (
        len(rows) != 4800
        or sum(r["target"] == "REFUSE" for r in rows) != 1200
        or max(target_counts.values()) > 2
    ):
        raise ValueError("corpus counts")
    return rows


def materialize(directory=HERE):
    prompt_data = (HERE / "prompt.txt").read_bytes()
    if sha(prompt_data) != PROMPT_SHA:
        raise ValueError("prompt changed")
    rows = build()
    prompt = prompt_data.decode().strip()
    files = {
        "prompt.txt": prompt_data,
        "train.jsonl": b"".join(encoded(r) for r in rows),
        "train-messages.jsonl": b"".join(
            encoded(
                {
                    "id": r["id"],
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": r["source"]},
                        {"role": "assistant", "content": r["target"]},
                    ],
                }
            )
            for r in rows
        ),
    }
    overlap = {
        "schema_version": 1,
        "scope": "Training only; no independent development or test read",
        "reserved_independent_nominal_lexicon": RESERVED,
        "vocabulary": {"animals": ANIMALS, "objects": ITEMS, "places": PLACES, "colors": COLORS},
        "groups": [
            {
                "id": r["id"],
                "group_id": r["group_id"],
                "form": r["form"],
                "tuple_sha256": sha(encoded(r["tuple"])),
                "source_sha256": sha(" ".join(r["source"].casefold().split()).encode()),
                "target_sha256": sha(r["target"].encode()) if r["facts"] else None,
            }
            for r in rows
        ],
    }
    files["train-overlap.json"] = encoded(overlap)
    manifest = {
        "schema_version": 1,
        "training_rows": 4800,
        "positive_rows": 3600,
        "refusal_rows": 1200,
        "linked_groups": len({r["group_id"] for r in rows}),
        "positive_graphs": len({r["target"] for r in rows if r["facts"]}),
        "max_positive_paraphrases_per_graph": max(
            Counter(r["target"] for r in rows if r["facts"]).values()
        ),
        "development_rows": 0,
        "development_data_read": False,
        "evaluation_data_read": False,
        "prompt_sha256": PROMPT_SHA,
        "grammar_sha256": GRAMMAR_SHA,
        "xgrammar_version": "0.2.6",
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "sources": {n: sha((ROOT / n).read_bytes()) for n in SOURCES},
        "files": {
            n: {"sha256": sha(data), "rows": len(data.splitlines())} for n, data in files.items()
        },
        "contrast_families": CONTRASTS,
        "semantic_families": SEMANTIC,
        "scope": (
            "Authored linked constructions and lexical combinations; "
            "not 4800 independent natural scenes"
        ),
    }
    directory.mkdir(parents=True, exist_ok=True)
    for n, data in files.items():
        (directory / n).write_bytes(data)
    (directory / "manifest.json").write_bytes(encoded(manifest))
    return manifest


if __name__ == "__main__":
    print(json.dumps(materialize(), indent=2))
