"""General dependency-to-draft extraction. No renderer or network API exists here.

Source spans remain authoritative. Parser predictions are review suggestions;
coverage checks detect omissions, not all semantic errors. No verb whitelist.
"""

from bookforge import privacy_policy as privacy


def normalize_breed_subjects(row):
    """Repair only a recognized literal subject phrase, never adjacent actors.

    The small dependency model can split a compound breed into multiple nsubj
    heads and label its last word PROPN. Source offsets and the shared lexical
    span constrain this correction; named entities and ambiguous attachments
    remain authoritative refusals. The original prediction is not mutated.
    """
    tokens = row["tokens"]
    protected = {
        i
        for span in row["entities"]
        if span["label"] in {"PERSON", "GPE", "LOC", "FAC", "ORG"}
        for i in range(span["start"], span["end"])
    }
    repaired = [dict(t) for t in tokens]
    for phrase in privacy.recognized_breed_phrases(row["text"]):
        indexes = [t["i"] for t in tokens if phrase.start() <= t["offset"] < phrase.end()]
        if not indexes or protected.intersection(indexes):
            continue
        first, head = indexes[0], indexes[-1]
        if first and tokens[first - 1]["text"] not in {".", "!", "?", ";"}:
            continue
        if (
            tokens[first]["offset"] != phrase.start()
            or tokens[head]["offset"] + len(tokens[head]["text"]) != phrase.end()
            or indexes != list(range(first, head + 1))
        ):
            continue
        subjects = [t for t in tokens[first : head + 1] if t["dep"] == "nsubj"]
        if not subjects or len({t["head"] for t in subjects}) != 1:
            continue
        predicate = subjects[0]["head"]
        if predicate <= head or tokens[predicate]["pos"] != "VERB":
            continue
        if any(
            t["dep"] in {"nsubj", "nsubjpass"} and t["head"] == predicate and t["i"] not in indexes
            for t in tokens
        ):
            continue
        if any(
            t["head"] != predicate or t["dep"] not in {"aux", "neg"}
            for t in tokens[head + 1 : predicate]
        ):
            continue
        if any(
            t["dep"] not in {"det", "amod", "compound", "nmod", "nummod", "nsubj"}
            or (t["dep"] != "nsubj" and t["head"] not in indexes)
            for t in tokens[first : head + 1]
        ):
            continue
        if any(t["head"] in indexes and t["i"] not in indexes for t in tokens):
            continue
        for index in indexes:
            t = repaired[index]
            t["head"] = predicate if index == head else head
            if t["offset"] >= phrase.start("breed"):
                t["dep"] = "nsubj" if index == head else "compound"
                t["pos"] = "NOUN"
    return {**row, "tokens": repaired}


def extract(row):
    ts = row["tokens"]
    children = {t["i"]: [] for t in ts}
    for t in ts:
        if t["head"] != t["i"]:
            children[t["head"]].append(t["i"])
    issues, entities, events, relations, absent = [], {}, [], [], []
    covered = set()
    predicates = {
        t["i"]
        for t in ts
        if t["pos"] == "VERB"
        or any(ts[c]["dep"] in {"nsubj", "nsubjpass"} for c in children[t["i"]])
    }

    def descendants(index, deps):
        result = {index}
        for c in children[index]:
            if ts[c]["dep"] in deps:
                result |= descendants(c, deps)
        return result

    def noun(index):
        token = ts[index]
        if token["pos"] == "PRON":
            issues.append({"kind": "unresolved_pronoun", "token": index})
        selected = descendants(index, {"amod", "compound", "nummod", "poss"})
        # Counts and adjectives stay bound in one literal phrase, including breeds.
        phrase = " ".join(ts[i]["text"] for i in sorted(selected))
        entities[index] = {
            "ref": index,
            "phrase": phrase,
            "tokens": sorted(selected),
            "head_lemma": token["lemma"],
        }
        covered.update(selected)
        return index

    def actor_group(index):
        return [noun(i) for i in sorted(descendants(index, {"conj"}))]

    def bound_negative(index):
        for token in ts:
            if token["dep"] != "neg":
                continue
            cursor, seen = token["i"], set()
            while cursor not in seen:
                if cursor == index:
                    return True
                if cursor in predicates:
                    break
                seen.add(cursor)
                cursor = ts[cursor]["head"]
        return False

    for index in sorted(predicates):
        token = ts[index]
        subjects = [c for c in children[index] if ts[c]["dep"] == "nsubj"]
        patients = [c for c in children[index] if ts[c]["dep"] == "nsubjpass"]
        objects = [c for c in children[index] if ts[c]["dep"] in {"dobj", "dative"}]
        agents = [c for c in children[index] if ts[c]["dep"] == "agent"]
        if patients:
            subjects = [c for a in agents for c in children[a] if ts[c]["dep"] == "pobj"]
            objects = patients
        if not subjects and token["dep"] == "acl" and ts[token["head"]]["pos"] in {"NOUN", "PROPN"}:
            subjects = [token["head"]]
        if not subjects and token["dep"] == "conj" and token["head"] in predicates:
            subjects = [c for c in children[token["head"]] if ts[c]["dep"] == "nsubj"]
        if len(subjects) != 1 or len(objects) > 1:
            issues.append({"kind": "unresolved_event_arguments", "token": index})
            continue
        particles = sorted(c for c in children[index] if ts[c]["dep"] == "prt")
        if particles and particles != list(range(index + 1, index + 1 + len(particles))):
            issues.append({"kind": "unresolved_predicate_particle", "token": index})
            continue
        action = " ".join(ts[i]["text"] for i in [index, *particles])
        covered.update(particles)
        actor_ids = actor_group(subjects[0])
        object_ids = actor_group(objects[0]) if objects else [None]
        for actor in actor_ids:
            for target in object_ids:
                events.append(
                    {
                        "token": index,
                        "actor": actor,
                        "action": action,
                        "lemma": token["lemma"],
                        "object": target,
                        "negative": bound_negative(index),
                    }
                )
        covered.add(index)
    for token in ts:
        index = token["i"]
        if token["pos"] in {"NOUN", "PROPN"} and any(
            ts[c]["text"].casefold() == "no" and ts[c]["dep"] == "det" for c in children[index]
        ):
            absent.append(noun(index))
        if token["dep"] != "prep":
            continue
        targets = [c for c in children[index] if ts[c]["dep"] == "pobj"]
        if len(targets) != 1:
            issues.append({"kind": "unresolved_preposition", "token": index})
            continue
        owner = token["head"]
        if owner not in predicates and ts[owner]["pos"] in {"NOUN", "PROPN"}:
            noun(owner)
        relations.append(
            {
                "owner_token": owner,
                "relation": token["text"],
                "target": noun(targets[0]),
                "token": index,
            }
        )
        covered.add(index)
    for token in ts:
        if (
            token["pos"] in {"NOUN", "PROPN", "VERB", "ADJ", "NUM", "ADV", "SCONJ"}
            and token["i"] not in covered
        ):
            issues.append({"kind": "unrepresented_content", "token": token["i"]})
    for token in ts:
        if token["dep"] in {"advcl", "ccomp", "xcomp"} and token["i"] not in predicates:
            issues.append({"kind": "unresolved_clause", "token": token["i"]})
    if not events:
        issues.append({"kind": "no_event"})
    proper = privacy.proper_name_candidates(row["text"])
    withheld = []
    predicted_names = {
        index
        for span in row["entities"]
        if span["label"] in {"PERSON", "GPE", "LOC", "FAC"}
        for index in range(span["start"], span["end"])
    }
    predicted_names.update(t["i"] for t in ts if t["pos"] == "PROPN")
    for span in row["entities"]:
        if span["label"] == "ORG":
            issues.append({"kind": "uncertain_named_entity", "token": span["start"]})
    for entity in entities.values():
        words = privacy.privacy_tokens(entity["phrase"])
        if predicted_names.intersection(entity["tokens"]) or any(
            p and any(tuple(words[i : i + len(p)]) == p for i in range(len(words))) for p in proper
        ):
            withheld.append(
                {
                    "ref": entity["ref"],
                    "local_text": entity["phrase"],
                    "reason": "proper_name_policy",
                    "omission_requires_review": True,
                }
            )
    # NER is predictive, so retain disagreement for review instead of silently
    # deleting common nouns/breeds mislabelled as organizations.
    ner_review = [
        e for e in row["entities"] if e["label"] in {"PERSON", "ORG", "GPE", "LOC", "FAC"}
    ]
    return {
        "id": row["id"],
        "entities": list(entities.values()),
        "events": events,
        "relations": relations,
        "absent": absent,
        "issues": issues,
        "syntax_coverage_pass": not issues,
        "withheld_local_only": withheld,
        "named_entity_review": ner_review,
        "requires_user_fact_review": True,
        "render_admitted": False,
    }


def audit_nominal_spans(row, labels):
    """Check literal bounded noun labels against learned nominal dependencies.

    This audits an existing graph; it does not repair or reinterpret a label.
    Privacy-sensitive predicted spans cannot be approved by a nominal audit.
    """
    tokens = row["tokens"]
    protected = {
        i
        for span in row["entities"]
        if span["label"] in {"PERSON", "GPE", "LOC", "FAC"}
        for i in range(span["start"], span["end"])
    }
    issues = []
    for label_index, label in enumerate(labels):
        words = privacy.privacy_tokens(label)
        accepted = False
        for start in range(len(tokens)):
            end = start + len(words)
            span = tokens[start:end]
            if len(span) != len(words) or tuple(t["text"].casefold() for t in span) != words:
                continue
            indexes = {t["i"] for t in span}
            if indexes & protected or any(t["pos"] not in {"NOUN", "PROPN", "ADJ"} for t in span):
                continue
            head = span[-1]
            if head["pos"] not in {"NOUN", "PROPN"}:
                continue
            if head["head"] in indexes and head["head"] != head["i"]:
                continue
            if any(
                t["dep"] not in {"amod", "compound"} or t["head"] not in indexes for t in span[:-1]
            ):
                continue
            accepted = True
            break
        if not accepted:
            issues.append({"label_index": label_index, "reason": "not_nominal_span"})
    return {"revision": "dependency-nominal-audit-v1", "accepted": not issues, "issues": issues}
