"""General dependency-to-draft extraction. No renderer or network API exists here.

Source spans remain authoritative. Parser predictions are review suggestions;
coverage checks detect omissions, not all semantic errors. No verb whitelist.
"""

from bookforge import privacy_policy as privacy


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
        actor_ids = actor_group(subjects[0])
        object_ids = actor_group(objects[0]) if objects else [None]
        for actor in actor_ids:
            for target in object_ids:
                events.append(
                    {
                        "token": index,
                        "actor": actor,
                        "action": token["text"],
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
