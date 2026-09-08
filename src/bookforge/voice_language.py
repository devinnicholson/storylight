"""Local learned-syntax candidate, validated by the existing scene-fact boundary.

Importing the module is lightweight. The explicit CLI starts a local Unix
socket service with one CPU pipeline. Neither path makes cloud calls or admits
rendering; every draft must be reviewed before use.
"""

from __future__ import annotations

from bookforge.privacy_policy import COLOR_WORDS, COUNT_WORDS
from bookforge.scene_facts import SceneFactsV2
from bookforge.voice_dependencies import (
    audit_nominal_spans,
    extract,
    normalize_breed_subjects,
    static_nominal_head,
)

REVISION = "dependency-scene-draft-v3"
RELATIONS = {
    "in": "inside",
    "inside": "inside",
    "on": "on",
    "above": "above",
    "over": "above",
    "under": "under",
    "below": "below",
    "behind": "behind",
    "beside": "beside",
    "next to": "next_to",
}


def row_from_doc(doc):
    return {
        "id": "request",
        "text": doc.text,
        "tokens": [
            {
                "i": t.i,
                "text": t.text,
                "offset": t.idx,
                "lemma": t.lemma_,
                "pos": t.pos_,
                "dep": t.dep_,
                "head": t.head.i,
            }
            for t in doc
        ],
        "entities": [
            {"text": e.text, "label": e.label_, "start": e.start, "end": e.end} for e in doc.ents
        ],
    }


def extract_graph(text: str, visual_style: str, *, nlp):
    if (
        not isinstance(text, str)
        or not 1 <= len(text) <= 500
        or not text.strip()
        or not isinstance(visual_style, str)
        or not 1 <= len(visual_style) <= 120
        or not visual_style.strip()
    ):
        return _response({"status": "needs_review", "reason": "input_bounds"})
    row = row_from_doc(nlp(text))
    nominal = static_nominal_head(row)
    # A phrase-context parser can mistake an unfinished verb for a noun root
    # ("a cat chasing"). Check that head independently before a static draft.
    head_pos = None
    if nominal is not None:
        isolated = list(nlp(row["tokens"][nominal]["text"]))
        if len(isolated) == 1:
            head_pos = isolated[0].pos_
    return _response(graph_from_row(row, visual_style, nominal_head_pos=head_pos))


def graph_from_row(row, visual_style, *, nominal_head_pos=None):
    row = normalize_breed_subjects(row)
    nominal = static_nominal_head(row)
    if nominal_head_pos not in {"NOUN", "PROPN"}:
        nominal = None
    if nominal is not None:
        row = {**row, "tokens": [dict(t) for t in row["tokens"]]}
        for token in row["tokens"]:
            if token["pos"] == "PROPN":
                token["pos"] = "NOUN"
    draft = extract(row, static_subject=nominal)
    result = {
        "revision": REVISION,
        "status": "needs_review",
        "facts": None,
        "reason": None,
        "render_admitted": False,
        "requires_fact_review": True,
        "local_omissions": draft["withheld_local_only"],
        "syntax_issues": draft["issues"],
    }
    if draft["issues"]:
        result["reason"] = "syntax_coverage_or_reference"
        return result
    if any(event["negative"] for event in draft["events"]):
        # Negation is detected across dependency ancestors. Do not turn a
        # partially interpreted negative event into an affirmative scene.
        result["reason"] = "negative_event_review"
        return result
    ts = row["tokens"]
    ents = {e["ref"]: e for e in draft["entities"]}
    withheld = {e["ref"] for e in draft["withheld_local_only"]}
    actors = {e["actor"] for e in draft["events"]}
    if nominal is not None:
        actors.add(nominal)
    targets = {e["object"] for e in draft["events"] if e["object"] is not None}
    if withheld & (actors | targets):
        result["reason"] = "named_actor_or_target"
        return result
    absents = set(draft["absent"])
    if actors & absents:
        result["reason"] = "contradictory_absence"
        return result
    location_targets = {
        r["target"]
        for r in draft["relations"]
        if r["relation"].casefold() in {"in", "inside"}
        and r["owner_token"] in {e["token"] for e in draft["events"]}
        and r["target"] not in withheld
    }
    if len(location_targets) > 1:
        result["reason"] = "multiple_event_settings"
        return result
    setting_target = next(iter(location_targets), None)
    if setting_target is not None and (
        len({e["token"] for e in draft["events"]}) != 1 or setting_target in actors | targets
    ):
        result["reason"] = "setting_scope_review"
        return result
    selected = set(ents) - withheld - absents - location_targets
    refs = {i: f"e{n}" for n, i in enumerate(sorted(selected), 1)}

    def entity(i):
        e = ents[i]
        token = ts[i]
        mods = [ts[t] for t in e["tokens"] if t != i]
        compounds = [t for t in mods if t["dep"] == "compound"]
        if nominal == i:
            compounds += [t for t in mods if t["dep"] == "amod" and any(
                child["head"] == t["i"] and child["dep"] == "compound" for child in mods
            )]
            compounds.sort(key=lambda t: t["i"])
        label = " ".join(t["text"].casefold() for t in [*compounds, token])
        adjectives = [t["text"].casefold() for t in mods
                      if t["dep"] == "amod" and t not in compounds]
        color = next((a for a in adjectives if a in COLOR_WORDS), None)
        attributes = [a for a in adjectives if a != color]
        nums = [t["text"].casefold() for t in mods if t["dep"] == "nummod"]
        if len(nums) > 1:
            raise ValueError("ambiguous count")
        count = int(COUNT_WORDS.get(nums[0], nums[0])) if nums else None
        if count is None and any(
            t["head"] == i and t["dep"] == "det" and t["text"].casefold() in {"a", "an"} for t in ts
        ):
            count = 1
        return {
            "ref": refs[i],
            "label": label,
            "color": color,
            "attributes": attributes,
            "count": count,
        }

    try:
        subjects, objects, relationships = [], [], []
        for i in sorted(selected):
            value = entity(i)
            if i in actors:
                actions = []
                for event in draft["events"]:
                    if event["actor"] != i:
                        continue
                    action = event["action"].casefold()
                    if event["object"] is not None:
                        action += " " + ents[event["object"]]["phrase"].casefold()
                    for relation in draft["relations"]:
                        if (
                            relation["owner_token"] == event["token"]
                            and relation["target"] not in withheld
                        ):
                            action += (
                                " "
                                + relation["relation"].casefold()
                                + " "
                                + ents[relation["target"]]["phrase"].casefold()
                            )
                    actions.append(action)
                value["actions"] = list(dict.fromkeys(actions))
                subjects.append(value)
            else:
                objects.append(value)
        for relation in draft["relations"]:
            if relation["target"] in withheld or relation["owner_token"] in withheld:
                continue
            if relation["target"] == setting_target:
                continue
            if any(
                e["token"] == relation["owner_token"] and e["object"] is not None
                for e in draft["events"]
            ):
                # Preserve the entire source predicate, but do not guess whether
                # a trailing location modifies its actor or direct object.
                continue
            kind = RELATIONS.get(relation["relation"].casefold())
            if kind is None:
                # Exact preposition already remains in the actor action. It is
                # not silently normalized to a different spatial relation.
                if relation["owner_token"] not in {e["token"] for e in draft["events"]}:
                    raise ValueError("unrepresented entity relation")
                continue
            owners = [e["actor"] for e in draft["events"] if e["token"] == relation["owner_token"]]
            if not owners and relation["owner_token"] in selected:
                owners = [relation["owner_token"]]
            for owner in owners:
                relationships.append(
                    {"source": refs[owner], "relation": kind, "target": refs[relation["target"]]}
                )
        facts = SceneFactsV2.model_validate(
            {
                "setting": {
                    "label": ents[setting_target]["phrase"].casefold()
                    if setting_target is not None
                    else "unspecified"
                },
                "subjects": subjects,
                "objects": objects,
                "relationships": relationships,
                "negatives": [
                    {"kind": "additional_object", "value": ents[i]["phrase"].casefold()}
                    for i in sorted(absents)
                ],
            }
        )
        # Original source, including withheld names, stays the validation basis.
        facts.validate_source_grounding(source_text=row["text"])
        prompt = facts.to_renderer_prompt(source_text=row["text"], visual_style=visual_style)
        if any(
            e["local_text"].casefold() in prompt.casefold() for e in draft["withheld_local_only"]
        ):
            raise ValueError("private name in prompt")
        result.update(
            status="omission_review" if withheld else "draft_ready",
            reason=None,
            facts=facts.model_dump(mode="json"),
            renderer_prompt_preview=prompt,
        )
    except (ValueError, KeyError):
        result["reason"] = "existing_fact_boundary"
    return result


def _response(value):
    return {
        "revision": REVISION,
        "status": value["status"],
        "facts": value.get("facts"),
        "reason": value.get("reason"),
        "render_admitted": False,
        "requires_fact_review": True,
        "local_omissions": value.get("local_omissions", []),
        "syntax_issues": value.get("syntax_issues", []),
        "renderer_prompt_preview": value.get("renderer_prompt_preview"),
    }


def serve(socket_path):
    """Serve bounded local requests over a fresh owner-only Unix socket."""
    import importlib.metadata
    import json
    import os
    import socketserver
    from http.server import BaseHTTPRequestHandler
    from pathlib import Path

    if importlib.metadata.version("spacy") != "3.8.11":
        raise RuntimeError("local language runtime version mismatch")
    import spacy

    spacy.require_cpu()
    nlp = spacy.load("en_core_web_sm")
    if nlp.meta.get("version") != "3.8.0":
        raise RuntimeError("local language model version mismatch")
    path = Path(socket_path)
    if not path.is_absolute() or not path.parent.is_dir() or os.path.lexists(path):
        raise ValueError("local language socket must be a fresh absolute path")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, *_args):
            pass

        def respond(self, status, body):
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            if self.path != "/v1/scene-facts":
                self.respond(404, {"error": "invalid_request"})
                return
            try:
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1 or not lengths[0].isdigit():
                    raise ValueError("length")
                size = int(lengths[0])
                if not 0 < size <= 8192 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("body bounds")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("content type")
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("incomplete body")

                def pairs(items):
                    value = {}
                    for key, item in items:
                        if key in value:
                            raise ValueError("duplicate key")
                        value[key] = item
                    return value

                value = json.loads(raw, object_pairs_hook=pairs)
                if not isinstance(value, dict) or set(value) not in (
                    {"text", "visual_style"},
                    {"text", "visual_style", "nominal_labels"},
                ):
                    raise ValueError("schema")
                if not isinstance(value["text"], str) or not 1 <= len(value["text"]) <= 500:
                    raise ValueError("text")
                if (
                    not isinstance(value["visual_style"], str)
                    or not 1 <= len(value["visual_style"]) <= 120
                ):
                    raise ValueError("style")
                labels = value.get("nominal_labels")
                if "nominal_labels" in value and (
                    not isinstance(labels, list)
                    or not 1 <= len(labels) <= 10
                    or any(
                        not isinstance(label, str) or not 1 <= len(label) <= 64 or not label.strip()
                        for label in labels
                    )
                ):
                    raise ValueError("nominal labels")
            except (ValueError, UnicodeError, OSError):
                self.respond(400, {"error": "invalid_request"})
                return
            try:
                if labels is not None:
                    result = audit_nominal_spans(row_from_doc(nlp(value["text"])), labels)
                else:
                    result = extract_graph(value["text"], value["visual_style"], nlp=nlp)
            except Exception:
                self.respond(503, {"error": "unavailable"})
                return
            self.respond(200, result)

        def send_error(self, code, message=None, explain=None):
            self.respond(code, {"error": "invalid_request"})

        def do_GET(self):
            self.respond(405, {"error": "invalid_request"})

    class Server(socketserver.UnixStreamServer):
        def handle_error(self, request, client_address):
            pass

    old_mask = os.umask(0o077)
    try:
        server = Server(str(path), Handler)
        path.chmod(0o600)
    finally:
        os.umask(old_mask)
    owned = path.stat()
    try:
        with server:
            server.serve_forever(poll_interval=0.25)
    finally:
        current = path.stat() if path.exists() else None
        if current and (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
            path.unlink()


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    args = parser.parse_args()
    try:
        serve(args.socket)
    except KeyboardInterrupt:
        pass
    except Exception:
        print("local language service unavailable", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
