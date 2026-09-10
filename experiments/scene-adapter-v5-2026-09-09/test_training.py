"""V5 training checks; independent development/test data are never read."""

import collections
import copy
import importlib.util
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storylight.scene_facts import SceneFactsGroundingError, SceneFactsV2

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v5_training", HERE / "generate_training.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generator.build()

    def test_exact_balance_linked_groups_and_maximum_paraphrases(self):
        self.assertEqual(len(self.rows), 4800)
        self.assertEqual(sum(r["target"] == "REFUSE" for r in self.rows), 1200)
        self.assertEqual(len({r["id"] for r in self.rows}), 4800)
        self.assertEqual(len({" ".join(r["source"].casefold().split()) for r in self.rows}), 4800)
        groups = collections.defaultdict(list)
        targets = collections.defaultdict(set)
        target_counts = collections.Counter()
        for row in self.rows:
            groups[row["group_id"]].append(row)
            if row["facts"]:
                targets[row["target"]].add(row["group_id"])
                target_counts[row["target"]] += 1
        self.assertEqual(len(groups), 1800)
        self.assertEqual(len({tuple(rows[0]["tuple"]) for rows in groups.values()}), 1800)
        self.assertTrue(all(len(group_ids) == 1 for group_ids in targets.values()))
        self.assertLessEqual(max(target_counts.values()), 2)
        self.assertEqual(len(targets), 1826)
        self.assertEqual(
            collections.Counter(
                (len(rows), sum(r["target"] == "REFUSE" for r in rows)) for rows in groups.values()
            ),
            {(4, 2): 600, (2, 0): 1200},
        )

    def test_reserved_lexicon_and_targeted_fact_fields(self):
        reserved = {form for word in generator.RESERVED for form in (word, generator.plural(word))}
        for row in self.rows:
            words = set(re.findall(r"[a-z]+", row["source"].casefold()))
            self.assertFalse(words & reserved, row["id"])
            facts = row["facts"]
            if not facts:
                continue
            family = row["family"]
            if family.startswith("temporal_"):
                self.assertEqual(len(facts["events"]), 2)
                self.assertEqual(facts["temporal_order"], [{"before": "e1", "after": "e2"}])
                self.assertNotEqual(facts["events"][0]["action"], facts["events"][1]["action"])
                self.assertFalse(facts["relationships"])
            elif family == "repeated_actor_actions":
                self.assertEqual(sorted(len(s["actions"]) for s in facts["subjects"]), [1, 2])
            elif family in {"without_action", "without_object"}:
                self.assertEqual(len(facts["negatives"]), 1)
                self.assertTrue(
                    all(
                        "without" not in a.split() and "not" not in a.split()
                        for s in facts["subjects"]
                        for a in s["actions"]
                    )
                )
                if family == "without_action":
                    self.assertEqual(facts["negatives"][0]["target"], "s1")
                else:
                    self.assertIsNone(facts["negatives"][0]["target"])
            elif family == "static_no_action":
                self.assertTrue(all(not s["actions"] for s in facts["subjects"]))
            elif family == "indistinguishable_roles_valid":
                self.assertEqual(len(facts["subjects"]), 2)
                self.assertEqual(len({s["color"] for s in facts["subjects"]}), 2)
                self.assertTrue(
                    all(s["count"] == 1 and len(s["actions"]) == 2 for s in facts["subjects"])
                )

    def test_explicit_vs_neighboring_clause_target_and_grounding_guard(self):
        groups = collections.defaultdict(list)
        for row in self.rows:
            if row["family"] == "intransitive_clause":
                groups[row["group_id"]].append(row)
        contrasts = 0
        for rows in groups.values():
            implicit, explicit = rows
            if not explicit["facts"]["relationships"]:
                continue
            contrasts += 1
            self.assertFalse(implicit["facts"]["relationships"])
            self.assertIn(" while ", implicit["source"])
            self.assertEqual(
                explicit["facts"]["relationships"],
                [
                    {
                        "source": "s1",
                        "relation": "looks_at",
                        "target": "s2",
                        "secondary_target": None,
                    }
                ],
            )
            changed = copy.deepcopy(implicit["facts"])
            changed["relationships"] = explicit["facts"]["relationships"]
            with self.assertRaises(SceneFactsGroundingError):
                SceneFactsV2.model_validate(changed).to_renderer_prompt(
                    source_text=implicit["source"], visual_style="rich watercolor"
                )
        self.assertEqual(contrasts, 26)

    def test_refusal_controls_have_linked_valid_neighbors(self):
        grouped = collections.defaultdict(list)
        for row in self.rows:
            grouped[row["group_id"]].append(row)
        for rows in grouped.values():
            refused = [r for r in rows if r["target"] == "REFUSE"]
            if not refused:
                continue
            self.assertEqual(len(refused), 2)
            self.assertEqual(sum(r["facts"] is not None for r in rows), 2)
            for row in refused:
                text = row["source"].casefold()
                family = row["family"]
                if family == "clipped_predicate":
                    self.assertTrue(text.endswith(("-", " is", " are")))
                elif family == "unfinished_correction":
                    self.assertTrue(
                        any(x in text for x in ("no", "instead", "with:", "change that"))
                    )
                elif family == "indistinguishable_roles":
                    self.assertTrue(text.startswith("two "))
                    self.assertIn("the other", text)
                elif family == "unbound_target":
                    self.assertTrue(
                        bool(re.search(r"\b(?:it|that)\b", text)) or "other one" in text
                    )
                elif family == "named_actor":
                    self.assertTrue(any(x in text for x in (" named ", " called ", " known as ")))
                else:
                    self.assertEqual(family, "named_place")
                    self.assertTrue(
                        any(
                            x in text
                            for x in (
                                "lisbon",
                                "prague",
                                "vienna",
                                "madrid",
                                "venice",
                                "oslo",
                                "naples",
                                "dublin",
                            )
                        )
                    )

    def test_reproduction_and_no_independent_data_reads(self):
        original = Path.read_bytes

        def guarded(path):
            self.assertFalse(path.name.startswith(("development", "screen", "author")), path)
            return original(path)

        with tempfile.TemporaryDirectory() as folder, patch.object(Path, "read_bytes", guarded):
            manifest = generator.materialize(Path(folder))
            self.assertEqual(
                json.loads(generator.encoded(manifest)),
                json.loads((HERE / "manifest.json").read_text()),
            )
            self.assertFalse(manifest["development_data_read"])
            self.assertFalse(manifest["evaluation_data_read"])
            for name, entry in manifest["files"].items():
                actual = (Path(folder) / name).read_bytes()
                self.assertEqual(actual, (HERE / name).read_bytes())
                self.assertEqual(generator.sha(actual), entry["sha256"])
            self.assertEqual(
                generator.sha((Path(folder) / "prompt.txt").read_bytes()), generator.PROMPT_SHA
            )

    @unittest.skipUnless(os.environ.get("STORYLIGHT_V5_TOKENIZER_DIR"), "local tokenizer not set")
    def test_actual_tokenizer_masks_limits_and_all_grammar_stops(self):
        import xgrammar as xgr
        from transformers import AutoTokenizer

        tokenizer_dir = Path(os.environ["STORYLIGHT_V5_TOKENIZER_DIR"])
        model_data = (generator.V2 / "results/gpu/model-manifest.json").read_bytes()
        self.assertEqual(
            generator.sha(model_data),
            "703bbb89d61aaed083846d7cb3d4ee1a68220e25de93a035b1f4b49d24062f2d",
        )
        model = json.loads(model_data)
        names = {
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
        }
        self.assertEqual({p.name for p in tokenizer_dir.iterdir()}, names)
        for name in names:
            self.assertEqual(
                generator.sha((tokenizer_dir / name).read_bytes()), model["files"][name]
            )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, local_files_only=True, trust_remote_code=False
        )
        stops = [1, 106, 50]
        compiled = xgr.GrammarCompiler(
            xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=262144, stop_token_ids=stops)
        ).compile_grammar((generator.V3 / "grammar.ebnf").read_text())
        totals, completions = [], []
        for row in map(json.loads, (HERE / "train-messages.jsonl").read_text().splitlines()):
            prefix = tokenizer.apply_chat_template(
                row["messages"][:-1],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            full = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            self.assertEqual(full[: len(prefix)], prefix)
            self.assertEqual(
                tokenizer.decode(full[len(prefix) :], skip_special_tokens=True).strip(),
                row["messages"][-1]["content"],
            )
            target = tokenizer.encode(row["messages"][-1]["content"], add_special_tokens=False)
            self.assertLessEqual(len(target) + 1, 256)
            self.assertLessEqual(len(prefix) + 256, 2048)
            self.assertLessEqual(len(full), 2048)
            for stop in stops:
                matcher = xgr.GrammarMatcher(compiled)
                self.assertTrue(all(matcher.accept_token(t) for t in [*target, stop]))
                self.assertTrue(matcher.is_terminated())
            totals.append(len(full))
            completions.append(len(full) - len(prefix))
        self.assertEqual(len(totals), 4800)
        print(
            json.dumps(
                {
                    "training_rows": 4800,
                    "max_total_tokens": max(totals),
                    "max_completion_tokens": max(completions),
                    "stops": stops,
                    "truncation": False,
                    "model_inference": False,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
