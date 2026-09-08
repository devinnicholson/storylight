"""Actual retained-parser consequence tests; no network or model downloads."""

import json
import unittest
from pathlib import Path

from bookforge.bounded_description import plan_bounded_description
from bookforge.voice_dependencies import audit_nominal_spans
from bookforge.voice_language import extract_graph, graph_from_row

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/voice-language-2026-09-07"


class CandidateTests(unittest.TestCase):
    def test_input_bounds_do_not_run_parser(self):
        def forbidden(_text):
            raise AssertionError("parser called")

        for text, style in (("", "watercolor"), ("x" * 501, "watercolor"), ("cat", "x" * 121)):
            result = extract_graph(text, style, nlp=forbidden)
            self.assertEqual(result["reason"], "input_bounds")
            self.assertTrue(result["requires_fact_review"])
            self.assertEqual(
                set(result),
                {
                    "revision",
                    "status",
                    "facts",
                    "reason",
                    "render_admitted",
                    "requires_fact_review",
                    "local_omissions",
                    "syntax_issues",
                    "renderer_prompt_preview",
                },
            )

    def test_nominal_audit_stops_bounded_adverb_laundering(self):
        rows = json.loads((ROOT / "nominal-audit-results.json").read_text())["rows"]
        for row in rows:
            result = audit_nominal_spans(row["parser_row"], row["labels"])
            self.assertEqual(result, row["result"])
            self.assertIs(result["accepted"], row["expected"])
        # The old bounded grammar alone accepts an adverb as part of a noun.
        # Its actual graph must fail the newly required learned noun audit.
        first = rows[0]
        old = plan_bounded_description(first["text"], "watercolor", 42)
        facts = old.plan.scene_facts
        labels = [item.label for item in (*facts.subjects, *facts.objects)]
        self.assertIn("cat slowly", labels)
        self.assertFalse(audit_nominal_spans(first["parser_row"], labels)["accepted"])

    def test_source_bound_projection_and_refusals(self):
        rows = {r["id"]: r for r in json.loads((ROOT / "syntax-results.json").read_text())["rows"]}
        cat = graph_from_row(rows["01"], "watercolor")
        self.assertEqual(cat["status"], "omission_review")
        self.assertFalse(cat["render_admitted"])
        self.assertEqual(cat["local_omissions"][0]["local_text"], "London")
        self.assertNotIn("london", json.dumps(cat["facts"]).lower())
        self.assertNotIn("london", cat["renderer_prompt_preview"].lower())
        self.assertEqual(cat["facts"]["subjects"][0]["actions"], ["chasing mouse in dark alleyway"])
        self.assertEqual(cat["facts"]["objects"][0]["label"], "mouse")
        self.assertEqual(cat["facts"]["setting"]["label"], "dark alleyway")
        self.assertEqual(len(cat["facts"]["objects"]), 1)
        # Predicted place names remain private even without capitalization.
        lower = json.loads(json.dumps(rows["01"]))
        lower["text"] = lower["text"].replace("London", "london")
        lower["tokens"][10]["text"] = "london"
        lower["tokens"][10]["pos"] = "NOUN"
        lower["entities"][0]["text"] = "london"
        hidden = graph_from_row(lower, "watercolor")
        self.assertEqual(hidden["status"], "omission_review")
        self.assertNotIn("london", hidden["renderer_prompt_preview"])
        self.assertEqual(hidden["local_omissions"][0]["local_text"], "london")
        for case in ("05", "06", "07", "08", "11", "12", "14", "17"):
            result = graph_from_row(rows[case], "watercolor")
            self.assertEqual(result["status"], "draft_ready", case)
            self.assertFalse(result["render_admitted"])
        counts = graph_from_row(rows["05"], "watercolor")["facts"]
        self.assertEqual(counts["subjects"][0]["count"], 2)
        self.assertEqual(counts["objects"][0]["count"], 3)
        absence = graph_from_row(rows["17"], "watercolor")["renderer_prompt_preview"]
        self.assertIn("Constraints: no dogs", absence)
        for case in ("09", "13", "16", "19", "20"):
            result = graph_from_row(rows[case], "watercolor")
            self.assertIsNone(result["facts"], case)
            self.assertFalse(result["render_admitted"])
        # Style cannot recover withheld names or bypass original privacy checks.
        self.assertIsNone(graph_from_row(rows["01"], "London watercolor")["facts"])
        self.assertIsNone(graph_from_row(rows["01"], "x" * 121)["facts"])


if __name__ == "__main__":
    unittest.main()
