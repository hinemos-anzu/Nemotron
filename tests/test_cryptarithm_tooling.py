from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cryptarithm_inventory as inventory
import cryptarithm_kaggle_run as kaggle_run
import cryptarithm_solver as solver
import cryptarithm_generate_verified_cot as cot
import cryptarithm_build_corpus_patch as patch_builder
import cryptarithm_validate_corpus_patch as patch_validator


class CryptarithmSolverTests(unittest.TestCase):
    def solve(self, question: str, answer: str, example_output: str, category: str = "cryptarithm_deduce") -> solver.SolverResult:
        row = {
            "problem_id": "case",
            "category": category,
            "question": question,
            "answer": answer,
            "examples": json.dumps([{"input": "AB?CD", "output": example_output}]),
        }
        return solver.solve_record(row)

    def test_core_rule_candidates_verify(self) -> None:
        cases = [
            ("forward_concat", "EFGH", "ABCD"),
            ("reverse_concat", "GHEF", "CDAB"),
            ("reverse_left", "FEGH", "BACD"),
            ("reverse_right", "EFHG", "ABDC"),
            ("reverse_both", "FEHG", "BADC"),
            ("interleave_lr", "EGFH", "ACBD"),
            ("interleave_rl", "GEHF", "CADB"),
        ]
        for expected_rule, answer, example_output in cases:
            with self.subTest(expected_rule=expected_rule):
                result = self.solve("EF?GH", answer, example_output)
                self.assertTrue(result.verified)
                self.assertEqual(result.rule_type, expected_rule)
                self.assertEqual(result.solver_answer, answer)

    def test_operator_conditioned_rule(self) -> None:
        row = {
            "problem_id": "op_conditioned",
            "category": "cryptarithm_deduce",
            "question": "EF#GH",
            "answer": "GHEF",
            "examples": json.dumps(
                [
                    {"input": "AB@CD", "output": "ABCD"},
                    {"input": "IJ#KL", "output": "KLIJ"},
                ]
            ),
        }
        result = solver.solve_record(row)
        self.assertTrue(result.verified)
        self.assertEqual(result.rule_type, "operator_conditioned_rule:#:reverse_concat")

    def test_no_examples_are_not_verified_by_fallback(self) -> None:
        result = solver.solve_record(
            {
                "problem_id": "no_examples",
                "category": "cryptarithm_deduce",
                "question": "EF?GH",
                "answer": "EFGH",
                "examples": "",
            }
        )
        self.assertFalse(result.verified)
        self.assertEqual(result.failure_type, "guess failure")
        self.assertEqual(result.notes, "no examples; refusing fallback verification")

    def test_malformed_question_is_extraction_failure(self) -> None:
        result = solver.solve_record(
            {
                "problem_id": "bad",
                "category": "cryptarithm_guess",
                "question": "no operator here",
                "answer": "EFGH",
                "examples": json.dumps([{"input": "AB?CD", "output": "ABCD"}]),
            }
        )
        self.assertFalse(result.verified)
        self.assertEqual(result.failure_type, "extraction failure")

    def test_clean_answer_normalizes_common_formats(self) -> None:
        self.assertEqual(solver.clean_answer(r"Therefore, the answer is \\boxed{ABCD}."), "ABCD")
        self.assertEqual(solver.clean_answer("answer: ABCD."), "ABCD")
        self.assertEqual(solver.clean_answer("'ABCD'"), "ABCD")


class CryptarithmPipelineTests(unittest.TestCase):
    def test_inventory_filters_crypto_records(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "cryptarithm" / "tiny_problems.jsonl"
        rows = inventory.build_inventory(inventory.load_records([fixture]), corpus_problem_ids=set())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["problem_id"], "json_forward")
        self.assertEqual(rows[0]["category"], "cryptarithm_deduce")

    def test_coverage_and_cot_skip_unverified_rows(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "cryptarithm" / "tiny_inventory.csv"
        with tempfile.TemporaryDirectory() as tmp:
            coverage_path = Path(tmp) / "coverage.csv"
            cot_path = Path(tmp) / "cot.jsonl"
            report_path = Path(tmp) / "report.md"
            with fixture.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            coverage_rows = solver.write_coverage(rows, coverage_path)
            self.assertEqual(sum(row["verified"] == "true" for row in coverage_rows), 2)
            self.assertEqual(sum(row["verified"] == "false" for row in coverage_rows), 1)
            generated = cot.write_jsonl(coverage_rows, cot_path)
            cot.write_report(coverage_rows, report_path, generated, cot_path)
            self.assertEqual(generated, 2)
            payloads = [json.loads(line) for line in cot_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["problem_id"] for row in payloads}, {"forward", "reverse"})
            self.assertIn("no_examples", report_path.read_text(encoding="utf-8"))

    def test_corpus_patch_builder_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cot_path = Path(tmp) / "cot.jsonl"
            patch_path = Path(tmp) / "patch.jsonl"
            cot_path.write_text(
                json.dumps(
                    {
                        "problem_id": "forward",
                        "category": "cryptarithm_deduce",
                        "rule_type": "forward_concat",
                        "question": "EF?GH",
                        "solver_answer": "EFGH",
                        "verified": True,
                        "reasoning_text": "Therefore, the answer is \\boxed{EFGH}.",
                        "mask_policy": "final_answer_and_rule_application",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            count, skipped = patch_builder.build_patch(cot_path, patch_path)
            self.assertEqual((count, skipped), (1, 0))
            self.assertEqual(patch_validator.validate_patch(patch_path, require_rows=True), 1)
            row = json.loads(patch_path.read_text(encoding="utf-8"))
            self.assertEqual(row["metadata"]["source"], "cryptarithm_solver_verified_cot")
            self.assertTrue(row["metadata"]["verified"])
            self.assertEqual(len(row["messages"]), 2)

    def test_corpus_patch_builder_skips_unverified_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cot_path = Path(tmp) / "cot.jsonl"
            patch_path = Path(tmp) / "patch.jsonl"
            cot_path.write_text(
                json.dumps(
                    {
                        "problem_id": "unsafe",
                        "category": "cryptarithm_deduce",
                        "rule_type": "forward_concat",
                        "question": "EF?GH",
                        "solver_answer": "EFGH",
                        "verified": False,
                        "reasoning_text": "unverified",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            count, skipped = patch_builder.build_patch(cot_path, patch_path)
            self.assertEqual((count, skipped), (0, 1))
            self.assertEqual(patch_path.read_text(encoding="utf-8"), "")
            with self.assertRaises(SystemExit):
                patch_validator.validate_patch(patch_path, require_rows=True)

    def test_cli_pipeline_with_fixture_files(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "cryptarithm" / "tiny_problems.jsonl"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inventory_path = tmp_path / "inventory.csv"
            coverage_path = tmp_path / "coverage.csv"
            cot_path = tmp_path / "cot.jsonl"
            report_path = tmp_path / "report.md"
            patch_path = tmp_path / "patch.jsonl"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "cryptarithm_inventory.py"),
                    "--inputs",
                    str(fixture),
                    "--output",
                    str(inventory_path),
                ],
                check=True,
                cwd=ROOT,
            )
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "cryptarithm_solver.py"), "--inventory", str(inventory_path), "--output", str(coverage_path)],
                check=True,
                cwd=ROOT,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "cryptarithm_generate_verified_cot.py"),
                    "--coverage",
                    str(coverage_path),
                    "--output",
                    str(cot_path),
                    "--report",
                    str(report_path),
                ],
                check=True,
                cwd=ROOT,
            )
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "cryptarithm_build_corpus_patch.py"), "--input", str(cot_path), "--output", str(patch_path), "--strict"],
                check=True,
                cwd=ROOT,
            )
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "cryptarithm_validate_corpus_patch.py"), "--patch", str(patch_path), "--require-rows"],
                check=True,
                cwd=ROOT,
            )

            with coverage_path.open("r", encoding="utf-8", newline="") as handle:
                coverage_rows = list(csv.DictReader(handle))
            self.assertEqual(len(coverage_rows), 1)
            self.assertEqual(coverage_rows[0]["verified"], "true")
            self.assertEqual(len(cot_path.read_text(encoding="utf-8").splitlines()), 1)
            self.assertIn("cryptarithm_deduce", report_path.read_text(encoding="utf-8"))

    def test_kaggle_runner_discovers_and_summarizes_fixture_data(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "cryptarithm" / "tiny_problems.jsonl"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input" / "dataset"
            output_dir = tmp_path / "working" / "cryptarithm"
            input_dir.mkdir(parents=True)
            (input_dir / "problems.jsonl").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

            args = type(
                "Args",
                (),
                {
                    "input_dir": [str(tmp_path / "input")],
                    "output_dir": str(output_dir),
                    "strict_patch": True,
                    "require_patch_rows": True,
                    "skip_empty_patch": True,
                },
            )()
            result = kaggle_run.run_pipeline(args)

            self.assertEqual(result["verified_cot_rows"], 1)
            self.assertEqual(result["patch_rows"], 1)
            self.assertTrue((output_dir / "cryptarithm_realdata_summary.md").exists())
            summary = (output_dir / "cryptarithm_realdata_summary.md").read_text(encoding="utf-8")
            self.assertIn("Cryptarithm Kaggle Coverage Summary", summary)
            self.assertIn("cryptarithm_deduce", summary)


if __name__ == "__main__":
    unittest.main()
