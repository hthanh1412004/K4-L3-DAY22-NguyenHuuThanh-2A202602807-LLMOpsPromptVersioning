"""Offline checks for LCEL wiring, routing, RAGAS schema and actual Guard fixes."""
import importlib
import json
import os
from pathlib import Path
import sys
import unittest
import asyncio
import contextlib
import io
import tempfile
from unittest.mock import patch
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
from langchain_core.outputs import LLMResult, Generation
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from guardrails import Guard
from guardrails.validator_base import OnFailAction

step1 = importlib.import_module("01_langsmith_rag_pipeline")
step2 = importlib.import_module("02_prompt_hub_ab_routing")
step3 = importlib.import_module("03_ragas_evaluation")
step4 = importlib.import_module("04_guardrails_validator")
from utils.ragas_evaluator import JSONRetryLLM, LangchainLLMWrapper


class LabTests(unittest.TestCase):
    def test_server_errors_retry_but_quota_errors_do_not(self):
        adapter = JSONRetryLLM(RunnableLambda(lambda value: value))
        valid = LLMResult(generations=[[Generation(text='{"ok":true}')]])
        with patch.object(LangchainLLMWrapper, "generate_text", side_effect=[RuntimeError("500 INTERNAL"), valid]) as generate, \
             patch("utils.ragas_evaluator.time.sleep") as sleep:
            self.assertIs(adapter.generate_text("fixture"), valid)
            self.assertEqual(generate.call_count, 2)
            sleep.assert_any_call(5)
        with patch.object(LangchainLLMWrapper, "agenerate_text", side_effect=[RuntimeError("503 UNAVAILABLE"), valid]) as generate, \
             patch("utils.ragas_evaluator.asyncio.sleep") as sleep:
            self.assertIs(asyncio.run(adapter.agenerate_text("fixture")), valid)
            self.assertEqual(generate.call_count, 2)
            sleep.assert_any_await(5)
        for message in ("429 RESOURCE_EXHAUSTED", "403 PERMISSION_DENIED"):
            with patch.object(LangchainLLMWrapper, "generate_text", side_effect=RuntimeError(message)) as generate:
                with self.assertRaisesRegex(RuntimeError, message):
                    adapter.generate_text("fixture")
                self.assertEqual(generate.call_count, 1)

    def test_runner_reports_partial_without_false_pass(self):
        import run_all
        fake_module = Mock()
        fake_module.main.return_value = "partial"
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with patch.object(run_all, "__file__", str(Path(directory) / "src" / "run_all.py")), \
                 patch.object(run_all.importlib, "import_module", return_value=fake_module), \
                 patch.object(sys, "argv", ["run_all.py", "--step", "3", "--eval-limit", "5"]), \
                 contextlib.redirect_stdout(output):
                run_all.main()
            fake_module.main.assert_called_once_with(eval_limit=5)
            self.assertIn("ĐÃ LƯU TIẾN ĐỘ", output.getvalue())
            self.assertNotIn("PASS", output.getvalue())
            self.assertNotIn("HOÀN THÀNH", output.getvalue())

    def test_incremental_budget_resumes_and_lock_prevents_duplicate_writer(self):
        from utils.evaluation_lock import evaluation_lock
        sample = dict(question="What is FAISS?", reference="Vector index",
                      answer="Vector index", contexts=["FAISS indexes vectors."])
        scores = {key: [0.9] for key in (
            "faithfulness", "answer_relevancy", "context_recall", "context_precision")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            with patch.object(step3, "evaluate", return_value=scores) as evaluate:
                self.assertIsNone(step3.evaluate_with_checkpoints([sample] * 3, None, None, path, "stable", limit=2))
                self.assertEqual(evaluate.call_count, 2)
            with patch.object(step3, "evaluate", return_value=scores) as evaluate:
                final = step3.evaluate_with_checkpoints([sample] * 3, None, None, path, "stable", limit=2)
                self.assertEqual(evaluate.call_count, 1)
                self.assertAlmostEqual(final["faithfulness"], 0.9)
            lock_path = Path(directory) / "ragas.lock"
            with evaluation_lock(lock_path):
                with self.assertRaisesRegex(RuntimeError, "khác đang chạy"):
                    with evaluation_lock(lock_path):
                        self.fail("Second writer acquired the same lock")
            with evaluation_lock(lock_path):
                pass

    def test_paced_candidates_are_separate_requests(self):
        adapter = JSONRetryLLM(RunnableLambda(lambda value: value), tokens_per_minute=12000)
        results = [LLMResult(generations=[[Generation(text=json.dumps({"candidate": i}))]]) for i in range(3)]
        with patch.object(adapter, "_reserve_delay", return_value=0), \
             patch.object(LangchainLLMWrapper, "agenerate_text", side_effect=results) as generate:
            output = asyncio.run(adapter.agenerate_text("fixture", n=3))
            self.assertEqual(len(output.generations[0]), 3)
            self.assertEqual(generate.call_count, 3)
            self.assertTrue(all(call.kwargs["n"] == 1 for call in generate.call_args_list))

    def test_scoring_resumes_after_timeout_and_rejects_nan(self):
        sample = dict(question="What is FAISS?", reference="Vector index",
                      answer="Vector index", contexts=["FAISS indexes vectors."])
        scores = {key: [0.9] for key in (
            "faithfulness", "answer_relevancy", "context_recall", "context_precision")}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "samples.json"
            with patch.object(step3, "evaluate", side_effect=[scores, TimeoutError(), TimeoutError()]):
                with self.assertRaisesRegex(RuntimeError, "Đã lưu 1 câu"):
                    step3.evaluate_with_checkpoints([sample, sample], None, None, checkpoint, "fixture")
            self.assertEqual(len(json.loads(checkpoint.read_text())["rows"]), 1)
            with patch.object(step3, "evaluate", return_value=scores) as evaluate:
                result = step3.evaluate_with_checkpoints([sample, sample], None, None, checkpoint, "fixture")
                self.assertEqual(evaluate.call_count, 1)
                self.assertEqual(result["faithfulness"], 0.9)
            with patch.object(step3, "evaluate", return_value={**scores, "faithfulness": [float("nan")]}):
                with self.assertRaisesRegex(ValueError, "NaN"):
                    step3.evaluate_with_checkpoints([sample], None, None, checkpoint, "changed-inputs")

    def test_daily_quota_stops_before_batch(self):
        from google.genai.errors import ClientError
        error = ClientError(429, {"error": {
            "code": 429, "message": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
            "status": "RESOURCE_EXHAUSTED",
        }})
        with patch.object(config, "EVAL_PROVIDER", "gemini"), \
             patch("google.genai.Client") as client:
            generate = client.return_value.__enter__.return_value.models.generate_content
            generate.side_effect = error
            with self.assertRaisesRegex(RuntimeError, "hết quota theo ngày"):
                step3.check_evaluator_quota()
            self.assertEqual(generate.call_count, 1)

    def test_separate_evaluator_requires_its_key(self):
        output = io.StringIO()
        with patch.multiple(config, PROVIDER="gemini", EVAL_PROVIDER="openai",
                            LANGSMITH_API_KEY="test-value", GOOGLE_API_KEY="test-value", OPENAI_API_KEY=""), \
             patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}), \
             contextlib.redirect_stdout(output):
            self.assertFalse(config.validate())
        self.assertIn("API key cho evaluator openai", output.getvalue())

    def test_evaluator_retries_malformed_unicode_json(self):
        malformed = LLMResult(generations=[[Generation(text=r'{"question":"bad\u12ọc","noncommittal":0}')]])
        valid = LLMResult(generations=[[Generation(text='{"question":"What is ML?","noncommittal":0}')]])
        adapter = JSONRetryLLM(RunnableLambda(lambda value: value))
        with patch.object(LangchainLLMWrapper, "generate_text", side_effect=[malformed, valid]) as generate:
            self.assertIs(adapter.generate_text("fixture"), valid)
            self.assertEqual(generate.call_count, 2)
        with patch.object(LangchainLLMWrapper, "agenerate_text", side_effect=[malformed, valid]) as generate:
            self.assertIs(asyncio.run(adapter.agenerate_text("fixture")), valid)
            self.assertEqual(generate.call_count, 2)

    def test_routing(self):
        versions = [step2.get_prompt_version(f"req-{i:04d}") for i in range(50)]
        self.assertEqual(set(versions), {step2.PROMPT_V1_NAME, step2.PROMPT_V2_NAME})
        self.assertEqual(versions, [step2.get_prompt_version(f"req-{i:04d}") for i in range(50)])

    def test_rag_wiring_and_dataset(self):
        docs = [Document(page_content="FAISS indexes vectors."), Document(page_content="RAG retrieves context.")]
        retriever = RunnableLambda(lambda question: docs)
        def respond(prompt):
            rendered = prompt.to_string()
            self.assertIn("FAISS indexes vectors.", rendered)
            self.assertIn("What is FAISS?", rendered)
            return "FAISS indexes vectors."
        llm = RunnableLambda(respond)
        class Store:
            def as_retriever(inner, search_kwargs):
                self.assertEqual(search_kwargs, {"k": 3})
                return retriever
        with patch.object(step1, "get_llm", return_value=llm):
            chain, _ = step1.build_rag_chain(Store())
            self.assertEqual(step1.ask(chain, "What is FAISS?"), "FAISS indexes vectors.")
        out = step3.run_rag(retriever, llm, step3.PROMPT_V1, "What is FAISS?")
        sample = step3.build_ragas_dataset([dict(question="What is FAISS?", reference="Vector index", **out)]).samples[0]
        self.assertEqual(sample.retrieved_contexts, [d.page_content for d in docs])
        self.assertEqual(sample.response, out["answer"])
        self.assertEqual(step2.SYSTEM_V1, step3.SYSTEM_V1)
        self.assertEqual(step2.SYSTEM_V2, step3.SYSTEM_V2)

    def test_pii_guard_fixes_output(self):
        guard = Guard().use(step4.PIIDetector(on_fail=OnFailAction.FIX))
        text = "Email alice@example.com; phone (555) 867-5309; SSN 123-45-6789; card 4532 1234 5678 9010."
        output = guard.validate(text).validated_output
        for kind in ["EMAIL", "PHONE", "SSN", "CREDIT_CARD"]:
            self.assertIn(f"[{kind}_REDACTED]", output)
        for secret in ["alice@example.com", "555", "123-45-6789", "4532"]:
            self.assertNotIn(secret, output)
        self.assertNotIn("([PHONE", output)
        self.assertEqual(guard.validate("Clean text.").validated_output, "Clean text.")

    def test_json_guard(self):
        guard = Guard().use(step4.JSONFormatter(on_fail=OnFailAction.FIX))
        cases = [('{"ok":true}', {"ok": True}),
                 ('```json\n{"text":"don\'t"}\n```', {"text": "don't"}),
                 ("{'name':'Alice'}", {"name": "Alice"}),
                 ('{"items":[1,2,],}', {"items": [1, 2]})]
        for text, expected in cases:
            self.assertEqual(json.loads(guard.validate(text).validated_output), expected)
        self.assertIn("error", json.loads(guard.validate("not JSON").validated_output))


if __name__ == "__main__":
    unittest.main()
