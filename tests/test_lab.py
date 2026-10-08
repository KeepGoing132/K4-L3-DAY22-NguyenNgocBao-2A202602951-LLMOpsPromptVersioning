"""Kiểm thử offline; không dùng kết quả mock làm evidence LangSmith/RAGAS."""
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from langsmith.utils import LangSmithConflictError
from utils.data_loader import build_vectorstore, load_knowledge_base, split_text
from qa_pairs import QA_PAIRS, SAMPLE_QUESTIONS

step1 = importlib.import_module("01_langsmith_rag_pipeline")
step2 = importlib.import_module("02_prompt_hub_ab_routing")
step3 = importlib.import_module("03_ragas_evaluation")
step4 = importlib.import_module("04_guardrails_validator")


class KeywordEmbeddings(Embeddings):
    """Embeddings test tất định, đủ kiểm tra index/retrieval không gọi mạng."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(text.lower().count(word)) for word in ("faiss", "neural", "rag")]


class LabTests(unittest.TestCase):
    def test_embedding_cache_batch_retry_and_task_separation(self):
        """Retry batch lỗi không gọi lại batch đạt; query không dùng vector document."""
        from unittest.mock import Mock
        from utils.cached_embeddings import CachedEmbeddings
        backend = Mock()
        backend.embed_documents.side_effect = [[[1.0], [2.0]], RuntimeError("429 retry in 1s"), [[3.0]]]
        backend.embed_query.return_value = [99.0]
        with tempfile.TemporaryDirectory() as temp, patch("utils.cached_embeddings.time.sleep") as sleep:
            cache = CachedEmbeddings(backend, "test-model", Path(temp) / "cache.json", batch_size=2)
            self.assertEqual(cache.embed_documents(["a", "b", "c", "a"]), [[1.0], [2.0], [3.0], [1.0]])
            self.assertEqual(backend.embed_documents.call_count, 3)
            sleep.assert_called_once_with(3.0)
            restored = CachedEmbeddings(backend, "test-model", cache.cache_path)
            self.assertEqual(restored.embed_documents(["c", "a"]), [[3.0], [1.0]])
            self.assertEqual(restored.embed_query("a"), [99.0])
            self.assertEqual(restored.embed_query("a"), [99.0])
            backend.embed_query.assert_called_once_with("a")
        with patch("utils.cached_embeddings.time.sleep") as sleep:
            with self.assertRaises(RuntimeError):
                CachedEmbeddings._retry(lambda: (_ for _ in ()).throw(RuntimeError("429 quota PerDay")))
            sleep.assert_not_called()

    def test_faiss_lcel_and_dataset(self):
        """Context retrieve phải tới LLM và tới đúng trường RAGAS."""
        chunks = split_text(load_knowledge_base())
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 500 for chunk in chunks))
        store = build_vectorstore([
            "FAISS is a library for similarity search.",
            "Neural networks use layers.",
            "RAG retrieves context before answering.",
        ], KeywordEmbeddings())
        seen = []

        def answer(prompt_value):
            messages = prompt_value.to_messages()
            seen.append(messages)
            self.assertIn("FAISS is a library", messages[0].content)
            self.assertEqual(messages[-1].content, "What is FAISS?")
            return "FAISS performs similarity search."

        llm = RunnableLambda(answer)
        with patch.object(step1, "get_llm", return_value=llm):
            chain, retriever = step1.build_rag_chain(store)
        self.assertEqual(step1.ask(chain, "What is FAISS?"), "FAISS performs similarity search.")
        for version, prompt in step3.PROMPTS.items():
            out = step3.run_rag(retriever, llm, prompt, "What is FAISS?")
            self.assertEqual(len(out["contexts"]), 3)
            ab = step2.ask_ab(retriever, llm, prompt, "What is FAISS?", version)
            self.assertEqual(ab["contexts"], out["contexts"])
            dataset = step3.build_ragas_dataset([{
                "question": "What is FAISS?", "reference": "Similarity search library.", **out,
            }])
            sample = dataset.samples[0]
            self.assertEqual(sample.user_input, "What is FAISS?")
            self.assertEqual(sample.response, out["answer"])
            self.assertEqual(sample.retrieved_contexts, out["contexts"])
            self.assertEqual(sample.reference, "Similarity search library.")
        self.assertEqual(len(seen), 5)

    def test_routing_across_processes(self):
        names = [step2.get_prompt_version(f"req-{i:04d}") for i in range(50)]
        self.assertEqual(set(names), {step2.PROMPT_V1_NAME, step2.PROMPT_V2_NAME})
        command = (
            "import importlib; m=importlib.import_module('02_prompt_hub_ab_routing'); "
            "print(m.get_prompt_version('req-0000'))"
        )
        for seed in ("1", "987"):
            result = subprocess.check_output(
                [sys.executable, "-c", command], cwd=Path(step2.__file__).parent,
                env={**os.environ, "PYTHONHASHSEED": seed}, text=True, encoding="utf-8",
            )
            self.assertEqual(result.strip(), names[0])

    def test_hub_errors_not_hidden(self):
        from unittest.mock import Mock
        client = Mock()
        client.push_prompt.side_effect = LangSmithConflictError("409 Nothing to commit")
        step2.push_prompts_to_hub(client)
        client.push_prompt.side_effect = RuntimeError("authentication failed")
        with self.assertRaises(RuntimeError):
            step2.push_prompts_to_hub(client)
        client.pull_prompt.side_effect = RuntimeError("Hub unavailable")
        with self.assertRaises(RuntimeError):
            step2.pull_prompts_from_hub(client)

    def test_shared_prompts_and_qa(self):
        self.assertEqual(len(QA_PAIRS), 50)
        self.assertEqual(SAMPLE_QUESTIONS, [q["question"] for q in QA_PAIRS])
        self.assertEqual(step2.SYSTEM_V1, step3.SYSTEM_V1)
        self.assertEqual(step2.SYSTEM_V2, step3.SYSTEM_V2)
        self.assertNotEqual(step2.SYSTEM_V1, step2.SYSTEM_V2)

    def test_ragas_mean_and_failed_samples(self):
        """Không bỏ qua NaN để làm đẹp điểm tổng hợp."""
        rows = [{"question": "Q", "answer": "A", "reference": "R", "contexts": ["C"]}] * 2
        fake_result = {key: [0.6, 1.0] for key in step3.METRIC_NAMES}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            with patch.object(step3, "ROOT", root), patch.object(step3, "get_llm"), \
                 patch.object(step3, "get_embeddings"), \
                 patch.object(step3, "evaluate", return_value=fake_result) as evaluate:
                scores = step3.run_ragas_eval(rows, "v1")
                self.assertTrue(all(score == 0.8 for score in scores.values()))
                self.assertEqual(len(evaluate.call_args.kwargs["metrics"]), 4)
                relevance = evaluate.call_args.kwargs["metrics"][1]
                self.assertEqual(relevance.strictness, 1 if step3.config.PROVIDER == "gemini" else 3)
                self.assertEqual(step3.answer_relevancy.strictness, 3)
                fake_result["faithfulness"] = [0.9, float("nan")]
                (root / "data" / "ragas_checkpoint_v1.json").unlink()
                with self.assertRaises(ValueError):
                    step3.run_ragas_eval(rows, "v1")

    def test_ragas_checkpoint_resumes_only_matching_inputs(self):
        """Không mất batch đã chấm thật; đổi answer thì không dùng điểm cũ."""
        rows = [{"question": f"Q{i}", "answer": "A", "reference": "R", "contexts": ["C"]} for i in range(7)]
        first = {key: [0.9] * 5 for key in step3.METRIC_NAMES}
        second = {key: [0.7] * 2 for key in step3.METRIC_NAMES}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            with patch.object(step3, "ROOT", root), patch.object(step3, "get_llm"), \
                 patch.object(step3, "get_embeddings"), patch.object(step3, "evaluate") as evaluate:
                evaluate.side_effect = [first, RuntimeError("quota day")]
                with self.assertRaises(RuntimeError):
                    step3.run_ragas_eval(rows, "v1")
                evaluate.reset_mock()
                evaluate.side_effect = None
                evaluate.return_value = second
                scores = step3.run_ragas_eval(rows, "v1")
                evaluate.assert_called_once()
                self.assertEqual(len(evaluate.call_args.args[0].samples), 2)
                self.assertAlmostEqual(scores["faithfulness"], (5 * 0.9 + 2 * 0.7) / 7)
                rows[0]["answer"] = "Changed answer"
                evaluate.reset_mock()
                evaluate.side_effect = [first, second]
                step3.run_ragas_eval(rows, "v1")
                self.assertEqual(evaluate.call_count, 2)

    def test_actual_guard_fix_outputs(self):
        """Kiểm tra tích hợp Guard, không chỉ validate() của class."""
        step4.demo_pii_guard()
        step4.demo_json_guard()

    def test_ragas_threshold_affects_exit_status(self):
        """Report điểm thấp vẫn được lưu, nhưng run_all không báo PASS."""
        from contextlib import nullcontext
        import json

        for faithfulness, fails in ((0.79, True), (0.8, False)):
            with self.subTest(faithfulness=faithfulness), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "data").mkdir()
                (root / "evidence").mkdir()
                scores = {key: faithfulness for key in step3.METRIC_NAMES}
                with patch.object(step3.config, "validate", return_value=True), \
                     patch.object(step3, "ROOT", root), \
                     patch.object(step3, "EVIDENCE_DIR", root / "evidence"), \
                     patch.object(step3, "capture_log", return_value=nullcontext()), \
                     patch.object(step3, "setup_vectorstore"), \
                     patch.object(step3, "collect_rag_outputs", return_value=[]), \
                     patch.object(step3, "run_ragas_eval", return_value=scores):
                    if fails:
                        with self.assertRaises(SystemExit) as error:
                            step3.main()
                        self.assertEqual(error.exception.code, 1)
                    else:
                        step3.main()
                report = json.loads((root / "data" / "ragas_report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["target_met"], not fails)
                self.assertEqual((root / "data" / "ragas_report.json").read_bytes(),
                                 (root / "evidence" / "03_ragas_report.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
