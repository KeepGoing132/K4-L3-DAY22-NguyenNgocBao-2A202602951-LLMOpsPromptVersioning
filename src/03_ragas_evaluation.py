"""Bước 3: đánh giá 50 QA × hai prompt bằng bốn metric RAGAS thật."""
import config
import json
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_community.cache import SQLAlchemyCache
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig
from prompts import SYSTEM_V1, SYSTEM_V2, PROMPT_V1, PROMPT_V2, PROMPTS
from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from utils.evidence import EVIDENCE_DIR, capture_log
from qa_pairs import QA_PAIRS

ROOT = Path(__file__).resolve().parent.parent
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def setup_vectorstore():
    """Tạo FAISS với chunk_size=500, overlap=50 như các bước trước."""
    return build_vectorstore(split_text(load_knowledge_base()), get_embeddings())


def run_rag(retriever, llm, prompt, question: str) -> dict:
    """Giữ contexts là list[str]; chỉ ghép khi truyền cho prompt."""
    contexts = [doc.page_content for doc in retriever.invoke(question)]
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": "\n\n".join(contexts), "question": question,
    })
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """Chạy toàn bộ QA; lưu answers và contexts để đối chiếu điểm."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    results = []
    print(f"\n🚀 Chạy {len(QA_PAIRS)} câu hỏi với prompt {prompt_version}")
    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, PROMPTS[prompt_version], qa["question"])
        results.append({"question": qa["question"], "reference": qa["reference"],
                        "answer": out["answer"], "contexts": out["contexts"]})
        print(f"[{i:02d}/{len(QA_PAIRS)}] {qa['question']}", flush=True)
    (ROOT / "data" / f"rag_outputs_{prompt_version}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """Ánh xạ bốn trường vào SingleTurnSample, không đưa reference cho LLM RAG."""
    return EvaluationDataset(samples=[
        SingleTurnSample(user_input=r["question"], response=r["answer"],
                         retrieved_contexts=r["contexts"], reference=r["reference"])
        for r in rag_results
    ])


def run_ragas_eval(rag_results: list, version: str) -> dict:
    """Chấm đủ bốn metric; không xuất điểm thành công nếu có NaN/thiếu sample."""
    print(f"\n📐 RAGAS {version}: có thể mất 5-30 phút.", flush=True)
    if config.PROVIDER == "gemini":
        print(f"  Evaluator Google: {config.RAGAS_GEMINI_MODEL}")
    # Gemini hiện chỉ hỗ trợ một candidate mỗi request; vẫn dùng metric RAGAS thật.
    relevance = replace(answer_relevancy, strictness=1) if config.PROVIDER == "gemini" else answer_relevancy
    evaluation_model = get_llm(temperature=0, model=config.RAGAS_GEMINI_MODEL) if config.PROVIDER == "gemini" else get_llm(temperature=0)
    # Chia sẻ limiter giữa các metric: cho phép chờ API đồng thời mà không burst.
    if config.PROVIDER == "gemini":
        # NullPool đóng kết nối sau mỗi thao tác, tránh khóa file trên Windows.
        cache_path = ROOT / "data" / "ragas_judge_cache.db"
        evaluation_model.cache = SQLAlchemyCache(create_engine(f"sqlite:///{cache_path}", poolclass=NullPool))
        evaluation_model.rate_limiter = InMemoryRateLimiter(
            requests_per_second=0.4, check_every_n_seconds=0.1, max_bucket_size=1,
        )
    fingerprint = hashlib.sha256(json.dumps({
        "rows": rag_results, "provider": config.PROVIDER,
        "judge": {"gemini": config.RAGAS_GEMINI_MODEL, "openai": config.OPENAI_MODEL,
                  "anthropic": config.ANTHROPIC_MODEL, "ollama": config.OLLAMA_MODEL,
                  "openrouter": config.OPENROUTER_MODEL}[config.PROVIDER],
        "embedding": config.GEMINI_EMBEDDING_MODEL if config.PROVIDER == "gemini" else
                     config.OLLAMA_EMBEDDING_MODEL if config.PROVIDER == "ollama" else config.OPENAI_EMBEDDING_MODEL,
        "metrics": METRIC_NAMES, "strictness": relevance.strictness,
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    checkpoint_path = ROOT / "data" / f"ragas_checkpoint_{version}.json"
    sample_scores = {key: [] for key in METRIC_NAMES}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("fingerprint") == fingerprint:
            saved = checkpoint["sample_scores"]
            lengths = {len(saved[key]) for key in METRIC_NAMES}
            if len(lengths) != 1 or max(lengths) > len(rag_results) or not all(
                    np.all(np.isfinite(np.asarray(saved[key], dtype=float))) for key in METRIC_NAMES):
                raise ValueError("Checkpoint RAGAS không hợp lệ")
            sample_scores = saved
    completed = len(sample_scores[METRIC_NAMES[0]])
    print(f"  Đã lưu {completed}/{len(rag_results)} QA; chấm tiếp theo nhóm 5.")
    for start in range(completed, len(rag_results), 5):
        batch = rag_results[start:start + 5]
        result = evaluate(
            build_ragas_dataset(batch),
            metrics=[faithfulness, relevance, context_recall, context_precision],
            llm=evaluation_model, embeddings=get_embeddings(),
            run_config=RunConfig(timeout=180, max_retries=3, max_workers=8 if config.PROVIDER == "gemini" else 2),
            raise_exceptions=True,
        )
        batch_scores = {}
        for key in METRIC_NAMES:
            values = np.asarray(result[key], dtype=float)
            if len(values) != len(batch) or not np.all(np.isfinite(values)):
                raise ValueError(f"{version}/{key}: thiếu điểm hoặc có NaN; cần chạy lại")
            batch_scores[key] = values.tolist()
        for key in METRIC_NAMES:
            sample_scores[key].extend(batch_scores[key])
        temp_path = checkpoint_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps({"fingerprint": fingerprint, "sample_scores": sample_scores}, allow_nan=False), encoding="utf-8")
        temp_path.replace(checkpoint_path)
        print(f"  💾 {version}: đã lưu {start + len(batch)}/{len(rag_results)} QA.", flush=True)
    scores = {key: float(np.mean(sample_scores[key])) for key in METRIC_NAMES}
    for key in METRIC_NAMES:
        print(f"  {key:30s}: {scores[key]:.4f}")
    (ROOT / "data" / f"ragas_samples_{version}.json").write_text(
        json.dumps(sample_scores, indent=2, allow_nan=False), encoding="utf-8")
    return scores


def load_saved_outputs(version):
    """Chỉ tái dùng kết quả đầy đủ đã chạy thật khi người chạy chọn --reuse-outputs."""
    path = ROOT / "data" / f"rag_outputs_{version}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != len(QA_PAIRS):
        raise ValueError(f"{path.name}: cần đủ 50 QA")
    for row, qa in zip(rows, QA_PAIRS):
        if (row.get("question") != qa["question"] or row.get("reference") != qa["reference"]
                or not isinstance(row.get("answer"), str) or not row["answer"].strip()
                or not isinstance(row.get("contexts"), list) or not row["contexts"]
                or not all(isinstance(context, str) and context for context in row["contexts"])):
            raise ValueError(f"{path.name}: QA/context/answer không hợp lệ")
    print(f"♻️ Dùng {len(rows)} câu trả lời đã lưu trong {path.name}")
    return rows


def main(reuse_outputs=False):
    """Chạy hai phiên bản, lưu report và bản sao evidence sau đánh giá đầy đủ."""
    if not config.validate():
        raise SystemExit(1)
    with capture_log("03_ragas_evaluation_log.txt"):
        if reuse_outputs:
            results = {version: load_saved_outputs(version) for version in PROMPTS}
        else:
            vectorstore = setup_vectorstore()
            results = {version: collect_rag_outputs(vectorstore, version) for version in PROMPTS}
        scores = {version: run_ragas_eval(results[version], version) for version in PROMPTS}
        print("\n" + "=" * 65)
        print(f"{'Metric':30s} {'V1':>8} {'V2':>8}  Winner")
        for metric in METRIC_NAMES:
            s1, s2 = scores["v1"][metric], scores["v2"][metric]
            winner = "Tie" if s1 == s2 else "V1" if s1 > s2 else "V2"
            print(f"{metric:30s} {s1:8.4f} {s2:8.4f}  {winner}")
        best_faith = max(s["faithfulness"] for s in scores.values())
        report = {
            "prompt_v1_scores": scores["v1"], "prompt_v2_scores": scores["v2"],
            "target_met": best_faith >= 0.8, "samples_per_version": len(QA_PAIRS),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provider": config.PROVIDER, "retrieval": {"chunk_size": 500, "chunk_overlap": 50, "k": 3},
            "generation_model": config.GEMINI_MODEL if config.PROVIDER == "gemini" else config.PROVIDER,
            "evaluation_model": config.RAGAS_GEMINI_MODEL if config.PROVIDER == "gemini" else config.PROVIDER,
            "system_prompts": {"v1": SYSTEM_V1, "v2": SYSTEM_V2},
            "answer_relevancy_strictness": 1 if config.PROVIDER == "gemini" else answer_relevancy.strictness,
            "reused_outputs": reuse_outputs,
        }
        content = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        (ROOT / "data" / "ragas_report.json").write_text(content, encoding="utf-8")
        (EVIDENCE_DIR / "03_ragas_report.json").write_text(content, encoding="utf-8")
        print("💾 Đã lưu data/ragas_report.json và evidence/03_ragas_report.json")
        if report["target_met"]:
            print(f"✅ Faithfulness = {best_faith:.4f} ≥ 0.8")
        else:
            print(f"⚠️ Faithfulness = {best_faith:.4f} < 0.8; cần cải thiện prompt/retrieval.")
            raise SystemExit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Đánh giá RAGAS với hai phiên bản prompt")
    parser.add_argument("--reuse-outputs", action="store_true", help="Chấm lại các câu trả lời đã lưu; chỉ dùng khi prompt/model/retrieval chưa đổi")
    main(reuse_outputs=parser.parse_args().reuse_outputs)
