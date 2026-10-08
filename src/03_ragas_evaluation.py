"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import hashlib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS
from prompt_templates import SYSTEM_V1, SYSTEM_V2
from utils.ragas_evaluator import JSONRetryLLM


def check_evaluator_quota():
    """Check Gemini once before scheduling hundreds of evaluation calls."""
    if config.EVAL_PROVIDER != "gemini":
        return
    from google import genai
    from google.genai.errors import ClientError
    with genai.Client(api_key=config.GOOGLE_API_KEY, http_options={
        "timeout": 20000, "retry_options": {"attempts": 1},
    }) as client:
        try:
            client.models.generate_content(
                model=config.GEMINI_EVAL_MODEL, contents='Return only {"ok":true}.',
            )
        except ClientError as exc:
            message = str(exc)
            daily_quota = "GenerateRequestsPerDay" in message or "requests_per_day" in message
            if exc.code == 429 and daily_quota:
                raise RuntimeError(
                    "Gemini đã hết quota theo ngày. Dừng trước khi chấm RAGAS; "
                    "câu trả lời đã lưu vẫn được giữ nguyên. Bật billing cho project Gemini, "
                    "hoặc đặt EVAL_PROVIDER cùng API key của provider còn quota trong .env, "
                    "rồi chạy lại --step 3. Key mới trong cùng project không tăng quota."
                ) from None
            raise


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)

    # Gợi ý: contexts = [doc.page_content for doc in docs]
    contexts = [doc.page_content for doc in docs]

    ctx_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    # Reuse a complete generation checkpoint only when all inputs match.
    fingerprint = hashlib.sha256(json.dumps({
        "prompt": prompt.pretty_repr(), "qa_pairs": QA_PAIRS,
        "knowledge_base": load_knowledge_base(), "provider": config.PROVIDER,
        "models": [config.OPENAI_MODEL, config.GEMINI_MODEL, config.ANTHROPIC_MODEL,
                   config.OLLAMA_MODEL, config.OPENROUTER_MODEL, config.GEMINI_EMBEDDING_MODEL,
                   config.OPENAI_EMBEDDING_MODEL, config.OLLAMA_EMBEDDING_MODEL],
        "retrieval": {"k": 3, "chunk_size": 500, "chunk_overlap": 50},
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cache_path = Path(__file__).parent.parent / ".cache" / f"rag_outputs_{prompt_version}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint and len(cached.get("results", [])) == len(QA_PAIRS):
            print(f"♻️ Dùng lại {len(QA_PAIRS)} kết quả thật đã lưu cho {prompt_version}")
            return cached["results"]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    # Preserve generated answers and individual contexts for review and debugging.
    output_path = Path(__file__).parent.parent / "data" / f"rag_outputs_{prompt_version}.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    cache_path.parent.mkdir(exist_ok=True)
    cache_path.write_text(json.dumps({"fingerprint": fingerprint, "results": results},
                                    ensure_ascii=False), encoding="utf-8")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def evaluate_with_checkpoints(rag_results, llm_eval, emb_eval, path, fingerprint, limit=None):
    """Persist each fully scored sample; resume only matching evaluator inputs."""
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    keys = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    rows = []
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            rows = cached.get("rows", [])
    if len(rows) > len(rag_results) or any(
        set(row) != set(keys) or any(not np.isfinite(v) for v in row.values())
        for row in rows
    ):
        raise ValueError("Checkpoint điểm không hợp lệ; không tính báo cáo")
    print(f"♻️ Đã chấm và lưu {len(rows)}/{len(rag_results)} câu hỏi")
    stop_at = len(rag_results) if limit is None else min(len(rag_results), len(rows) + limit)
    for index in range(len(rows), stop_at):
        print(f"  Đang chấm câu {index + 1}/{len(rag_results)} (4 metrics)...", flush=True)
        # A completed sample is durable even when a later request fails.
        for attempt in range(2):
            try:
                result = evaluate(
                    build_ragas_dataset([rag_results[index]]), metrics=metrics,
                    llm=llm_eval, embeddings=emb_eval, raise_exceptions=True,
                    show_progress=False,
                    run_config=RunConfig(max_workers=1 if llm_eval and llm_eval.tokens_per_minute else 2,
                                         timeout=600, max_retries=2),
                )
                break
            except TimeoutError:
                if attempt == 1:
                    raise RuntimeError(
                        f"TimeoutError khi chấm câu {index + 1}. Đã lưu {len(rows)} câu; "
                        "chạy lại --step 3 để tiếp tục từ câu này."
                    ) from None
                print("  TimeoutError: thử lại câu hiện tại một lần.", flush=True)
        row = {}
        for key in keys:
            values = result[key]
            if len(values) != 1 or values[0] is None or not np.isfinite(values[0]):
                raise ValueError(f"Câu {index + 1}: {key} thiếu/NaN; không lưu điểm")
            row[key] = float(values[0])
        rows.append(row)
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"fingerprint": fingerprint, "rows": rows},
                                        allow_nan=False), encoding="utf-8")
        temporary.replace(path)
        print(f"  ✅ Đã lưu {index + 1}/{len(rag_results)} câu hỏi", flush=True)
    if len(rows) < len(rag_results):
        print(f"⏸ Đã lưu {len(rows)}/{len(rag_results)} câu. Đổi key trong .env nếu cần rồi chạy lại.")
        return None
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def run_ragas_eval(rag_results: list, version: str, limit=None):
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version}; thời gian phụ thuộc tốc độ API.")

    score_fingerprint = hashlib.sha256(json.dumps({
        "results": rag_results, "provider": config.PROVIDER,
        "evaluator_provider": config.EVAL_PROVIDER,
        "evaluator_models": [config.GEMINI_EVAL_MODEL, config.OPENAI_MODEL,
                             config.ANTHROPIC_MODEL, config.OLLAMA_MODEL, config.OPENROUTER_MODEL],
        "embedding_models": [config.GEMINI_EMBEDDING_MODEL, config.OPENAI_EMBEDDING_MODEL,
                             config.OLLAMA_EMBEDDING_MODEL],
        "evaluator_config": "json-mode-separate-candidates-v3-gemma-minimal",
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    score_cache = Path(__file__).parent.parent / ".cache" / f"ragas_scores_{version}.json"
    if score_cache.exists():
        cached = json.loads(score_cache.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == score_fingerprint:
            print(f"♻️ Dùng lại điểm {version} đã chấm hoàn tất bằng cùng cấu hình evaluator")
            return cached["scores"]

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    evaluator = get_llm(provider=config.EVAL_PROVIDER, temperature=0,
                        model=config.GEMINI_EVAL_MODEL if config.EVAL_PROVIDER == "gemini" else None)
    if config.EVAL_PROVIDER == "gemini" and config.GEMINI_EVAL_MODEL.startswith("gemini-"):
        evaluator.response_mime_type = "application/json"
    if config.EVAL_PROVIDER == "gemini" and config.GEMINI_EVAL_MODEL.startswith("gemini-3"):
        evaluator.reasoning_effort = "minimal" if "flash-lite" in config.GEMINI_EVAL_MODEL else "low"
    # Generate n separate requests instead of mutating a shared model's n/temperature.
    # Gemini rejects candidate_count > 1; concurrent mutation also mixes metric jobs.
    is_gemma = config.EVAL_PROVIDER == "gemini" and config.GEMINI_EVAL_MODEL.startswith("gemma-")
    if is_gemma:
        evaluator.reasoning_effort = "minimal"
        evaluator.timeout = 90
        evaluator.max_retries = 2
    llm_eval = JSONRetryLLM(evaluator, bypass_n=True, bypass_temperature=True,
                            tokens_per_minute=12000 if is_gemma else None)
    emb_eval = get_embeddings()

    # Gợi ý:
    #   result = evaluate(
    #       dataset,
    #       metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    #       llm=llm_eval,
    #       embeddings=emb_eval,
    #   )
    scores = evaluate_with_checkpoints(
        rag_results, llm_eval, emb_eval,
        score_cache.with_name(f"ragas_samples_{version}.json"), score_fingerprint,
        limit=limit,
    )
    if scores is None:
        return None

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    score_cache.parent.mkdir(exist_ok=True)
    score_cache.write_text(json.dumps({"fingerprint": score_fingerprint, "scores": scores},
                                     allow_nan=False), encoding="utf-8")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main(eval_limit=None):
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    check_evaluator_quota()
    vectorstore = setup_vectorstore()

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1", limit=eval_limit)
    v2_scores = run_ragas_eval(v2_results, "v2", limit=eval_limit)
    if v1_scores is None or v2_scores is None:
        print("\nĐợt chấm đã lưu. Báo cáo chỉ được tạo khi đủ 50 câu cho cả V1/V2.")
        return "partial"

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "Hòa" if s1 == s2 else ("← V1" if s1 > s2 else "← V2")
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    report = {
        "sample_count_per_version": len(QA_PAIRS),
        "provider": config.PROVIDER,
        "generation_model": config.GEMINI_MODEL if config.PROVIDER == "gemini" else None,
        "evaluator": {"provider": config.EVAL_PROVIDER,
                      "gemini_model": config.GEMINI_EVAL_MODEL if config.EVAL_PROVIDER == "gemini" else None,
                      "reasoning_effort": ("minimal" if "flash-lite" in config.GEMINI_EVAL_MODEL or config.GEMINI_EVAL_MODEL.startswith("gemma-") else "low") if config.EVAL_PROVIDER == "gemini" and config.GEMINI_EVAL_MODEL.startswith(("gemini-3", "gemma-")) else None,
                      "separate_candidate_requests": True},
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    # Gợi ý: report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    evidence_path = report_path.parent.parent / "evidence" / "03_ragas_report.json"
    evidence_path.parent.mkdir(exist_ok=True)
    evidence_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
