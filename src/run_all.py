"""
Chạy tất cả các bước lab theo thứ tự hoặc chỉ một bước cụ thể.

Cách dùng:
    python run_all.py            # chạy tất cả 4 bước
    python run_all.py --step 1   # chỉ chạy Bước 1
    python run_all.py --step 3   # chỉ chạy Bước 3 (RAGAS ~15-30 phút)
"""
import sys
import argparse
import importlib
import contextlib
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


STEPS = {
    1: ("Bước 1: LangSmith RAG Pipeline",   "01_langsmith_rag_pipeline"),
    2: ("Bước 2: Prompt Hub & A/B Routing",  "02_prompt_hub_ab_routing"),
    3: ("Bước 3: RAGAS Evaluation",          "03_ragas_evaluation"),
    4: ("Bước 4: Guardrails AI Validators",  "04_guardrails_validator"),
}


class Tee:
    """Write the actual console output as UTF-8 evidence, including on Windows."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_step(step_num: int, eval_limit=None):
    if step_num == 3:
        from utils.evaluation_lock import evaluation_lock
        lock_path = Path(__file__).parent.parent / ".cache" / "ragas.lock"
        try:
            with evaluation_lock(lock_path):
                return _run_step(step_num, eval_limit=eval_limit)
        except RuntimeError as error:
            print(f"❌ {error}")
            return False
    return _run_step(step_num, eval_limit=eval_limit)


def _run_step(step_num: int, eval_limit=None):
    title, module_name = STEPS[step_num]
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    try:
        module = importlib.import_module(module_name)
        evidence = Path(__file__).parent.parent / "evidence"
        evidence.mkdir(exist_ok=True)
        filenames = {
            1: ["01_rag_pipeline_log.txt"],
            2: ["02_ab_routing_log.txt"],
            3: ["03_ragas_evaluation_log.txt"],
            4: ["04_pii_demo_log.txt", "04_json_demo_log.txt"],
        }
        with contextlib.ExitStack() as stack:
            logs = [stack.enter_context((evidence / name).open("w", encoding="utf-8"))
                    for name in filenames[step_num]]
            with contextlib.redirect_stdout(Tee(sys.stdout, *logs)):
                try:
                    result = module.main(eval_limit=eval_limit) if step_num == 3 else module.main()
                    if result == "partial":
                        print(f"\n⏸ {title} — ĐÃ LƯU TIẾN ĐỘ, CHƯA CHẤM ĐỦ")
                        return None
                except Exception as error:
                    details = traceback.format_exc()
                    import config
                    for secret in (config.GOOGLE_API_KEY, config.OPENAI_API_KEY,
                                   config.LANGSMITH_API_KEY, config.ANTHROPIC_API_KEY,
                                   config.OPENROUTER_API_KEY):
                        if secret:
                            details = details.replace(secret, "[REDACTED]")
                    print(f"\n❌ {title} — {type(error).__name__}")
                    print(details)
                    return False
        print(f"\n✅ {title} — HOÀN THÀNH")
        return True
    except SystemExit as e:
        if e.code != 0:
            print(f"\n❌ {title} — DỪNG (config thiếu hoặc lỗi)")
        return e.code == 0
    except Exception as e:
        print(f"\n❌ {title} — LỖI: {e}")
        return False


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Chạy Day22 Lab: LangSmith + Prompt Versioning + RAGAS + Guardrails"
    )
    parser.add_argument(
        "--eval-limit", type=int,
        help="Chấm tối đa N câu mới cho mỗi phiên bản rồi lưu và dừng (dùng với --step 3)"
    )
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3, 4],
        help="Chỉ chạy bước được chỉ định (1-4)"
    )
    args = parser.parse_args()
    if args.eval_limit is not None and (args.eval_limit < 1 or args.step != 3):
        parser.error("--eval-limit phải >= 1 và dùng cùng --step 3")

    steps_to_run = [args.step] if args.step else list(STEPS.keys())

    results = {}
    for step_num in steps_to_run:
        success = run_step(step_num, eval_limit=args.eval_limit)
        results[step_num] = success
        if not success and not args.step:
            print(f"\n⛔ Dừng lại do Bước {step_num} chưa hoàn tất.")
            break

    # Tổng kết
    print(f"\n{'=' * 60}")
    print("  Tổng kết")
    print(f"{'=' * 60}")
    for step_num, success in results.items():
        title = STEPS[step_num][0]
        status = "⏸ ĐÃ LƯU TIẾN ĐỘ" if success is None else ("✅ PASS" if success else "❌ FAIL")
        print(f"  {status}  {title}")
    if any(result is False for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
