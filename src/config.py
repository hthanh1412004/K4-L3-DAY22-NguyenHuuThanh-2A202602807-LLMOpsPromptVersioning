"""
Tải cấu hình từ file .env và thiết lập biến môi trường LangSmith.

⚠️  Import module này TRƯỚC KHI import bất kỳ thư viện LangChain nào.
    config.py tự động set LANGCHAIN_* vào os.environ khi được import.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Tải .env từ thư mục gốc của project (Lab/)
_root = Path(__file__).parent.parent
load_dotenv(_root / ".env")

# ── LangSmith — PHẢI set trước khi import LangChain ──────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "day22-lab")
os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ── Provider mặc định ─────────────────────────────────────────────────────
# Đổi giá trị PROVIDER trong .env: openai | gemini | anthropic | ollama | openrouter
PROVIDER = os.getenv("PROVIDER", "openai").lower()
EVAL_PROVIDER = os.getenv("EVAL_PROVIDER", PROVIDER).lower()

# ── OpenAI ────────────────────────────────────────────────────────────────
OPENAI_API_KEY         = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL        = os.getenv("OPENAI_BASE_URL", "")   # để trống nếu dùng OpenAI chính thức
OPENAI_MODEL           = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ── Google Gemini ─────────────────────────────────────────────────────────
GOOGLE_API_KEY          = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL            = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_EVAL_MODEL       = os.getenv("GEMINI_EVAL_MODEL", "") or GEMINI_MODEL
GEMINI_EMBEDDING_MODEL  = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

# ── Anthropic ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# ── Ollama (local, không cần API key) ────────────────────────────────────
OLLAMA_BASE_URL         = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL            = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBEDDING_MODEL  = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ── OpenRouter ────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── LangSmith ─────────────────────────────────────────────────────────────
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "day22-lab")


def validate() -> bool:
    """
    Kiểm tra các biến môi trường bắt buộc đã được cấu hình.
    Trả về True nếu hợp lệ, False nếu thiếu.
    """
    missing = []
    def configured(value):
        return bool(value.strip()) and not value.strip().startswith("your_")
    supported = {"openai", "gemini", "anthropic", "ollama", "openrouter"}
    if EVAL_PROVIDER not in supported:
        missing.append("EVAL_PROVIDER hợp lệ (openai/gemini/anthropic/ollama/openrouter)")
    evaluator_keys = {"openai": OPENAI_API_KEY, "gemini": GOOGLE_API_KEY,
                      "anthropic": ANTHROPIC_API_KEY, "openrouter": OPENROUTER_API_KEY}
    if EVAL_PROVIDER in evaluator_keys:
        key = evaluator_keys[EVAL_PROVIDER]
        if not configured(key):
            missing.append(f"API key cho evaluator {EVAL_PROVIDER}")

    if PROVIDER not in {"openai", "gemini", "anthropic", "ollama", "openrouter"}:
        missing.append("PROVIDER hợp lệ (openai/gemini/anthropic/ollama/openrouter)")
    if os.environ["LANGCHAIN_TRACING_V2"].lower() != "true":
        missing.append("LANGCHAIN_TRACING_V2=true")

    if not configured(LANGSMITH_API_KEY):
        missing.append("LANGCHAIN_API_KEY (LangSmith)")

    if PROVIDER == "openai" and not configured(OPENAI_API_KEY):
        missing.append("OPENAI_API_KEY")
    elif PROVIDER == "gemini" and not configured(GOOGLE_API_KEY):
        missing.append("GOOGLE_API_KEY")
    elif PROVIDER == "anthropic" and not configured(ANTHROPIC_API_KEY):
        missing.append("ANTHROPIC_API_KEY")
    elif PROVIDER == "openrouter" and not configured(OPENROUTER_API_KEY):
        missing.append("OPENROUTER_API_KEY")
    if PROVIDER in {"anthropic", "openrouter"} and not configured(OPENAI_API_KEY):
        missing.append("OPENAI_API_KEY (embeddings)")
    # Ollama: không cần API key

    if missing:
        print("⚠️  Thiếu biến môi trường:")
        for m in missing:
            print(f"   - {m}")
        print("   Hãy kiểm tra file .env của bạn (xem .env.example để biết thêm).")
        return False

    print(f"✅ Config OK  |  Provider: {PROVIDER.upper()}  |  Project: {LANGSMITH_PROJECT}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
