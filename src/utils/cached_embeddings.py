"""Persistent embedding cache with conservative Gemini free-tier pacing."""
import hashlib
import json
from pathlib import Path
import threading
import time
from langchain_core.embeddings import Embeddings


class CachedEmbeddings(Embeddings):
    """Cache by model/text, serialize concurrent calls and retry quota errors."""
    def __init__(self, delegate, model: str):
        self.delegate = delegate
        self.lock = threading.Lock()
        digest = hashlib.sha256(model.encode()).hexdigest()[:16]
        self.path = Path(__file__).resolve().parents[2] / ".cache" / f"embeddings-{digest}.json"
        self.path.parent.mkdir(exist_ok=True)
        self.cache = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.next_request = 0.0

    def _embed(self, texts, is_query=False):
        with self.lock:
            keys = [hashlib.sha256((("query:" if is_query else "") + text).encode("utf-8")).hexdigest()
                    for text in texts]
            missing = list(dict.fromkeys((key, text) for key, text in zip(keys, texts) if key not in self.cache))
            for offset in range(0, len(missing), 20):
                batch = missing[offset:offset + 20]
                for attempt in range(5):
                    time.sleep(max(0, self.next_request - time.monotonic()))
                    # Budget below 100 requests/minute, including parallel RAGAS threads.
                    self.next_request = time.monotonic() + len(batch) * 0.75
                    try:
                        vectors = ([self.delegate.embed_query(batch[0][1])] if is_query else
                                   self.delegate.embed_documents([text for _, text in batch]))
                        break
                    except Exception as exc:
                        if attempt == 4 or not any(s in str(exc) for s in ["429", "RESOURCE_EXHAUSTED"]):
                            raise
                        print("Embedding quota: chờ 30 giây rồi thử lại.")
                        self.next_request = time.monotonic() + 30
                if len(vectors) != len(batch):
                    raise ValueError("Embedding provider returned an incomplete batch")
                self.cache.update({key: vector for (key, _), vector in zip(batch, vectors)})
                temporary = self.path.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.cache), encoding="utf-8")
                temporary.replace(self.path)
            return [self.cache[key] for key in keys]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], is_query=True)[0]
