"""Cache embeddings thật và retry quota theo phút, không tạo vector mô phỏng."""
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from langchain_core.embeddings import Embeddings

_LOCK = threading.RLock()
CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "embedding_cache.json"


class CachedEmbeddings(Embeddings):
    """Giữ riêng document/query vectors, phân batch và lưu sau mỗi batch đạt."""

    def __init__(self, backend, namespace, cache_path=CACHE_PATH, batch_size=40):
        self.backend = backend
        self.namespace = namespace
        self.cache_path = Path(cache_path)
        self.batch_size = batch_size

    def _key(self, task, text):
        return hashlib.sha256(f"{self.namespace}\0{task}\0{text}".encode("utf-8")).hexdigest()

    def _read(self):
        if not self.cache_path.exists():
            return {}
        return json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _save(self, cache):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(cache, allow_nan=False), encoding="utf-8")
        temporary.replace(self.cache_path)

    @staticmethod
    def _retry(call):
        for attempt in range(5):
            try:
                return call()
            except Exception as error:
                message = str(error)
                # Không chờ khi hết quota ngày hoặc model có quota bằng 0.
                rate_limited = "429" in message or "RESOURCE_EXHAUSTED" in message
                daily = "PerDay" in message or "requests_per_day" in message
                unavailable = re.search(r"limit:\s*0(?:\D|$)", message)
                if not rate_limited or daily or unavailable or attempt == 4:
                    raise
                hint = re.search(r"retry in ([\d.]+)s", message, re.IGNORECASE)
                delay = min(60, float(hint.group(1)) + 2) if hint else 60
                print(f"⏳ Gemini quota theo phút: chờ {delay:.1f}s rồi thử lại.", flush=True)
                time.sleep(delay)

    def embed_documents(self, texts):
        with _LOCK:
            cache = self._read()
            missing = list(dict.fromkeys(text for text in texts if self._key("document", text) not in cache))
            for offset in range(0, len(missing), self.batch_size):
                batch = missing[offset:offset + self.batch_size]
                vectors = self._retry(lambda: self.backend.embed_documents(batch))
                if len(vectors) != len(batch):
                    raise ValueError("Embedding API trả thiếu vector")
                cache.update({self._key("document", text): vector for text, vector in zip(batch, vectors)})
                self._save(cache)
                print(f"📦 Embeddings: {min(offset + len(batch), len(missing))}/{len(missing)} đoạn mới.", flush=True)
            return [cache[self._key("document", text)] for text in texts]

    def embed_query(self, text):
        with _LOCK:
            cache = self._read()
            key = self._key("query", text)
            if key not in cache:
                cache[key] = self._retry(lambda: self.backend.embed_query(text))
                self._save(cache)
            return cache[key]
