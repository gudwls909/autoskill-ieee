from .file_utils import *
from .json_utils import *
from .record_utils import EventRecorder
import json, time


class Tee:
    """터미널 출력을 유지하면서 파일에도 동시 기록."""
    def __init__(self, original, file_obj):
        self.original = original
        self.file = file_obj

    def write(self, data):
        self.original.write(data)
        self.file.write(data)

    def flush(self):
        self.original.flush()
        self.file.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.original.fileno()


try:
    from langchain_openai import OpenAIEmbeddings as _OpenAIEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import OpenAIEmbeddings as _OpenAIEmbeddings
    except Exception:
        _OpenAIEmbeddings = None

def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)

class LoggedEmbeddings:
    """OpenAIEmbeddings wrapper that logs each embedding call to embedding_log.jsonl."""
    def __init__(self, agent_name: str, log_path: str, **kwargs):
        self._emb = _OpenAIEmbeddings(**kwargs)
        self.agent_name = agent_name
        self.log_path = log_path

    def embed_documents(self, texts):
        embs = self._emb.embed_documents(texts)
        record = {
            "ts": time.time(),
            "agent": self.agent_name,
            "model": getattr(self._emb, "model", None),
            "n_texts": len(texts),
            "approx_tokens_total": sum(_approx_tokens(t) for t in texts),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return embs

    def embed_query(self, text):
        emb = self._emb.embed_query(text)
        record = {
            "ts": time.time(),
            "agent": self.agent_name,
            "model": getattr(self._emb, "model", None),
            "n_texts": 1,
            "approx_tokens_total": _approx_tokens(text),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return emb

try:
    from langchain.callbacks.base import BaseCallbackHandler
except ModuleNotFoundError:
    try:
        from langchain_core.callbacks.base import BaseCallbackHandler
    except ModuleNotFoundError:
        class BaseCallbackHandler:
            pass

class TokenLogger(BaseCallbackHandler):
    def __init__(self, agent_name, log_path):
        self.agent_name = agent_name
        self.log_path = log_path

    def on_llm_end(self, response, **kwargs):
        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage", {}) or {}
        record = {
            "agent": self.agent_name,
            "model": llm_output.get("model_name"),
            "prompt": usage.get("prompt_tokens", 0),
            "completion": usage.get("completion_tokens", 0),
            "total": usage.get("total_tokens", 0),
            "ts": time.time(),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
