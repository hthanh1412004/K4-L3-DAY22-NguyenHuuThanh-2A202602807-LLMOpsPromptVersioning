"""Retry malformed evaluator JSON without altering generated answers or scores."""
from langchain_core.utils.json import parse_json_markdown
from ragas.llms.base import LangchainLLMWrapper
from langchain_core.prompt_values import StringPromptValue
from langchain_core.outputs import LLMResult
import asyncio
import threading
import time
import re


class JSONRetryLLM(LangchainLLMWrapper):
    """RAGAS prompts request JSON; regenerate if the provider returns invalid JSON."""
    def __init__(self, *args, tokens_per_minute=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens_per_minute = tokens_per_minute
        self._pacing_lock = threading.Lock()
        self._next_request = 0.0

    def _reserve_delay(self, args, kwargs):
        if not self.tokens_per_minute:
            return 0.0
        prompt = args[0] if args else kwargs["prompt"]
        n = kwargs.get("n", args[1] if len(args) > 1 else 1)
        # Conservative input estimate for Vietnamese + JSON-heavy RAGAS prompts.
        estimated_tokens = len(prompt.to_string()) / 2
        interval = max(3.0, estimated_tokens * 60 / self.tokens_per_minute) * n
        with self._pacing_lock:
            now = time.monotonic()
            start = max(now, self._next_request)
            self._next_request = start + interval
            return start - now

    @staticmethod
    def _transient_server_error(error):
        current = error
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            code = getattr(current, "code", None) or getattr(current, "status_code", None)
            if code in (500, 502, 503, 504) or re.match(r"^(500|502|503|504)\b", str(current)):
                return True
            current = current.__cause__
        return False

    @staticmethod
    def _check(result):
        for group in result.generations:
            for generation in group:
                parse_json_markdown(generation.text)
        return result

    def generate_text(self, *args, **kwargs):
        n = kwargs.get("n", args[1] if len(args) > 1 else 1)
        if self.tokens_per_minute and n > 1:
            args, kwargs = self._single_candidate(args, kwargs)
            results = [self.generate_text(*args, **kwargs) for _ in range(n)]
            return LLMResult(generations=[[r.generations[0][0] for r in results]])
        args, kwargs = self._json_prompt(args, kwargs)
        for attempt in range(4):
            time.sleep(self._reserve_delay(args, kwargs))
            try:
                result = super().generate_text(*args, **kwargs)
            except Exception as error:
                if attempt == 3 or not self._transient_server_error(error):
                    raise
                delay = 5 * 2 ** attempt
                print(f"Máy chủ evaluator lỗi tạm thời; chờ {delay} giây rồi thử lại.", flush=True)
                time.sleep(delay)
                continue
            try:
                return self._check(result)
            except ValueError:
                if attempt == 3:
                    raise
                print("Evaluator trả JSON lỗi; gọi lại cùng prompt.")

    async def agenerate_text(self, *args, **kwargs):
        n = kwargs.get("n", args[1] if len(args) > 1 else 1)
        if self.tokens_per_minute and n > 1:
            args, kwargs = self._single_candidate(args, kwargs)
            results = []
            for _ in range(n):
                results.append(await self.agenerate_text(*args, **kwargs))
            return LLMResult(generations=[[r.generations[0][0] for r in results]])
        args, kwargs = self._json_prompt(args, kwargs)
        for attempt in range(4):
            await asyncio.sleep(self._reserve_delay(args, kwargs))
            try:
                result = await super().agenerate_text(*args, **kwargs)
            except Exception as error:
                if attempt == 3 or not self._transient_server_error(error):
                    raise
                delay = 5 * 2 ** attempt
                print(f"Máy chủ evaluator lỗi tạm thời; chờ {delay} giây rồi thử lại.", flush=True)
                await asyncio.sleep(delay)
                continue
            try:
                return self._check(result)
            except ValueError:
                if attempt == 3:
                    raise
                print("Evaluator trả JSON lỗi; gọi lại cùng prompt.")

    @staticmethod
    def _single_candidate(args, kwargs):
        kwargs = dict(kwargs)
        if len(args) > 1:
            args = (args[0], 1, *args[2:])
            kwargs.pop("n", None)
        else:
            kwargs["n"] = 1
        return args, kwargs

    @staticmethod
    def _json_prompt(args, kwargs):
        suffix = "\nWhen serializing JSON, use literal Unicode characters, never Unicode escape sequences."
        if args:
            prompt = args[0]
            value = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
            args = (StringPromptValue(text=value + suffix), *args[1:])
        else:
            kwargs = dict(kwargs)
            kwargs["prompt"] = StringPromptValue(text=kwargs["prompt"].to_string() + suffix)
        return args, kwargs
