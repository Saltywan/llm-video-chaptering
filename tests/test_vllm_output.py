"""Native-channel and failure accounting for the batched inference driver."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location("vllm_pair", Path(__file__).resolve().parents[1] / "scripts/infer_vllm_pair.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))

    def parse_response(self, raw, prefix):
        self.prefix = prefix
        return {"content": raw.removesuffix("<turn|>"), "thinking": None}


class NativeOutputTests(unittest.TestCase):
    def result(self, text, alias="M1", finish="stop"):
        tokenizer = Tokenizer()
        completion = SimpleNamespace(token_ids=tokenizer.encode(text), finish_reason=finish, stop_reason=None)
        request = SimpleNamespace(outputs=[completion], prompt_token_ids=[1, 2, 3], metrics=None)
        window = {"schema_version": 1, "video_id": "fixture", "window_index": 0,
                  "sentence_count": 2, "L": 0, "R": 2, "A": 0, "B": 2,
                  "sentences": ["First.", "Second."]}
        return driver.decode_result(tokenizer, alias, window, "native prompt", request, "revision"), tokenizer

    def test_native_qwen_final_and_exact_token_cost(self):
        raw = 'reasoning</think>\n\n{"chapters":[{"start":0,"title":"Topic"}]}<|im_end|>'
        result, _ = self.result(raw)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["usage"]["completion_tokens"], len(raw))
        self.assertEqual(result["usage"]["reasoning_tokens"], len("reasoning"))
        self.assertIsNone(result["wall_seconds"])
        self.assertIsNone(result["reasoning_budget"])

    def test_continued_reasoning_after_close_is_not_repaired(self):
        raw = 'thought</think>more thought</think>{"chapters":[{"start":0,"title":"Topic"}]}'
        result, _ = self.result(raw)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["raw_text"], raw)
        self.assertTrue(result["final_text"].startswith("more thought"))

    def test_truncation_retains_raw_output_and_cost(self):
        result, _ = self.result('unfinished reasoning', finish="length")
        self.assertEqual(result["status"], "truncated")
        self.assertEqual(result["usage"]["completion_tokens"], len('unfinished reasoning'))
        self.assertEqual(result["raw_text"], 'unfinished reasoning')
        self.assertIsNone(result["usage"]["reasoning_tokens"])

    def test_gemma_uses_official_parser_with_original_prefix(self):
        result, tokenizer = self.result('{"chapters":[{"start":0,"title":"Topic"}]}<turn|>', alias="M2")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(tokenizer.prefix, "native prompt")
        self.assertEqual(result["usage"]["reasoning_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
