import unittest

from chaptering.inference import NoRedirect, build_prompt, completion_url, final_content, strip_qwen_terminal_tokens


class InferenceTests(unittest.TestCase):
    def test_qwen_hf_dual_termination_framing(self):
        text = '{"chapters":[{"start":0,"title":"Bread"}]}'
        self.assertEqual(strip_qwen_terminal_tokens(text + '<|im_end|>\n<|endoftext|>'), text)

    def test_qwen_framing_does_not_repair_json_or_strip_title_content(self):
        text = '{"chapters":[{"start":0,"title":"Literal <|im_end|>"}]}'
        self.assertEqual(strip_qwen_terminal_tokens(text), text)
        self.assertEqual(strip_qwen_terminal_tokens('```json\n{}\n```<|im_end|>'), '```json\n{}\n```')

    def test_authenticated_endpoint_requires_https_and_no_embedded_credentials(self):
        self.assertEqual(completion_url("https://example.com/v1/"),
                         "https://example.com/v1/chat/completions")
        for url in ["http://example.com/v1", "https://key@example.com/v1", "https://example.com/v1?key=x"]:
            with self.assertRaises(ValueError):
                completion_url(url)

    def test_authenticated_requests_never_follow_redirects(self):
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, "Found", {},
                                                       "https://other.example/v1"))

    def test_prompt_does_not_expose_reference_or_metadata(self):
        row = {"video_id": "a", "sentences": ["One."],
               "reference": {"chapter_titles": ["SECRET_GOLD"], "chapter_starts": [0]},
               "chapter_count": "SECRET_COUNT"}
        self.assertNotIn("SECRET", build_prompt(row))
        self.assertIn("[0] One.", build_prompt(row))

    def test_thinking_is_separated_before_final_json(self):
        text, info = final_content({"content": '<think>{"bad":"reasoning"}</think> {"chapters":[]}'})
        self.assertEqual(text, '{"chapters":[]}')
        self.assertEqual(info["reasoning_source"], "explicit_think_block")

    def test_api_reasoning_is_not_used_as_final(self):
        with self.assertRaises(ValueError):
            final_content({"content": None, "reasoning_content": '{"chapters":[]}'})

    def test_unclosed_thinking_is_failure(self):
        with self.assertRaises(ValueError):
            final_content({"content": '<think>{"chapters":[]}'})


if __name__ == "__main__":
    unittest.main()
