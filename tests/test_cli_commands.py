import unittest
from unittest.mock import patch

from tts_summarizer import cli


class CliCommandTests(unittest.TestCase):
    def test_speak_posts_normalized_request(self):
        calls = []

        def fake_base_url(config, config_path):
            return "http://127.0.0.1:9999"

        def fake_post(url, payload, timeout):
            calls.append((url, payload, timeout))
            return {"status": "accepted"}

        with patch("tts_summarizer.cli.daemon_base_url", fake_base_url), patch(
            "tts_summarizer.cli.post_json", fake_post
        ):
            code = cli.main(["speak", "--caller", "manual", "--session-id", "s", "--text", "hello"])
        self.assertEqual(code, 0)
        self.assertEqual(calls[0][0], "http://127.0.0.1:9999/v1/speak")
        self.assertEqual(calls[0][1]["text"], "hello")
        self.assertEqual(calls[0][1]["session_id"], "s")

    def test_speak_no_daemon_is_best_effort_success(self):
        with patch("tts_summarizer.cli.daemon_base_url", lambda config, config_path: None):
            self.assertEqual(cli.main(["speak", "--text", "hello"]), 0)


if __name__ == "__main__":
    unittest.main()
