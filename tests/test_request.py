import json
import unittest

from tts_summarizer.request import RequestError, SpeechRequest


class RequestTests(unittest.TestCase):
    def test_json_request_keeps_session_identity(self):
        req = SpeechRequest.from_json(
            {"text": "hello", "caller": "manual", "session_id": "abc"}
        )
        self.assertEqual(req.text, "hello")
        self.assertEqual(req.session_key(), "manual:abc")

    def test_missing_text_fails(self):
        with self.assertRaises(RequestError):
            SpeechRequest.from_json({"caller": "manual"})

    def test_cli_text_wins_over_stdin(self):
        req = SpeechRequest.from_cli("from arg", '{"text":"from stdin"}', "cli", "s1")
        self.assertEqual(req.text, "from arg")
        self.assertEqual(req.session_key(), "cli:s1")

    def test_stdin_json_supported(self):
        payload = json.dumps(
            {"text": "from stdin", "caller": "hook", "session_id": "s2"}
        )
        req = SpeechRequest.from_cli(None, payload, None, None)
        self.assertEqual(req.text, "from stdin")
        self.assertEqual(req.session_key(), "hook:s2")


if __name__ == "__main__":
    unittest.main()
