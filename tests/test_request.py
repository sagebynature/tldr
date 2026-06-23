import unittest

from tldr.request import RequestError, SpeechRequest


class RequestTests(unittest.TestCase):
    def test_speech_request_uses_headers_for_identity(self):
        request = SpeechRequest.from_json(
            {"text": "hello", "caller": "body", "session_id": "body-session"},
            caller="header",
            session_id="header-session",
        )

        self.assertEqual(request.caller, "header")
        self.assertEqual(request.session_id, "header-session")
        self.assertEqual(request.session_key(), "header:header-session")

    def test_speech_request_defaults_to_summarize_true(self):
        request = SpeechRequest.from_json({"text": "hello"}, caller="c", session_id="s")

        self.assertIs(request.summarize, True)

    def test_speech_request_accepts_summarize_false(self):
        request = SpeechRequest.from_json(
            {"text": "hello", "summarize": False}, caller="c", session_id="s"
        )

        self.assertIs(request.summarize, False)

    def test_speech_request_accepts_tts_profile(self):
        request = SpeechRequest.from_json(
            {"text": "hello", "tts_profile": "kokoro"}, caller="c", session_id="s"
        )

        self.assertEqual(request.tts_profile, "kokoro")

    def test_speech_request_accepts_summarizer_profile(self):
        request = SpeechRequest.from_json(
            {"text": "hello", "summarizer_profile": "fast"}, caller="c", session_id="s"
        )

        self.assertEqual(request.summarizer_profile, "fast")

    def test_missing_text_fails(self):
        with self.assertRaises(RequestError):
            SpeechRequest.from_json({"metadata": {"x": 1}})


if __name__ == "__main__":
    unittest.main()
