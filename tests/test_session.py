import unittest

from tts_summarizer.config import Config
from tts_summarizer.request import SpeechRequest
from tts_summarizer.session import SessionManager


class SessionTests(unittest.TestCase):
    def test_same_session_interrupts_previous_token(self):
        manager = SessionManager(Config().session)
        first = manager.begin(SpeechRequest(text="one", caller="c", session_id="s"))
        second = manager.begin(SpeechRequest(text="two", caller="c", session_id="s"))
        self.assertTrue(first.cancelled())
        self.assertFalse(second.cancelled())

    def test_different_session_does_not_interrupt(self):
        manager = SessionManager(Config().session)
        first = manager.begin(SpeechRequest(text="one", caller="c", session_id="s1"))
        second = manager.begin(SpeechRequest(text="two", caller="c", session_id="s2"))
        self.assertFalse(first.cancelled())
        self.assertFalse(second.cancelled())


if __name__ == "__main__":
    unittest.main()
