import contextlib
import io
import unittest

from tts_summarizer import cli


class CliCommandTests(unittest.TestCase):
    def test_speak_command_is_shelved(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["speak", "--text", "hello"])

        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
