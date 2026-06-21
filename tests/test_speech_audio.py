import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from tts_summarizer.audio import AudioPlayer
from tts_summarizer.config import AudioConfig, SessionConfig, TtsConfig
from tts_summarizer.request import SpeechRequest
from tts_summarizer.session import SessionManager
from tts_summarizer.speech import AudioChunk, SpeechGenerator


class FakeBackend:
    def generate(self, text, config):
        return [AudioChunk(samples=[0.0, 0.25, -0.25], sample_rate=8000)]


class SpeechAudioTests(unittest.TestCase):
    def test_speech_generator_passes_text(self):
        generator = SpeechGenerator(TtsConfig(sample_rate=8000), backend=FakeBackend())
        generator.generate("hello")

    def test_mlx_backend_forwards_generate_kwargs_and_stream(self):
        calls = []

        class Result:
            audio = [0.1, -0.1]
            sample_rate = 16000

        class Model:
            def generate(self, **kwargs):
                calls.append(kwargs)
                return [Result()]

        config = TtsConfig(
            model="fake",
            sample_rate=8000,
            stream=False,
            generate_kwargs={"cfg_scale": 2.5, "steps": 30},
        )

        from tts_summarizer.speech import MlxAudioBackend

        backend = MlxAudioBackend()
        backend._model = Model()
        backend._model_name = config.model

        chunks = backend.generate("hello", config)

        self.assertEqual(calls, [{"text": "hello", "cfg_scale": 2.5, "steps": 30, "stream": False}])
        self.assertEqual(
            chunks,
            [AudioChunk(samples=[0.1, -0.1], sample_rate=16000)],
        )


    def test_mlx_backend_preserves_stream_override_in_generate_kwargs(self):
        calls = []

        class Model:
            def generate(self, **kwargs):
                calls.append(kwargs)
                return []

        config = TtsConfig(
            model="fake",
            stream=False,
            generate_kwargs={"stream": True},
        )

        from tts_summarizer.speech import MlxAudioBackend

        backend = MlxAudioBackend()
        backend._model = Model()
        backend._model_name = config.model

        self.assertEqual(backend.generate("hello", config), [])
        self.assertEqual(calls, [{"text": "hello", "stream": True}])
    def test_audio_player_file_backend_writes_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            player = AudioPlayer(AudioConfig(backend="file", output_dir=tmp, save=True))
            player.play([AudioChunk(samples=[0.0, 0.5, -0.5], sample_rate=8000)])
            files = list(Path(tmp).glob("*.wav"))
        self.assertEqual(len(files), 1)

    def test_audio_player_auto_and_ffplay_backends_use_ffplay_command(self):
        calls = []

        class FinishedProcess:
            def poll(self):
                return 0

        def fake_popen(args):
            calls.append(args)
            return FinishedProcess()

        for backend in ("auto", "ffplay"):
            with self.subTest(backend=backend):
                calls.clear()
                with tempfile.TemporaryDirectory() as tmp:
                    player = AudioPlayer(
                        AudioConfig(backend=backend, output_dir=tmp, save=False)
                    )
                    with patch("tts_summarizer.audio.subprocess.Popen", fake_popen):
                        player.play([AudioChunk(samples=[0.0], sample_rate=8000)])

                    self.assertEqual(len(calls), 1)
                    wav_path = calls[0][-1]
                    self.assertEqual(
                        calls[0],
                        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", wav_path],
                    )
                    self.assertTrue(Path(wav_path).exists())


    def test_audio_player_terminates_ffplay_when_token_cancelled(self):
        events = []
        manager = SessionManager(SessionConfig(interrupt_same_session=True))
        token = manager.begin(SpeechRequest(text="old", caller="c", session_id="s"))

        class Proc:
            def __init__(self, command):
                self.polls = 0

            def poll(self):
                self.polls += 1
                if self.polls == 1:
                    manager.begin(SpeechRequest(text="new", caller="c", session_id="s"))
                return None

            def terminate(self):
                events.append("terminated")

        with tempfile.TemporaryDirectory() as tmp:
            player = AudioPlayer(AudioConfig(backend="ffplay", output_dir=tmp, save=False))
            with patch("tts_summarizer.audio.subprocess.Popen", Proc):
                player.play([AudioChunk(samples=[0.0], sample_rate=8000)], token=token)

        self.assertEqual(events, ["terminated"])

if __name__ == "__main__":
    unittest.main()
