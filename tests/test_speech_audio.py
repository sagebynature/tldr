import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from tts_summarizer.audio import AudioPlayer
from tts_summarizer.config import AudioConfig, TtsConfig
from tts_summarizer.speech import AudioChunk, SpeechGenerator


class FakeBackend:
    def generate(self, text, config):
        return [AudioChunk(samples=[0.0, 0.25, -0.25], sample_rate=8000)]


class SpeechAudioTests(unittest.TestCase):
    def test_speech_generator_passes_text(self):
        generator = SpeechGenerator(TtsConfig(sample_rate=8000), backend=FakeBackend())
        generator.generate("hello")

    def test_mlx_backend_uses_configured_generate_kwargs(self):
        calls = []

        class Model:
            pass

        model = Model()
        config = TtsConfig(
            model="fake",
            sample_rate=8000,
            generate_kwargs={"cfg_scale": 2.5, "steps": 30},
        )

        def fake_generate_audio(*args, **kwargs):
            calls.append((args, kwargs))

        from tts_summarizer.speech import MlxAudioBackend

        backend = MlxAudioBackend()
        backend._model = model
        backend._model_name = config.model

        with patch("tts_summarizer.speech.generate_audio", fake_generate_audio):
            backend.generate("hello", config)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], ("hello",))
        self.assertEqual(
            calls[0][1],
            {
                "model": model,
                "stream": True,
                "play": True,
                "cfg_scale": 2.5,
                "steps": 30,
            },
        )

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


if __name__ == "__main__":
    unittest.main()
