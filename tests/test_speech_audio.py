import tempfile
import unittest
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
        class Result:
            audio = [0.0]
            sample_rate = 8000

        class Model:
            sample_rate = 8000

            def __init__(self):
                self.kwargs = {}

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return [Result()]

        class Backend:
            def __init__(self, model):
                self.model = model

            def generate(self, text, config):
                from tts_summarizer.speech import MlxAudioBackend

                backend = MlxAudioBackend()
                backend._model = self.model
                backend._model_name = config.model
                return backend.generate(text, config)

        model = Model()
        config = TtsConfig(
            model="fake",
            sample_rate=8000,
            generate_kwargs={"cfg_scale": 2.5, "steps": 30},
        )
        chunks = Backend(model).generate("hello", config)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(model.kwargs, {"text": "hello", "cfg_scale": 2.5, "steps": 30})

    def test_audio_player_file_backend_writes_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            player = AudioPlayer(AudioConfig(backend="file", output_dir=tmp, save=True))
            player.play([AudioChunk(samples=[0.0, 0.5, -0.5], sample_rate=8000)])
            files = list(Path(tmp).glob("*.wav"))
        self.assertEqual(len(files), 1)


if __name__ == "__main__":
    unittest.main()
