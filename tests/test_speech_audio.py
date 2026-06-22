import io
import unittest
import wave

import tts_summarizer.speech
from tts_summarizer.audio import chunks_to_wav_bytes
from tts_summarizer.config import TtsConfig
from tts_summarizer.speech import AudioChunk, SpeechGenerator


class FakeBackend:
    def generate(self, text, config):
        return [AudioChunk(samples=[0.0, 0.25, -0.25], sample_rate=8000)]


class SpeechAudioTests(unittest.TestCase):
    def test_speech_generator_passes_text(self):
        generator = SpeechGenerator(TtsConfig(sample_rate=8000), backend=FakeBackend())

        chunks = generator.generate("hello")

        self.assertEqual(
            chunks, [AudioChunk(samples=[0.0, 0.25, -0.25], sample_rate=8000)]
        )

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
        backend = tts_summarizer.speech.MlxAudioBackend()
        backend._model = Model()
        backend._model_name = config.model

        chunks = backend.generate("hello", config)

        self.assertEqual(
            calls, [{"text": "hello", "cfg_scale": 2.5, "steps": 30, "stream": False}]
        )
        self.assertEqual(chunks, [AudioChunk(samples=[0.1, -0.1], sample_rate=16000)])

    def test_mlx_backend_preserves_stream_override_in_generate_kwargs(self):
        calls = []

        class Model:
            def generate(self, **kwargs):
                calls.append(kwargs)
                return []

        config = TtsConfig(model="fake", stream=False, generate_kwargs={"stream": True})
        backend = tts_summarizer.speech.MlxAudioBackend()
        backend._model = Model()
        backend._model_name = config.model

        self.assertEqual(backend.generate("hello", config), [])
        self.assertEqual(calls, [{"text": "hello", "stream": True}])

    def test_chunks_to_wav_bytes_returns_readable_wav(self):
        body = chunks_to_wav_bytes(
            [AudioChunk(samples=[0.0, 0.5, -0.5], sample_rate=8000)]
        )

        self.assertTrue(body.startswith(b"RIFF"))
        with wave.open(io.BytesIO(body), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 8000)
            self.assertEqual(wav.getnframes(), 3)

    def test_chunks_to_wav_bytes_appends_multiple_chunks(self):
        body = chunks_to_wav_bytes(
            [
                AudioChunk(samples=[0.0], sample_rate=8000),
                AudioChunk(samples=[1.0], sample_rate=8000),
            ]
        )

        with wave.open(io.BytesIO(body), "rb") as wav:
            self.assertEqual(wav.getnframes(), 2)

    def test_chunks_to_wav_bytes_rejects_mixed_sample_rates(self):
        with self.assertRaises(ValueError):
            chunks_to_wav_bytes(
                [
                    AudioChunk(samples=[0.0], sample_rate=8000),
                    AudioChunk(samples=[0.0], sample_rate=16000),
                ]
            )


if __name__ == "__main__":
    unittest.main()
