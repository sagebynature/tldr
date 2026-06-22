from collections.abc import Iterable
import unittest
import json
from typing import cast
import tts_summarizer.speech
from tts_summarizer.audio import chunks_to_wav_stream
from tts_summarizer.config import TtsConfig, TtsProfileConfig
from tts_summarizer.speech import (
    AudioBytes,
    AudioChunk,
    RemoteTtsBackend,
    SpeechGenerator,
)


class FakeBackend:
    def generate(self, text, config):
        return [AudioChunk(samples=[0.0, 0.25, -0.25], sample_rate=8000)]


class SpeechAudioTests(unittest.TestCase):
    def test_remote_tts_backend_posts_openai_audio_speech_request(self):
        calls = []

        class Response:
            def __init__(self):
                self._chunks = [b"RIFFremote", b""]

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, _size=-1):
                return self._chunks.pop(0)

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            return Response()

        config = TtsProfileConfig(
            backend="remote",
            base_url="http://127.0.0.1:9100/v1/",
            api_key="omlx",
            model="mlx-community/Kokoro-82M-bf16",
            stream=True,
            generate_kwargs={"voice": "af_heart", "response_format": "mp3"},
        )

        output = RemoteTtsBackend(urlopen=fake_urlopen, timeout=7).generate(
            "hello", config
        )
        self.assertIsInstance(output, AudioBytes)
        self.assertEqual(b"".join(output.chunks), b"RIFFremote")

        request, timeout = calls[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "http://127.0.0.1:9100/v1/audio/speech")
        self.assertEqual(timeout, 7)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.headers["Content-type"], "application/json")
        self.assertEqual(request.headers["Authorization"], "Bearer omlx")
        self.assertEqual(body["model"], "mlx-community/Kokoro-82M-bf16")
        self.assertEqual(body["input"], "hello")
        self.assertIs(body["stream"], True)
        self.assertEqual(body["voice"], "af_heart")
        self.assertEqual(body["response_format"], "wav")

    def test_remote_tts_backend_raises_before_returning_audio_when_open_fails(self):
        class RemoteOpenError(Exception):
            pass

        calls = []

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            raise RemoteOpenError("remote down")

        config = TtsProfileConfig(
            backend="remote",
            base_url="http://127.0.0.1:9100/v1",
            model="model",
        )

        with self.assertRaises(RemoteOpenError):
            RemoteTtsBackend(urlopen=fake_urlopen, timeout=3).generate("hello", config)

        self.assertEqual(len(calls), 1)

    def test_remote_tts_backend_omits_empty_authorization(self):
        headers = []

        class Response:
            def __init__(self):
                self._chunks = [b"RIFF", b""]

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, _size=-1):
                return self._chunks.pop(0)

        def fake_urlopen(request, timeout):
            headers.append(request.headers)
            return Response()

        config = TtsProfileConfig(
            backend="remote",
            base_url="http://127.0.0.1:9100/v1",
            api_key="",
            model="model",
        )

        output = RemoteTtsBackend(urlopen=fake_urlopen).generate("hello", config)

        self.assertEqual(b"".join(output.chunks), b"RIFF")
        self.assertNotIn("Authorization", headers[0])

    def test_speech_generator_passes_text(self):
        generator = SpeechGenerator(
            TtsConfig(profiles={"qwen": TtsProfileConfig(sample_rate=8000)}),
            backend=FakeBackend(),
        )

        chunks = generator.generate("hello")

        self.assertEqual(
            chunks, [AudioChunk(samples=[0.0, 0.25, -0.25], sample_rate=8000)]
        )

    def test_speech_generator_selects_named_profile(self):
        calls = []

        class CapturingBackend:
            def generate(self, text, config):
                calls.append((text, config))
                return [AudioChunk(samples=[0.0], sample_rate=config.sample_rate)]

        generator = SpeechGenerator(
            TtsConfig(
                default_profile="qwen",
                profiles={
                    "qwen": TtsProfileConfig(model="qwen", sample_rate=24000),
                    "kokoro": TtsProfileConfig(
                        model="kokoro",
                        sample_rate=16000,
                        generate_kwargs={"speed": 1.6},
                    ),
                },
            ),
            backend=CapturingBackend(),
        )

        output = generator.generate("hello", profile_name="kokoro")
        chunks = list(cast(Iterable[AudioChunk], output))

        self.assertEqual(chunks[0].sample_rate, 16000)
        self.assertEqual(calls[0][1].model, "kokoro")
        self.assertEqual(calls[0][1].generate_kwargs["speed"], 1.6)

    def test_speech_generator_routes_remote_profile_to_remote_backend(self):
        calls = []

        class LocalBackend:
            def generate(self, text, config):
                calls.append(("local", text, config.model))
                return [AudioChunk(samples=[0.0], sample_rate=8000)]

        class RemoteBackend:
            def generate(self, text, config):
                calls.append(("remote", text, config.model))
                return AudioBytes([b"RIFFremote"])

        generator = SpeechGenerator(
            TtsConfig(
                default_profile="remote",
                profiles={
                    "local": TtsProfileConfig(model="local-model"),
                    "remote": TtsProfileConfig(
                        backend="remote",
                        base_url="http://127.0.0.1:9100/v1",
                        model="remote-model",
                    ),
                },
            )
        )
        generator.backend = tts_summarizer.speech.RoutingSpeechBackend(
            local=LocalBackend(), remote=RemoteBackend()
        )

        output = generator.generate("hello")

        self.assertEqual(output, AudioBytes([b"RIFFremote"]))
        self.assertEqual(calls, [("remote", "hello", "remote-model")])

    def test_speech_generator_rejects_unknown_backend(self):
        generator = SpeechGenerator(
            TtsConfig(
                profiles={"bad": TtsProfileConfig(backend="wat", model="m")},
                default_profile="bad",
            )
        )

        with self.assertRaisesRegex(ValueError, "unknown TTS backend: wat"):
            generator.generate("hello")

    def test_speech_generator_rejects_unknown_profile(self):
        generator = SpeechGenerator(TtsConfig(), backend=FakeBackend())

        with self.assertRaises(ValueError):
            generator.generate("hello", profile_name="missing")

    def test_mlx_backend_forwards_generate_kwargs_and_stream(self):
        calls = []

        class Result:
            audio = [0.1, -0.1]
            sample_rate = 16000

        class Model:
            def generate(self, **kwargs):
                calls.append(kwargs)
                return [Result()]

        config = TtsProfileConfig(
            model="fake",
            sample_rate=8000,
            stream=False,
            generate_kwargs={"cfg_scale": 2.5, "steps": 30},
        )
        backend = tts_summarizer.speech.MlxAudioBackend()
        backend._models[config.model] = Model()

        chunks = list(backend.generate("hello", config))

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

        config = TtsProfileConfig(
            model="fake", stream=False, generate_kwargs={"stream": True}
        )
        backend = tts_summarizer.speech.MlxAudioBackend()
        backend._models[config.model] = Model()

        self.assertEqual(list(backend.generate("hello", config)), [])
        self.assertEqual(calls, [{"text": "hello", "stream": True}])

    def test_mlx_backend_caches_loaded_models_by_name(self):
        loaded = []

        class Model:
            def generate(self, **kwargs):
                return []

        backend = tts_summarizer.speech.MlxAudioBackend()

        def load(model_name):
            loaded.append(model_name)
            return Model()

        backend._load_model = load

        list(backend.generate("hello", TtsProfileConfig(model="a")))
        list(backend.generate("hello", TtsProfileConfig(model="b")))
        list(backend.generate("hello again", TtsProfileConfig(model="a")))

        self.assertEqual(loaded, ["a", "b"])

    def test_kokoro_sinegen_patch_aligns_noise_length_mismatch(self):
        try:
            import mlx.core as mx
            from mlx_audio.tts.models.kokoro.istftnet import SineGen
        except ImportError:
            self.skipTest("mlx-audio Kokoro not installed")

        tts_summarizer.speech._patch_kokoro_sinegen()
        sine = SineGen(samp_rate=24000, upsample_scale=300, harmonic_num=8)
        setattr(sine, "_f02sine", lambda _fn: mx.ones((1, 4, 9)))

        sine_waves, uv, noise = sine(mx.ones((1, 3, 1)))
        mx.eval(sine_waves, uv, noise)

        self.assertEqual(sine_waves.shape, (1, 3, 9))
        self.assertEqual(uv.shape, (1, 3, 1))

    def test_chunks_to_wav_stream_yields_header_before_consuming_chunks(self):
        events = []

        def chunks():
            events.append("consumed")
            yield AudioChunk(samples=[0.0], sample_rate=8000)

        stream = chunks_to_wav_stream(chunks())

        self.assertTrue(next(stream).startswith(b"RIFF"))
        self.assertEqual(events, [])
        self.assertEqual(next(stream), b"\x00\x00")
        self.assertEqual(events, ["consumed"])


if __name__ == "__main__":
    unittest.main()
