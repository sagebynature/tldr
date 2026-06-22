from pathlib import Path
import tomllib
import unittest


class InstallationArtifactTests(unittest.TestCase):
    def test_docker_compose_uses_container_network_config(self):
        compose = Path("docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("${TTS_SUMMARIZER_CONFIG:-./config.docker.example.toml}", compose)
        self.assertIn("/config/config.toml:ro", compose)
        self.assertIn("host.docker.internal:host-gateway", compose)
        self.assertIn("9200:9200", compose)

    def test_dockerfile_installs_with_uv(self):
        dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

        self.assertIn("COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/", dockerfile)
        self.assertIn("RUN uv sync --locked --no-dev", dockerfile)
        self.assertNotIn("pip install", dockerfile)

    def test_docker_runtime_uses_module_entrypoint(self):
        dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
        compose = Path("docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn('PYTHONPATH="/app/src"', dockerfile)
        self.assertIn('CMD ["python", "-m", "tts_summarizer", "serve", "--config", "/config/config.toml"]', dockerfile)
        self.assertIn('command: ["python", "-m", "tts_summarizer", "serve", "--config", "/config/config.toml"]', compose)

    def test_docker_example_config_reaches_host_models_and_publishes_daemon(self):
        config = tomllib.loads(Path("config.docker.example.toml").read_text(encoding="utf-8"))

        self.assertEqual(config["server"]["host"], "0.0.0.0")
        self.assertEqual(config["server"]["port"], 9200)
        self.assertEqual(
            config["summarizer"]["profiles"]["remote-qwen25"]["base_url"],
            "http://host.docker.internal:9000/v1",
        )
        self.assertEqual(
            config["tts"]["profiles"]["remote-kokoro"]["base_url"],
            "http://host.docker.internal:9000/v1",
        )


if __name__ == "__main__":
    unittest.main()
