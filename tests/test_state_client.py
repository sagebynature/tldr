import os
import tempfile
import unittest
from pathlib import Path

from echobrief.config import load_config
from echobrief.state import read_state, write_state


class StateClientTests(unittest.TestCase):
    def test_write_and_read_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.toml"
            cfg_path.write_text(
                f'[server]\nstate_dir = "{tmp}/state"\n', encoding="utf-8"
            )
            cfg = load_config(str(cfg_path), cwd=Path(tmp), home=Path(tmp))
            write_state(cfg, "127.0.0.1", 4321, os.getpid())
            state = read_state(cfg)
        assert state is not None
        self.assertEqual(state.host, "127.0.0.1")
        self.assertEqual(state.port, 4321)
        self.assertEqual(state.pid, os.getpid())

    def test_missing_state_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.toml"
            cfg_path.write_text(
                f'[server]\nstate_dir = "{tmp}/missing-state"\n', encoding="utf-8"
            )
            cfg = load_config(str(cfg_path), cwd=Path(tmp), home=Path(tmp))
            self.assertIsNone(read_state(cfg))


if __name__ == "__main__":
    unittest.main()
