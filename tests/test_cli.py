import unittest
from tldr.cli import main


class CliTests(unittest.TestCase):
    def test_config_check_command_exists(self):
        self.assertEqual(main(["config-check"]), 0)

    def test_unknown_command_fails(self):
        self.assertNotEqual(main(["not-a-command"]), 0)


if __name__ == "__main__":
    unittest.main()
