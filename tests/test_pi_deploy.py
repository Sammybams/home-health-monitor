from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PiDeployScriptTests(unittest.TestCase):
    def run_script(self, name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["sh", str(ROOT / "deploy" / name), *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_installer_help_gives_complete_three_command_flow(self) -> None:
        result = self.run_script("install-pi.sh", "--help")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("git clone", result.stdout)
        self.assertIn("install-pi.sh", result.stdout)
        self.assertIn("verify-pi.sh", result.stdout)

    def test_installer_refuses_a_checkout_outside_opt(self) -> None:
        result = self.run_script("install-pi.sh")

        self.assertNotEqual(0, result.returncode)
        self.assertIn("/opt/home-health-monitor", result.stderr)

    def test_verifier_help_explains_success_conditions(self) -> None:
        result = self.run_script("verify-pi.sh", "--help")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("model_loaded", result.stdout)
        self.assertIn("normal or anomaly", result.stdout)


if __name__ == "__main__":
    unittest.main()
