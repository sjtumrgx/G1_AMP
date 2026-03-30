from pathlib import Path
import subprocess
import sys


PLAY_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "instinct_rl" / "play_mujoco.py"


def test_play_mujoco_headless_zero_depth_smoke():
    result = subprocess.run(
        [
            sys.executable,
            str(PLAY_SCRIPT_PATH),
            "--load_run",
            "20260327_163647",
            "--headless",
            "--depth-mode",
            "zeros",
            "--sim_duration",
            "0.05",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Completed MuJoCo sim2sim smoke run" in result.stdout
