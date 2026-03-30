from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NEW_SCRIPT = REPO_ROOT / "scripts" / "instinct_rl" / "play_depth.py"
OLD_SCRIPT = (
    REPO_ROOT
    / "source"
    / "instinctlab"
    / "instinctlab"
    / "tasks"
    / "parkour"
    / "scripts"
    / "play.py"
)


def test_parkour_depth_play_entrypoint_moved_to_scripts_directory():
    assert NEW_SCRIPT.is_file()
    assert not OLD_SCRIPT.exists()
