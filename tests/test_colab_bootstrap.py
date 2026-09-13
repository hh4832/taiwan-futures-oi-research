import json
from pathlib import Path

from src.config import CONFIG, DRIVE_ROOT_FOLDER_ID


EXPECTED_DRIVE_ROOT = "1JrDCBf__DZA5aIlp3jxUqATsuQohtpLG"
NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "01_run_research.ipynb"


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_drive_folder_id_defined():
    assert DRIVE_ROOT_FOLDER_ID
    assert CONFIG.drive_folder_id == DRIVE_ROOT_FOLDER_ID


def test_drive_folder_id_matches_expected_root():
    assert DRIVE_ROOT_FOLDER_ID == EXPECTED_DRIVE_ROOT


def test_upload_uses_configured_parent_folder():
    upload_cell = next(cell for cell in _code_cells() if "upload_archive_to_drive" in cell)
    assert "from src.config import DRIVE_ROOT_FOLDER_ID" in upload_cell
    assert "upload_archive_to_drive(archive, DRIVE_ROOT_FOLDER_ID)" in upload_cell
    assert 'upload_result["parent_folder_id"] == DRIVE_ROOT_FOLDER_ID' in upload_cell


def test_public_clone_does_not_use_token_or_authenticated_url():
    clone_cell = next(cell for cell in _code_cells() if '"git", "clone"' in cell)
    assert "GITHUB_TOKEN" not in clone_cell
    assert "x-access-token" not in clone_cell
    assert "userdata" not in clone_cell
    assert "REPO_URL" in clone_cell


def test_finlab_secret_cell_is_self_contained_and_does_not_print_token():
    secret_cell = next(cell for cell in _code_cells() if "FINLAB_API_TOKEN" in cell)
    assert "import os" in secret_cell
    assert "from google.colab import userdata" in secret_cell
    assert 'userdata.get("FINLAB_API_TOKEN")' in secret_cell
    assert "FINLAB_API_TOKEN is missing from Colab Secrets" in secret_cell
    assert "print(finlab_token)" not in secret_cell
