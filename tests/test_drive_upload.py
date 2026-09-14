import re
import sys
import types
from pathlib import Path

import pytest

from src import drive_upload


class Request:
    def __init__(self, result):
        self.result = result

    def execute(self):
        return self.result


class FilesApi:
    def __init__(self, run_name=None):
        self.items = {}
        self.created_folders = 0
        if run_name:
            self.add_folder("run-id", run_name, "parent")

    def add_folder(self, file_id, name, parent):
        self.items[file_id] = {
            "id": file_id,
            "name": name,
            "mimeType": drive_upload.DRIVE_FOLDER_MIME_TYPE,
            "parents": [parent],
        }

    def add_file(self, file_id, name, parent, size):
        self.items[file_id] = {
            "id": file_id,
            "name": name,
            "mimeType": "text/plain",
            "parents": [parent],
            "size": str(size),
        }

    def get(self, fileId, fields):
        return Request({
            "id": fileId,
            "mimeType": drive_upload.DRIVE_FOLDER_MIME_TYPE,
            "capabilities": {"canAddChildren": True},
        })

    def list(self, q, fields, pageSize, pageToken=None):
        parent = re.search(r"'([^']+)' in parents", q).group(1)
        name = re.search(r"name = '([^']+)'", q)
        items = [item for item in self.items.values() if parent in item["parents"]]
        if name:
            items = [item for item in items if item["name"] == name.group(1)]
        return Request({"files": items})

    def create(self, body, fields, media_body=None):
        assert media_body is None
        self.created_folders += 1
        file_id = f"folder-{self.created_folders}"
        self.add_folder(file_id, body["name"], body["parents"][0])
        return Request(self.items[file_id])


class Service:
    def __init__(self, files_api):
        self.files_api = files_api

    def files(self):
        return self.files_api


class HttpError(Exception):
    def __init__(self, status):
        self.resp = types.SimpleNamespace(status=status)
        super().__init__(f"HTTP {status}")


def archive_at(tmp_path: Path):
    archive = tmp_path / "archive"
    archive.mkdir()
    for name in ("run_info.txt", "primary_results.csv", "research_summary.md"):
        (archive / name).write_text(name, encoding="utf-8")
    return archive


def mock_upload(files_api, uploaded=None):
    def upload(service, path, parent_id):
        if uploaded is not None:
            uploaded.append(path.name)
        file_id = f"file-{len(files_api.items)}"
        files_api.add_file(file_id, path.name, parent_id, path.stat().st_size)
    return upload


def test_drive_http_transport_has_timeout(monkeypatch):
    captured = {}
    httplib2 = types.ModuleType("httplib2")
    httplib2.Http = lambda timeout: captured.setdefault("timeout", timeout)
    auth_http = types.ModuleType("google_auth_httplib2")
    auth_http.AuthorizedHttp = lambda credentials, http: (credentials, http)
    discovery = types.ModuleType("googleapiclient.discovery")
    discovery.build = lambda *args, **kwargs: captured.setdefault("client", kwargs)
    colab = types.ModuleType("google.colab")
    colab.auth = types.SimpleNamespace(authenticate_user=lambda: None)
    google = types.ModuleType("google")
    google.__path__ = []
    google.auth = types.SimpleNamespace(default=lambda: ("credentials", None))
    monkeypatch.setitem(sys.modules, "httplib2", httplib2)
    monkeypatch.setitem(sys.modules, "google_auth_httplib2", auth_http)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.auth", google.auth)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "googleapiclient.discovery", discovery)
    assert drive_upload._build_drive_service()
    assert captured["timeout"] == drive_upload.DRIVE_HTTP_TIMEOUT_SECONDS == 60


def test_drive_retry_is_finite():
    calls, sleeps = [], []
    def timeout():
        calls.append(1)
        raise TimeoutError("slow")
    with pytest.raises(drive_upload.DriveUploadError, match="attempt=4"):
        drive_upload._with_retry(
            timeout, stage="upload", file_name="data.csv", sleep=sleeps.append
        )
    assert len(calls) == 4
    assert sleeps == [2, 4, 8]


def test_drive_transient_error_retries():
    calls, sleeps = [], []
    def transient_then_ok():
        calls.append(1)
        if len(calls) < 3:
            raise HttpError(503)
        return "ok"
    assert drive_upload._with_retry(
        transient_then_ok, stage="list", sleep=sleeps.append
    ) == "ok"
    assert len(calls) == 3
    assert sleeps == [2, 4]


def test_drive_permanent_error_does_not_retry():
    calls = []
    def forbidden():
        calls.append(1)
        raise HttpError(403)
    with pytest.raises(drive_upload.DriveUploadError, match="attempt=1"):
        drive_upload._with_retry(
            forbidden, stage="auth", sleep=lambda _: pytest.fail("no retry")
        )
    assert len(calls) == 1


def test_existing_run_folder_is_not_duplicated(tmp_path, monkeypatch):
    archive = archive_at(tmp_path)
    api = FilesApi(run_name=archive.name)
    monkeypatch.setattr(drive_upload, "_upload_resumable_file", mock_upload(api))
    result = drive_upload.upload_archive_to_drive(
        archive, "parent", service=Service(api)
    )
    assert api.created_folders == 0
    assert result["remote_folder_id"] == "run-id"


def test_existing_identical_file_is_skipped(tmp_path, monkeypatch):
    archive = archive_at(tmp_path)
    api = FilesApi(run_name=archive.name)
    existing = archive / "primary_results.csv"
    api.add_file("existing", existing.name, "run-id", existing.stat().st_size)
    uploaded = []
    monkeypatch.setattr(
        drive_upload, "_upload_resumable_file", mock_upload(api, uploaded)
    )
    result = drive_upload.upload_archive_to_drive(
        archive, "parent", service=Service(api)
    )
    assert "primary_results.csv" in result["skipped_files"]
    assert "primary_results.csv" not in uploaded


def test_upload_returns_structured_result(tmp_path, monkeypatch):
    archive = archive_at(tmp_path)
    api = FilesApi()
    monkeypatch.setattr(drive_upload, "_upload_resumable_file", mock_upload(api))
    result = drive_upload.upload_archive_to_drive(
        archive, "parent", service=Service(api)
    )
    required = {
        "status", "parent_folder_id", "remote_folder_id", "remote_folder_name",
        "uploaded_files", "skipped_files", "failed_files",
    }
    assert required <= result.keys()
    assert result["status"] == "success"
    assert result["failed_files"] == []
