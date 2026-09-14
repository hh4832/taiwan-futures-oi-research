"""Reliable, resumable, and idempotent Google Drive archive upload."""

from __future__ import annotations

import mimetypes
import socket
import time
from pathlib import Path
from typing import Callable


DRIVE_HTTP_TIMEOUT_SECONDS = 60
DRIVE_MAX_RETRIES = 3
DRIVE_RETRY_BACKOFF_SECONDS = (2, 4, 8)
DRIVE_UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024
DRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
TRANSIENT_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})


class DriveUploadError(RuntimeError):
    """A Drive upload failure with non-secret diagnostic context."""


def _build_drive_service():
    """Build an authenticated Drive client with a real transport timeout."""
    try:
        from google.colab import auth
    except ImportError as exc:
        raise RuntimeError("Google Drive upload requires Colab authentication") from exc

    import google.auth
    import httplib2
    from google_auth_httplib2 import AuthorizedHttp
    from googleapiclient.discovery import build

    auth.authenticate_user()
    credentials, _ = google.auth.default()
    transport = httplib2.Http(timeout=DRIVE_HTTP_TIMEOUT_SECONDS)
    authorized_http = AuthorizedHttp(credentials, http=transport)
    return build(
        "drive", "v3", http=authorized_http, cache_discovery=False
    )


def _http_status(exc: Exception) -> int | None:
    status = getattr(getattr(exc, "resp", None), "status", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _is_transient(exc: Exception) -> bool:
    status = _http_status(exc)
    if status is not None:
        return status in TRANSIENT_HTTP_STATUSES
    return isinstance(exc, (TimeoutError, socket.timeout, ConnectionError, OSError))


def _with_retry(
    operation: Callable[[], object],
    *,
    stage: str,
    file_name: str = "-",
    sleep: Callable[[float], None] = time.sleep,
):
    """Retry only transient failures, at most three times."""
    for attempt in range(1, DRIVE_MAX_RETRIES + 2):
        try:
            return operation()
        except Exception as exc:
            retries_used = attempt - 1
            if not _is_transient(exc) or retries_used >= DRIVE_MAX_RETRIES:
                raise DriveUploadError(
                    f"Drive request failed: stage={stage}; file={file_name}; "
                    f"attempt={attempt}; error_type={type(exc).__name__}; error={exc}"
                ) from exc
            delay = DRIVE_RETRY_BACKOFF_SECONDS[retries_used]
            print(
                f"[Drive] Retry {attempt}/{DRIVE_MAX_RETRIES} in {delay}s: "
                f"stage={stage}, file={file_name}, error={type(exc).__name__}"
            )
            sleep(delay)
    raise AssertionError("finite retry loop exhausted unexpectedly")


def _escape_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _list_children(service, parent_id: str, name: str | None = None):
    query = f"'{_escape_query(parent_id)}' in parents and trashed = false"
    if name is not None:
        query += f" and name = '{_escape_query(name)}'"
    items, page_token = [], None
    while True:
        request = service.files().list(
            q=query,
            fields="nextPageToken,files(id,name,mimeType,parents,size)",
            pageSize=1000,
            pageToken=page_token,
        )
        response = _with_retry(
            request.execute, stage="list_children", file_name=name or "-"
        )
        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return items


def _find_exact_child(service, parent_id: str, name: str, *, folder: bool):
    matches = [
        item for item in _list_children(service, parent_id, name)
        if (item.get("mimeType") == DRIVE_FOLDER_MIME_TYPE) == folder
    ]
    if len(matches) > 1:
        raise DriveUploadError(
            f"Drive duplicate ambiguity: stage=find_child; file={name}; "
            f"attempt=1; error_type=DuplicateExactName; matches={len(matches)}"
        )
    return matches[0] if matches else None


def _create_folder(service, name: str, parent_id: str) -> str:
    request = service.files().create(
        body={
            "name": name,
            "mimeType": DRIVE_FOLDER_MIME_TYPE,
            "parents": [parent_id],
        },
        fields="id,name,parents",
    )
    result = _with_retry(
        request.execute, stage="create_folder", file_name=name
    )
    if parent_id not in result.get("parents", []):
        raise DriveUploadError(
            f"Drive request failed: stage=verify_folder; file={name}; "
            "attempt=1; error_type=UnexpectedParent"
        )
    return result["id"]


def _find_or_create_folder(service, name: str, parent_id: str):
    existing = _find_exact_child(service, parent_id, name, folder=True)
    if existing:
        return existing["id"], False
    return _create_folder(service, name, parent_id), True


def _upload_resumable_file(service, path: Path, parent_id: str):
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(
        str(path),
        mimetype=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        chunksize=DRIVE_UPLOAD_CHUNK_SIZE,
        resumable=True,
    )
    request = service.files().create(
        body={"name": path.name, "parents": [parent_id]},
        media_body=media,
        fields="id,name,parents,size",
    )
    response = None
    while response is None:
        _, response = _with_retry(
            lambda: request.next_chunk(num_retries=0),
            stage="upload_chunk",
            file_name=path.name,
        )
    if parent_id not in response.get("parents", []):
        raise DriveUploadError(
            f"Drive request failed: stage=verify_file; file={path.name}; "
            "attempt=1; error_type=UnexpectedParent"
        )


def _validate_archive(archive: Path):
    if not archive.exists():
        raise FileNotFoundError(f"Archive does not exist: {archive}")
    if not archive.is_dir():
        raise NotADirectoryError(f"Archive is not a directory: {archive}")
    required = {"run_info.txt", "primary_results.csv", "research_summary.md"}
    missing = sorted(name for name in required if not (archive / name).is_file())
    if missing:
        raise RuntimeError(f"Archive is incomplete; missing: {missing}")
    return sorted(path for path in archive.rglob("*") if path.is_file())


def upload_archive_to_drive(
    archive_dir: str | Path,
    parent_folder_id: str,
    service=None,
) -> dict:
    """Upload or resume one exact archive without creating duplicate runs."""
    archive = Path(archive_dir)
    files = _validate_archive(archive)
    print(f"[Drive] Preparing upload: {archive.name} ({len(files)} files)")
    if service is None:
        print("[Drive] Creating authenticated timeout-enabled transport")
        service = _build_drive_service()

    parent = _with_retry(
        service.files().get(
            fileId=parent_folder_id,
            fields="id,name,mimeType,capabilities",
        ).execute,
        stage="validate_parent",
        file_name=archive.name,
    )
    if parent.get("mimeType") != DRIVE_FOLDER_MIME_TYPE:
        raise ValueError("Configured Drive parent ID is not a folder")
    if not parent.get("capabilities", {}).get("canAddChildren", False):
        raise PermissionError("Configured Drive folder does not allow uploads")

    print(f"[Drive] Finding or creating remote folder: {archive.name}")
    root_id, created = _find_or_create_folder(
        service, archive.name, parent_folder_id
    )
    print(f"[Drive] {'Created' if created else 'Resuming'} folder: {root_id}")

    folder_ids = {Path(): root_id}
    directories = sorted(
        {path.relative_to(archive).parent for path in files if path.parent != archive},
        key=lambda item: (len(item.parts), item.as_posix()),
    )
    for relative_dir in directories:
        parent_relative = (
            relative_dir.parent if relative_dir.parent != Path(".") else Path()
        )
        folder_ids[relative_dir], _ = _find_or_create_folder(
            service, relative_dir.name, folder_ids[parent_relative]
        )

    uploaded, skipped, failed = [], [], []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(archive)
        relative_name = relative.as_posix()
        parent_relative = relative.parent if relative.parent != Path(".") else Path()
        print(f"[Drive] Uploading {index}/{len(files)}: {relative_name}")
        try:
            existing = _find_exact_child(
                service, folder_ids[parent_relative], path.name, folder=False
            )
            if existing:
                remote_size = int(existing.get("size", -1))
                if remote_size == path.stat().st_size:
                    skipped.append(relative_name)
                    print(f"[Drive] Skipped identical file: {relative_name}")
                    continue
                raise DriveUploadError(
                    f"Drive immutable conflict: stage=compare_file; "
                    f"file={relative_name}; attempt=1; error_type=SizeMismatch"
                )
            _upload_resumable_file(service, path, folder_ids[parent_relative])
            uploaded.append(relative_name)
        except Exception as exc:
            failed.append({
                "path": relative_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(f"[Drive] Upload failed: {relative_name} ({type(exc).__name__})")
            break

    if failed:
        raise DriveUploadError(
            f"Drive archive upload incomplete: stage=upload_files; "
            f"file={failed[0]['path']}; attempt=1; "
            f"error_type={failed[0]['error_type']}; details={failed}"
        )

    print(
        f"[Drive] Upload success: uploaded={len(uploaded)}, skipped={len(skipped)}"
    )
    return {
        "status": "success",
        "parent_folder_id": parent_folder_id,
        "remote_folder_id": root_id,
        "remote_folder_name": archive.name,
        "uploaded_files": uploaded,
        "skipped_files": skipped,
        "failed_files": failed,
        # Backward-compatible fields retained for existing callers.
        "local_archive": str(archive),
        "archive_name": archive.name,
        "archive_folder_id": root_id,
        "uploaded_file_count": len(uploaded),
    }
