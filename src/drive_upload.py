import mimetypes
from pathlib import Path


def _create_drive_folder(service, name: str, parent_id: str) -> str:
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    created = service.files().create(body=metadata, fields="id,parents").execute()
    if parent_id not in created.get("parents", []):
        raise RuntimeError("Created archive folder does not have the requested parent ID")
    return created["id"]


def upload_archive_to_drive(
    archive_dir: str | Path,
    parent_folder_id: str,
    service=None,
) -> dict:
    """Upload one immutable local archive beneath a canonical Drive folder ID."""
    archive = Path(archive_dir)
    if not archive.is_dir():
        raise FileNotFoundError(f"Archive does not exist: {archive}")
    required = {"run_info.txt", "primary_results.csv", "research_summary.md"}
    missing = sorted(name for name in required if not (archive / name).is_file())
    if missing:
        raise RuntimeError(f"Archive is incomplete; missing: {missing}")

    if service is None:
        try:
            from google.colab import auth
        except ImportError as exc:
            raise RuntimeError("Google Drive upload requires Colab authentication") from exc
        from googleapiclient.discovery import build
        auth.authenticate_user()
        service = build("drive", "v3")

    parent = service.files().get(
        fileId=parent_folder_id, fields="id,name,mimeType,capabilities"
    ).execute()
    if parent.get("mimeType") != "application/vnd.google-apps.folder":
        raise ValueError("Configured Drive parent ID is not a folder")
    if not parent.get("capabilities", {}).get("canAddChildren", False):
        raise PermissionError("Configured Drive folder does not allow uploads")

    from googleapiclient.http import MediaFileUpload

    root_id = _create_drive_folder(service, archive.name, parent_folder_id)
    folder_ids = {archive: root_id}
    uploaded = []
    failures = []
    for path in sorted(archive.rglob("*")):
        relative = path.relative_to(archive)
        parent_path = path.parent
        try:
            if path.is_dir():
                folder_ids[path] = _create_drive_folder(
                    service, path.name, folder_ids[parent_path]
                )
                continue
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            metadata = {"name": path.name, "parents": [folder_ids[parent_path]]}
            media = MediaFileUpload(str(path), mimetype=mime_type, resumable=True)
            created = service.files().create(
                body=metadata, media_body=media, fields="id,name,parents"
            ).execute()
            if folder_ids[parent_path] not in created.get("parents", []):
                raise RuntimeError("Uploaded file has an unexpected parent")
            uploaded.append(str(relative))
        except Exception as exc:
            failures.append({"path": str(relative), "error": repr(exc)})

    listing = service.files().list(
        q=f"'{root_id}' in parents and trashed = false",
        fields="files(id,name,mimeType)",
        pageSize=1000,
    ).execute()
    if failures:
        raise RuntimeError(
            f"Drive archive upload incomplete: {len(failures)} failures; "
            f"local archive retained at {archive}; details={failures}"
        )
    return {
        "local_archive": str(archive),
        "archive_name": archive.name,
        "parent_folder_id": parent_folder_id,
        "archive_folder_id": root_id,
        "uploaded_file_count": len(uploaded),
        "top_level_item_count": len(listing.get("files", [])),
        "failed_files": failures,
        "status": "success",
    }
