from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi


def download_file(
    url: str,
    dest: Path,
    max_attempts: int = 20,
    wait_seconds: int = 10,
) -> None:
    last_status: int | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(url) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                if status == 200:
                    with dest.open("wb") as f:
                        shutil.copyfileobj(resp, f)
                    break

                last_status = status
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_seconds = int(retry_after)
                    except ValueError:
                        pass
        except urllib.error.HTTPError as exc:
            status = exc.code
            last_status = status
            if status not in {202, 429} and status < 500:
                raise RuntimeError(f"Download failed: {exc}") from exc

        if attempt < max_attempts:
            time.sleep(wait_seconds)
    else:
        raise RuntimeError(f"Download did not become ready (last status {last_status})")

    if dest.stat().st_size == 0:
        raise RuntimeError("Downloaded file is empty")


def extract_tarball(tar_path: Path, extract_dir: Path) -> None:
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)


def upload_folder(repo_id: str, folder: Path, token: str | None) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        path_in_repo="",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Chemeleon checkpoints and upload to Hugging Face."
    )
    parser.add_argument(
        "--tar-path",
        default=None,
        help="Path to a local checkpoints.tar.gz (skips download)",
    )
    parser.add_argument(
        "--tar-url",
        default="https://figshare.com/ndownloader/files/54966305",
        help="Chemeleon checkpoints tarball URL",
    )
    parser.add_argument(
        "--repo-id",
        default="G-reen/chemeleon_reupload",
        help="Hugging Face repo id (e.g. org/name)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (or set HUGGINGFACE_HUB_TOKEN)",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Optional working directory (default: temp)",
    )
    args = parser.parse_args()

    token = args.token
    work_dir: Path | None = Path(args.work_dir) if args.work_dir else None

    tar_path_override = Path(args.tar_path).expanduser() if args.tar_path else None

    if work_dir:
        work_dir.mkdir(parents=True, exist_ok=True)
        tar_path = work_dir / "checkpoints.tar.gz"
        extract_dir = work_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        if tar_path_override:
            if not tar_path_override.is_file():
                raise RuntimeError(f"Tar path not found: {tar_path_override}")
            shutil.copy2(tar_path_override, tar_path)
        else:
            download_file(args.tar_url, tar_path)
        extract_tarball(tar_path, extract_dir)
        upload_folder(args.repo_id, extract_dir, token)
    else:
        with tempfile.TemporaryDirectory(prefix="chemeleon_upload_") as tmp:
            tmp_path = Path(tmp)
            tar_path = tmp_path / "checkpoints.tar.gz"
            extract_dir = tmp_path / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            if tar_path_override:
                if not tar_path_override.is_file():
                    raise RuntimeError(f"Tar path not found: {tar_path_override}")
                shutil.copy2(tar_path_override, tar_path)
            else:
                download_file(args.tar_url, tar_path)
            extract_tarball(tar_path, extract_dir)
            upload_folder(args.repo_id, extract_dir, token)


if __name__ == "__main__":
    main()
