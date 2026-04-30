import argparse
import logging
import urllib.request
from pathlib import Path

from tqdm import tqdm


LOGGER = logging.getLogger("download_checkpoints")

POINTBERT_DVAE_URL = "https://cloud.tsinghua.edu.cn/f/c76274f9afb34cdbb57e/?dl=1"
POINTBERT_TRANSFORMER_URL = "https://cloud.tsinghua.edu.cn/f/202b29805eea45d7be92/?dl=1"
POINTMAE_URL = "https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth"
PCPMAE_URL = "https://drive.google.com/file/d/184PgZnPUIlxVB4Ipmw3UXMeGLimyiTr7/view?usp=drive_link"
POINTGPT_URL = "https://drive.google.com/file/d/1gTFI327kXVDFQ90JfYX0zIS4opM1EkqX/view?usp=drive_link"


DIRECT_CHECKPOINTS = (
    ("Point-BERT dVAE", POINTBERT_DVAE_URL, "dvae.pth"),
    ("Point-BERT transformer", POINTBERT_TRANSFORMER_URL, "Point-BERT.pth"),
    ("Point-MAE", POINTMAE_URL, "pointmae_pretrain.pth"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pretrained point-cloud backbone checkpoints."
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Directory where checkpoints will be downloaded.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing checkpoint files.",
    )
    parser.add_argument(
        "--skip-google-drive",
        action="store_true",
        help="Skip Google Drive checkpoints that require gdown.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class DownloadProgressBar(tqdm):
    def update_to(self, block_num: int = 1, block_size: int = 1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(block_num * block_size - self.n)


def download_direct(name: str, url: str, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        LOGGER.info("Skipping %s; file already exists: %s", name, output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s to %s", name, output_path)
    with DownloadProgressBar(
        unit="B",
        unit_scale=True,
        miniters=1,
        desc=output_path.name,
    ) as progress:
        urllib.request.urlretrieve(url, output_path, reporthook=progress.update_to)


def require_gdown():
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required to download Google Drive checkpoints. "
            "Install it with `python3 -m pip install --user gdown`, "
            "or rerun with `--skip-google-drive`."
        ) from exc
    return gdown


def download_google_drive(destination: Path, overwrite: bool) -> None:
    gdown = require_gdown()

    pcpmae_path = destination / "PCP-MAE-300.pth"
    if pcpmae_path.exists() and not overwrite:
        LOGGER.info("Skipping PCP-MAE; file already exists: %s", pcpmae_path)
    else:
        LOGGER.info("Downloading PCP-MAE-300 to %s", pcpmae_path)
        gdown.download(
            url=PCPMAE_URL,
            output=str(pcpmae_path),
            quiet=False,
            fuzzy=True,
        )

    pointgpt_path = destination / "pointgpt_pretrained.pth"
    if pointgpt_path.exists() and not overwrite:
        LOGGER.info("Skipping PointGPT; file already exists: %s", pointgpt_path)
    else:
        LOGGER.info("Downloading PointGPT to %s", pointgpt_path)
        gdown.download(
            url=POINTGPT_URL,
            output=str(pointgpt_path),
            quiet=False,
            fuzzy=True,
        )


def main() -> int:
    args = parse_args()
    configure_logging()

    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading checkpoints into %s", destination)

    for name, url, filename in DIRECT_CHECKPOINTS:
        download_direct(
            name=name,
            url=url,
            output_path=destination / filename,
            overwrite=args.overwrite,
        )

    if args.skip_google_drive:
        LOGGER.info("Skipping Google Drive checkpoints")
    else:
        download_google_drive(destination=destination, overwrite=args.overwrite)

    LOGGER.info("Checkpoint download complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
