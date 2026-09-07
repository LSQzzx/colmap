#!/usr/bin/env python3
"""Bundle ONNX Runtime provider plugins into an auditwheel-repaired wheel."""

import argparse
import base64
import csv
import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

PROVIDER_NAMES = (
    "libonnxruntime_providers_shared.so",
    "libonnxruntime_providers_cuda.so",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_wheel", type=Path, help="Unrepaired wheel passed to auditwheel"
    )
    parser.add_argument(
        "wheelhouse", type=Path, help="Directory written by auditwheel"
    )
    parser.add_argument(
        "install_prefix",
        type=Path,
        help="COLMAP installation prefix containing ONNX Runtime libraries",
    )
    return parser.parse_args()


def find_repaired_wheel(source_wheel: Path, wheelhouse: Path) -> Path:
    distribution, version, python_tag, abi_tag, _ = source_wheel.name.split(
        "-", 4
    )
    wheel_pattern = f"{distribution}-{version}-{python_tag}-{abi_tag}-*.whl"
    matches = sorted(wheelhouse.glob(wheel_pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one repaired wheel for {source_wheel.name}, "
            f"found: {matches}"
        )
    return matches[0]


def find_provider_libraries(install_prefix: Path) -> tuple[Path, ...]:
    search_directories = (install_prefix / "lib", install_prefix / "lib64")
    providers = []
    for name in PROVIDER_NAMES:
        matches = [
            directory / name
            for directory in search_directories
            if (directory / name).is_file()
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one {name} below {install_prefix}, found: {matches}"
            )
        providers.append(matches[0])
    return tuple(providers)


def record_hash(path: Path) -> str:
    checksum = hashlib.sha256(path.read_bytes()).digest()
    encoded_checksum = base64.urlsafe_b64encode(checksum)
    digest = encoded_checksum.rstrip(b"=").decode("ascii")
    return f"sha256={digest}"


def write_record(root: Path) -> None:
    record_path = next(root.glob("*.dist-info/RECORD"))
    rows = []
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path != record_path
        ),
        key=lambda path: path.as_posix(),
    )
    for path in files:
        rows.append(
            (
                path.relative_to(root).as_posix(),
                record_hash(path),
                str(path.stat().st_size),
            )
        )
    rows.append((record_path.relative_to(root).as_posix(), "", ""))
    with record_path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(rows)


def main() -> None:
    args = parse_args()
    wheel = find_repaired_wheel(args.source_wheel, args.wheelhouse)
    providers = find_provider_libraries(args.install_prefix)

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory) / "wheel"
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root)
        library_directories = [
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.endswith(".libs")
        ]
        if len(library_directories) != 1:
            raise RuntimeError(
                "Expected one wheel library directory, "
                f"found: {library_directories}"
            )
        for provider in providers:
            shutil.copy2(provider, library_directories[0] / provider.name)
        write_record(root)

        replacement = wheel.with_suffix(".replacement.whl")
        with zipfile.ZipFile(
            replacement, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            files = sorted(
                (path for path in root.rglob("*") if path.is_file()),
                key=lambda path: path.as_posix(),
            )
            for path in files:
                archive.write(path, path.relative_to(root).as_posix())
        os.replace(replacement, wheel)
    print(f"Bundled ONNX Runtime providers into: {wheel}")


if __name__ == "__main__":
    main()
