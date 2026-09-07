from collections.abc import Iterator
import gzip
from pathlib import Path


class ITCHFileError(ValueError):
    pass


def read_binaryfile(path: str | Path) -> Iterator[bytes]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with gzip.open(path, "rb") as file:
        while True:
            length_bytes = file.read(2)

            if not length_bytes:
                raise ITCHFileError("File ended before session terminator")

            if len(length_bytes) != 2:
                raise ITCHFileError("Incomplete BinaryFILE message header")

            length = int.from_bytes(length_bytes, "big")

            if length == 0:
                return

            message = file.read(length)

            if len(message) != length:
                raise ITCHFileError(
                    f"Truncated message: expected {length} bytes, got {len(message)}"
                )

            yield message