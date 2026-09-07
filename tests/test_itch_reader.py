import gzip

import pytest

from merit.data.nasdaq.itch_reader import ITCHFileError, read_binaryfile


def write_binaryfile(path, messages: list[bytes]) -> None:
    with gzip.open(path, "wb") as file:
        for message in messages:
            file.write(len(message).to_bytes(2, "big"))
            file.write(message)

        file.write(b"\x00\x00")


def test_reads_binaryfile_messages(tmp_path) -> None:
    path = tmp_path / "sample.txt.gz"
    messages = [b"ABC", b"HELLO", b"\x01\x02\x03\x04"]

    write_binaryfile(path, messages)

    assert list(read_binaryfile(path)) == messages


def test_zero_length_message_ends_session(tmp_path) -> None:
    path = tmp_path / "sample.txt.gz"

    with gzip.open(path, "wb") as file:
        file.write((3).to_bytes(2, "big"))
        file.write(b"ABC")
        file.write(b"\x00\x00")
        file.write((4).to_bytes(2, "big"))
        file.write(b"DEFG")

    assert list(read_binaryfile(path)) == [b"ABC"]


def test_missing_end_marker_is_rejected(tmp_path) -> None:
    path = tmp_path / "sample.txt.gz"

    with gzip.open(path, "wb") as file:
        file.write((3).to_bytes(2, "big"))
        file.write(b"ABC")

    with pytest.raises(
        ITCHFileError,
        match="session terminator",
    ):
        list(read_binaryfile(path))


def test_truncated_message_is_rejected(tmp_path) -> None:
    path = tmp_path / "sample.txt.gz"

    with gzip.open(path, "wb") as file:
        file.write((5).to_bytes(2, "big"))
        file.write(b"ABC")

    with pytest.raises(
        ITCHFileError,
        match="Truncated message",
    ):
        list(read_binaryfile(path))


def test_incomplete_header_is_rejected(tmp_path) -> None:
    path = tmp_path / "sample.txt.gz"

    with gzip.open(path, "wb") as file:
        file.write(b"\x00")

    with pytest.raises(
        ITCHFileError,
        match="Incomplete BinaryFILE message header",
    ):
        list(read_binaryfile(path))