from datetime import date

from merit.data.nasdaq.pipeline import replay_itch_file


def test_pipeline_streams_decoded_events(tmp_path, monkeypatch) -> None:
    messages = [b"message-1", b"message-2", b"message-3"]

    def fake_reader(path):
        assert path == "sample.gz"
        yield from messages

    class FakeDecoder:
        def __init__(self, trading_date):
            assert trading_date == date(2026, 1, 2)

        def decode(self, message):
            return f"decoded:{message.decode()}"

    monkeypatch.setattr(
        "merit.data.nasdaq.pipeline.read_binaryfile",
        fake_reader,
    )
    monkeypatch.setattr(
        "merit.data.nasdaq.pipeline.ITCHDecoder",
        FakeDecoder,
    )

    events = list(
        replay_itch_file(
            "sample.gz",
            date(2026, 1, 2),
        )
    )

    assert events == [
        "decoded:message-1",
        "decoded:message-2",
        "decoded:message-3",
    ]