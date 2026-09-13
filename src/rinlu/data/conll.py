from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable


SENTIMENT_LABELS = ("negative", "neutral", "positive")
KNOWN_LANGUAGE_TAGS = {"Hin", "Eng", "O", "EMT"}
_MOJIBAKE_MARKERS = frozenset("ÂÃâðïà�")


@dataclass(frozen=True)
class SentimentExample:
    uid: str
    text: str
    tokens: tuple[str, ...]
    language_tags: tuple[str, ...]
    label: str | None
    source_split: str

    def to_json(self, include_label: bool = True) -> dict:
        row = asdict(self)
        row["tokens"] = list(self.tokens)
        row["language_tags"] = list(self.language_tags)
        if not include_label:
            row.pop("label", None)
        return row


def reconstruct_text(tokens: Iterable[str]) -> str:
    """Reconstruct the supplied token sequence without inventing missing text."""
    return " ".join(token for token in tokens if token).strip()


def duplicate_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def repair_mojibake(text: str) -> str:
    """Repair a UTF-8 sequence accidentally decoded through Windows-1252.

    The repair is accepted only when round-tripping is valid and removes known
    mojibake markers. This protects ordinary non-ASCII Hinglish text.
    """
    original_score = sum(character in _MOJIBAKE_MARKERS for character in text)
    if original_score == 0:
        return text
    def windows_1252_bytes(value: str) -> bytes:
        encoded = bytearray()
        for character in value:
            try:
                encoded.extend(character.encode("cp1252"))
            except UnicodeEncodeError:
                # Some broken decoders preserve undefined 0x80--0x9F bytes as
                # matching C1 controls. Recover those bytes explicitly.
                if ord(character) <= 0xFF:
                    encoded.append(ord(character))
                else:
                    raise
        return bytes(encoded)

    best = text
    for _ in range(2):
        candidates = [best]
        for encoder in (windows_1252_bytes, lambda value: value.encode("latin1")):
            try:
                candidates.append(encoder(best).decode("utf-8"))
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        candidate = min(
            candidates,
            key=lambda value: (
                sum(character in _MOJIBAKE_MARKERS for character in value),
                value == best,
            ),
        )
        if candidate == best:
            break
        best = candidate
    return best


def parse_conll(path: Path, split: str, require_labels: bool) -> list[SentimentExample]:
    records: list[SentimentExample] = []
    uid: str | None = None
    label: str | None = None
    tokens: list[str] = []
    tags: list[str] = []

    def flush() -> None:
        nonlocal uid, label, tokens, tags
        if uid is None:
            return
        if not tokens:
            raise ValueError(f"{path}: record {uid!r} has no tokens")
        records.append(
            SentimentExample(
                uid=uid,
                text=reconstruct_text(tokens),
                tokens=tuple(tokens),
                language_tags=tuple(tags),
                label=label,
                source_split=split,
            )
        )
        uid, label, tokens, tags = None, None, [], []

    with path.open("r", encoding="utf-8-sig", newline=None) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t")
            if fields[0] == "meta":
                flush()
                if len(fields) < 2 or not fields[1].strip():
                    raise ValueError(f"{path}:{line_number}: malformed meta line")
                uid = fields[1].strip()
                label = fields[2].strip() if len(fields) >= 3 and fields[2].strip() else None
                if require_labels and label not in SENTIMENT_LABELS:
                    raise ValueError(
                        f"{path}:{line_number}: invalid sentiment label {label!r}"
                    )
                continue
            if uid is None:
                raise ValueError(f"{path}:{line_number}: token appears before metadata")
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected token and language tag")
            token, tag = fields
            # A few source rows contain only a language tag (for example ``\tO``).
            # They carry no recoverable text, so exclude them and expose their count
            # in the preparation manifest instead of inventing a replacement token.
            if not token:
                continue
            tokens.append(repair_mojibake(token))
            tags.append(tag.strip())

    flush()
    ids = [record.uid for record in records]
    if len(ids) != len(set(ids)):
        repeated = [uid for uid, count in Counter(ids).items() if count > 1]
        raise ValueError(f"{path}: duplicate UIDs: {repeated[:10]}")
    return records


def read_test_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Uid", "Sentiment"]:
            raise ValueError(f"{path}: expected columns Uid,Sentiment")
        for line_number, row in enumerate(reader, start=2):
            uid = row["Uid"].strip()
            label = row["Sentiment"].strip()
            if not uid or label not in SENTIMENT_LABELS:
                raise ValueError(f"{path}:{line_number}: invalid test-label row")
            if uid in labels:
                raise ValueError(f"{path}:{line_number}: duplicate UID {uid}")
            labels[uid] = label
    return labels


def attach_labels(
    records: Iterable[SentimentExample], labels: dict[str, str]
) -> list[SentimentExample]:
    records = list(records)
    record_ids = {record.uid for record in records}
    if record_ids != set(labels):
        missing = sorted(record_ids - set(labels))
        extra = sorted(set(labels) - record_ids)
        raise ValueError(f"test UID mismatch; missing={missing[:5]}, extra={extra[:5]}")
    return [replace(record, label=labels[record.uid]) for record in records]


def write_jsonl(
    path: Path, records: Iterable[SentimentExample], include_label: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            json.dump(record.to_json(include_label=include_label), handle, ensure_ascii=False)
            handle.write("\n")


def read_jsonl(path: Path, require_labels: bool = True) -> list[SentimentExample]:
    records: list[SentimentExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.get("label")
            if require_labels and label not in SENTIMENT_LABELS:
                raise ValueError(f"{path}:{line_number}: missing or invalid label")
            records.append(
                SentimentExample(
                    uid=str(row["uid"]),
                    text=str(row["text"]),
                    tokens=tuple(row.get("tokens", [])),
                    language_tags=tuple(row.get("language_tags", [])),
                    label=label,
                    source_split=str(row.get("source_split", "unknown")),
                )
            )
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_empty_token_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8-sig", newline=None) as handle:
        for line in handle:
            if line.startswith("\t"):
                count += 1
    return count


def count_repairable_token_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8-sig", newline=None) as handle:
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) == 2 and fields[0] and repair_mojibake(fields[0]) != fields[0]:
                count += 1
    return count
