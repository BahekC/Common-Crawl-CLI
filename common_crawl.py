#!/usr/bin/env python3
"""CLI-поиск архивных страниц Common Crawl."""

from __future__ import annotations

import argparse
import gzip
import io
import re
import sys
import textwrap
import zlib
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

import cdx_toolkit
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator


DATA_ROOT = "https://data.commoncrawl.org"
USER_AGENT = "common-crawl-study-cli/2.0 (+https://commoncrawl.org/)"
DEFAULT_TIMEOUT = 30
SNIPPET_LENGTH = 260

# Каждая внутренняя группа — альтернативный набор основ, который должен
# встретиться целиком. Группы связаны логическим ИЛИ.
KEYWORD_ALIASES: dict[str, tuple[tuple[str, ...], ...]] = {
    "мгу": (
        ("мгу",),
        ("msu",),
        ("московск", "государствен", "университет"),
    ),
    "мфти": (("мфти",), ("mipt",), ("физтех",)),
    "пнипу": (
        ("пнипу",),
        ("pstu",),
        ("пермск", "политех"),
    ),
}


class CommonCrawlError(RuntimeError):
    """Понятная пользователю ошибка Common Crawl или входных данных."""


@dataclass(frozen=True)
class Capture:
    url: str
    timestamp: str
    filename: str
    offset: int
    length: int
    title: str = "—"
    snippet: str = ""
    content: str = field(default="", repr=False, compare=False)

    @property
    def archived_at(self) -> str:
        try:
            return datetime.strptime(self.timestamp, "%Y%m%d%H%M%S").strftime(
                "%Y-%m-%d %H:%M"
            )
        except ValueError:
            return self.timestamp or "—"

    @property
    def crawl(self) -> str:
        match = re.search(r"crawl-data/(CC-MAIN-\d{4}-\d{2})/", self.filename)
        return match.group(1) if match else "—"


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ожидается целое число") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("значение должно быть больше нуля")
    return number


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="common_crawl.py",
        description=(
            "Поиск архивных страниц через CDX-индекс Common Crawl. "
            "Без --show-text слова ищутся в URL; с --show-text и доменом "
            "проверяется содержимое точечно загруженных WARC-записей."
        ),
        epilog=(
            "Примеры:\n"
            "  python common_crawl.py news --domain pstu.ru --limit 5\n"
            "  python common_crawl.py commoncrawl.org/get-started --show-text\n"
            "  python common_crawl.py 'кафедра' 'ИТАС' --domain pstu.ru --show-text\n\n"
            "Без --show-text и с --domain все слова должны встречаться в URL. "
            "Для проверки текста используйте --show-text и задайте домен или URL."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "keywords",
        nargs="+",
        help="одно или несколько ключевых слов; также можно передать URL",
    )
    parser.add_argument(
        "--domain",
        metavar="DOMAIN",
        help="ограничить поиск доменом и его поддоменами, например pstu.ru",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=10,
        metavar="N",
        help="максимальное число результатов (по умолчанию: 10)",
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="точечно загрузить WARC и показать заголовок и фрагмент страницы",
    )
    parser.add_argument(
        "--crawls",
        type=_positive_int,
        default=3,
        metavar="N",
        help="число последних обходов Common Crawl (по умолчанию: 3)",
    )
    return parser


def _normalize_domain(domain: str) -> str:
    candidate = domain.strip()
    if "://" not in candidate:
        candidate = "//" + candidate
    host = urlsplit(candidate).hostname
    if not host:
        raise CommonCrawlError(f"Некорректный домен: {domain!r}")
    return host.encode("idna").decode("ascii")


def _looks_like_url(value: str) -> bool:
    return "." in value or "/" in value or "://" in value


def _transliterate_ru(value: str) -> str:
    table = str.maketrans(
        {
            "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
            "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
            "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
            "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
            "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch",
            "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "",
            "э": "e", "ю": "yu", "я": "ya",
        }
    )
    return value.casefold().translate(table)


def _url_hints(keywords: Sequence[str]) -> list[str]:
    """Создаёт ASCII-подсказки для релевантных URL."""
    hints: list[str] = []
    for keyword in keywords:
        alias_groups = KEYWORD_ALIASES.get(keyword.casefold().strip(), ())
        for group in alias_groups:
            for alias in group:
                if alias.isascii() and alias not in hints:
                    hints.append(alias)
        transliterated = _transliterate_ru(keyword)
        parts = re.findall(r"[a-z0-9]{3,}", transliterated)
        variants = (["-".join(parts)] if len(parts) > 1 else []) + sorted(
            parts, key=len, reverse=True
        )
        for variant in variants:
            for hint in (variant, variant.replace("kh", "h")):
                if hint and hint not in hints:
                    hints.append(hint)
    return hints[:8]


def _query_specs(
    keywords: Sequence[str], domain: str | None, *, content_search: bool
) -> list[tuple[str, str, str | None]]:
    """Возвращает (URL, matchType, URL-подсказка) для CDXFetcher."""
    if domain:
        host = _normalize_domain(domain)
        if not content_search:
            return [(host, "domain", None)]
        specs: list[tuple[str, str, str | None]] = [
            (host, "exact", None),
            (f"www.{host}", "exact", None),
        ]
        specs.extend((host, "domain", hint) for hint in _url_hints(keywords))
        specs.extend(
            [
                (f"www.{host}", "host", None),
                (host, "host", None),
                (host, "domain", None),
            ]
        )
        return specs

    url_keywords = [keyword for keyword in keywords if _looks_like_url(keyword)]
    query_keywords = url_keywords or list(keywords)
    return [
        (
            keyword.strip(),
            "prefix" if _looks_like_url(keyword) else "exact",
            None,
        )
        for keyword in query_keywords
        if keyword.strip()
    ]


def _capture_from_mapping(item: Mapping[str, Any]) -> Capture | None:
    try:
        return Capture(
            url=str(item["url"]),
            timestamp=str(item["timestamp"]),
            filename=str(item["filename"]),
            offset=int(item["offset"]),
            length=int(item["length"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def search_cdx(
    keywords: Sequence[str],
    *,
    domain: str | None,
    limit: int,
    crawls: int = 3,
    filter_url_keywords: bool = True,
    fetcher: Any | None = None,
) -> list[Capture]:
    """Ищет WARC-координаты через высокоуровневый cdx-toolkit."""
    if not keywords:
        raise CommonCrawlError("Нужно указать хотя бы одно ключевое слово")
    specs = _query_specs(
        keywords, domain, content_search=not filter_url_keywords
    )
    if not specs:
        raise CommonCrawlError("Ключевые слова не должны быть пустыми")

    try:
        cdx = fetcher or cdx_toolkit.CDXFetcher(source="cc", crawl=str(crawls))
    except Exception as exc:
        raise CommonCrawlError(f"Не удалось инициализировать CDX: {exc}") from exc

    if filter_url_keywords:
        per_query_limit = min(max(limit * 20, 100), 1000)
    else:
        per_query_limit = max(3, (limit + len(specs) - 1) // len(specs))

    found: list[Capture] = []
    seen: set[tuple[str, str]] = set()
    lowered_keywords = [keyword.casefold() for keyword in keywords]
    query_errors: list[str] = []

    for query, match_type, url_hint in specs:
        filters = ["=status:200", "mime:text/html"]
        if url_hint:
            filters.append(f"url:{url_hint}")
        if domain and filter_url_keywords:
            filters.extend(
                f"url:{keyword.casefold()}"
                for keyword in keywords
                if keyword.isascii()
            )
        try:
            items = cdx.get(
                query,
                matchType=match_type,
                filter=filters,
                limit=per_query_limit,
            )
        except Exception as exc:
            query_errors.append(str(exc))
            continue

        for item in items:
            capture = _capture_from_mapping(item)
            if capture is None:
                continue
            if domain and filter_url_keywords:
                searchable_url = unquote(capture.url).casefold()
                if not all(word in searchable_url for word in lowered_keywords):
                    continue
            key = (capture.url, capture.timestamp)
            if key in seen:
                continue
            seen.add(key)
            found.append(capture)
            if len(found) >= limit:
                return found

    if not found and len(query_errors) == len(specs):
        detail = query_errors[-1] if query_errors else "неизвестная ошибка"
        raise CommonCrawlError(
            "CDX временно недоступен после повторных попыток cdx-toolkit: " + detail
        )
    return found[:limit]


def _headers_to_dict(headers: Any) -> dict[str, str]:
    if headers is None:
        return {}
    return {str(name).lower(): str(value) for name, value in headers.headers}


def _dechunk(payload: bytes) -> bytes:
    output = bytearray()
    position = 0
    while position < len(payload):
        line_end = payload.find(b"\r\n", position)
        if line_end < 0:
            return payload
        try:
            size = int(payload[position:line_end].split(b";", 1)[0], 16)
        except ValueError:
            return payload
        position = line_end + 2
        if size == 0:
            return bytes(output)
        output.extend(payload[position : position + size])
        position += size + 2
    return bytes(output)


def _decode_http_body(headers: Mapping[str, str], payload: bytes) -> bytes:
    if "chunked" in headers.get("transfer-encoding", "").lower():
        payload = _dechunk(payload)
    encoding = headers.get("content-encoding", "").lower()
    try:
        if "gzip" in encoding:
            return gzip.decompress(payload)
        if "deflate" in encoding:
            try:
                return zlib.decompress(payload)
            except zlib.error:
                return zlib.decompress(payload, -zlib.MAX_WBITS)
        if "br" in encoding:
            import brotli

            return brotli.decompress(payload)
    except (OSError, zlib.error) as exc:
        raise CommonCrawlError("Не удалось распаковать HTTP-тело из WARC") from exc
    return payload


def _detect_charset(headers: Mapping[str, str], payload: bytes) -> str:
    content_type = headers.get("content-type", "")
    match = re.search(r"charset\s*=\s*[\"']?([^;\s\"']+)", content_type, re.I)
    if match:
        return match.group(1)
    prefix = payload[:4096].decode("ascii", errors="ignore")
    match = re.search(r"<meta[^>]+charset\s*=\s*[\"']?([^\s\"'/>;]+)", prefix, re.I)
    return match.group(1) if match else "utf-8"


def parse_warc_record(compressed_record: bytes) -> tuple[str, str]:
    """Разбирает одну gzip-сжатую WARC-запись библиотекой warcio."""
    try:
        records = ArchiveIterator(io.BytesIO(compressed_record), arc2warc=True)
        record = next(records)
        headers = _headers_to_dict(record.http_headers)
        payload = record.content_stream().read()
    except (StopIteration, EOFError, OSError, ValueError) as exc:
        raise CommonCrawlError("Не удалось разобрать WARC-запись") from exc

    payload = _decode_http_body(headers, payload)
    charset = _detect_charset(headers, payload)
    try:
        html = payload.decode(charset, errors="replace")
    except LookupError:
        html = payload.decode("utf-8", errors="replace")

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else "—"
    for tag in soup(["script", "style", "noscript", "svg", "template"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    return title or "—", text


def enrich_with_text(
    capture: Capture,
    *,
    session: requests.Session | Any | None = None,
) -> Capture:
    """Загружает только диапазон одной WARC-записи и извлекает HTML."""
    client = session or requests.Session()
    end = capture.offset + capture.length - 1
    url = f"{DATA_ROOT}/{capture.filename.lstrip('/')}"
    try:
        response = client.get(
            url,
            headers={
                "Range": f"bytes={capture.offset}-{end}",
                "User-Agent": USER_AGENT,
            },
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise CommonCrawlError(f"Ошибка загрузки WARC: {exc}") from exc
    if response.status_code != 206:
        raise CommonCrawlError(
            f"Сервер не подтвердил Range-запрос (HTTP {response.status_code}); "
            "полная загрузка WARC отменена"
        )
    if len(response.content) != capture.length:
        raise CommonCrawlError(
            f"Сервер вернул {len(response.content)} байт вместо {capture.length}"
        )
    title, text = parse_warc_record(response.content)
    snippet = textwrap.shorten(text, width=SNIPPET_LENGTH, placeholder="…") if text else "—"
    return replace(capture, title=title, snippet=snippet, content=text)


def _keyword_stems(keywords: Sequence[str]) -> list[str]:
    stems: list[str] = []
    for keyword in keywords:
        for token in re.findall(r"[^\W_]+", keyword.casefold(), flags=re.UNICODE):
            token = token.rstrip("ьъ")
            if len(token) > 5:
                token = token[:-2]
            if token:
                stems.append(token)
    return stems


def _keyword_alternatives(keyword: str) -> tuple[tuple[str, ...], ...]:
    aliases = KEYWORD_ALIASES.get(keyword.casefold().strip())
    if aliases:
        return aliases
    stems = tuple(_keyword_stems([keyword]))
    return (stems,) if stems else ()


def _capture_contains_keywords(capture: Capture, keywords: Sequence[str]) -> bool:
    searchable = " ".join((capture.title, capture.content)).casefold()
    content_keywords = [keyword for keyword in keywords if not _looks_like_url(keyword)]
    groups = [_keyword_alternatives(keyword) for keyword in content_keywords]
    if not groups:
        return True
    if any(not alternatives for alternatives in groups):
        return False
    return all(
        any(all(stem in searchable for stem in alternative) for alternative in alternatives)
        for alternatives in groups
    )


def _relevant_snippet(text: str, keywords: Sequence[str]) -> str:
    if not text:
        return "—"
    content_keywords = [keyword for keyword in keywords if not _looks_like_url(keyword)]
    stems = _keyword_stems(content_keywords)
    for keyword in content_keywords:
        for alternative in _keyword_alternatives(keyword):
            stems.extend(alternative)
    folded = text.casefold()
    positions = [position for stem in stems if (position := folded.find(stem)) >= 0]
    if not positions:
        return textwrap.shorten(text, width=SNIPPET_LENGTH, placeholder="…")
    start = max(0, min(positions) - 20)
    if start:
        next_space = text.find(" ", start)
        start = next_space + 1 if next_space >= 0 else start
    fragment = text[start : start + SNIPPET_LENGTH]
    if start:
        fragment = "…" + fragment
    if start + SNIPPET_LENGTH < len(text):
        fragment = fragment.rstrip() + "…"
    return fragment


def _truncate(value: str, width: int) -> str:
    return value if len(value) <= width else value[: width - 1] + "…"


def render_table(captures: Iterable[Capture], *, show_text: bool) -> str:
    """Формирует таблицу через pandas."""
    rows: list[dict[str, str]] = []
    for capture in captures:
        row = {
            "URL": _truncate(capture.url, 48 if not show_text else 36),
            "Дата архивации": capture.archived_at,
            "Обход": capture.crawl,
            "Заголовок страницы": _truncate(capture.title, 34),
        }
        if show_text:
            row["Фрагмент текста"] = _truncate(capture.snippet or "—", 58)
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.to_string(index=False, justify="left")


def run(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = create_parser().parse_args(argv)
    try:
        candidate_limit = min(max(args.limit * 4, 12), 40)
        captures = search_cdx(
            args.keywords,
            domain=args.domain,
            limit=candidate_limit if args.show_text else args.limit,
            crawls=args.crawls,
            filter_url_keywords=not args.show_text,
        )
        if not captures:
            print(f"Ничего не найдено в последних обходах: {args.crawls}.")
            print(
                "Для поиска по тексту задайте --domain или URL и добавьте --show-text."
            )
            return 0

        if args.show_text:
            enriched: list[Capture] = []
            failures = 0
            with requests.Session() as session:
                progress = tqdm(
                    total=len(captures),
                    desc="Проверка WARC",
                    unit="стр.",
                    file=sys.stderr,
                )
                for capture in captures:
                    try:
                        page = enrich_with_text(capture, session=session)
                        if _capture_contains_keywords(page, args.keywords):
                            enriched.append(
                                replace(
                                    page,
                                    snippet=_relevant_snippet(
                                        page.content, args.keywords
                                    ),
                                    content="",
                                )
                            )
                    except CommonCrawlError:
                        failures += 1
                    finally:
                        progress.update(1)
                    if len(enriched) >= args.limit:
                        break
                progress.close()
            captures = enriched
            if not captures:
                print(
                    "Совпадений в тексте не найдено среди "
                    f"{candidate_limit} или менее приоритетных страниц."
                )
                print("Уточните домен/URL, слова или увеличьте --crawls.")
                if failures:
                    print(f"Не удалось прочитать WARC-записей: {failures}.")
                return 0

        crawl_names = sorted({capture.crawl for capture in captures}, reverse=True)
        print("Найдены данные обходов: " + ", ".join(crawl_names))
        print(render_table(captures, show_text=args.show_text))
        return 0
    except CommonCrawlError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nОперация отменена.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(run())
