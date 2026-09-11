import io
import unittest

from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

import common_crawl as cc


class FakeFetcher:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def get(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.rows


class FakeResponse:
    def __init__(self, content, status_code=206):
        self.content = content
        self.status_code = status_code


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.headers = None

    def get(self, _url, *, headers, timeout):
        self.headers = headers
        self.timeout = timeout
        return self.response


def make_warc(html: bytes) -> bytes:
    output = io.BytesIO()
    writer = WARCWriter(output, gzip=True)
    http_headers = StatusAndHeaders(
        "200 OK",
        [("Content-Type", "text/html; charset=utf-8")],
        protocol="HTTP/1.1",
    )
    record = writer.create_warc_record(
        "https://example.org",
        "response",
        payload=io.BytesIO(html),
        http_headers=http_headers,
    )
    writer.write_record(record)
    record.raw_stream.close()
    return output.getvalue()


class CommonCrawlTests(unittest.TestCase):
    def test_cli_defaults(self):
        args = cc.create_parser().parse_args(["example.org"])
        self.assertEqual(args.limit, 10)
        self.assertEqual(args.crawls, 3)
        self.assertFalse(args.show_text)

    def test_search_uses_high_level_cdx_fetcher(self):
        row = {
            "url": "https://pstu.ru/news/perm-polytech",
            "timestamp": "20250801123045",
            "filename": "crawl-data/CC-MAIN-2025-30/x/file.warc.gz",
            "offset": "10",
            "length": "20",
        }
        fetcher = FakeFetcher([row])
        captures = cc.search_cdx(
            ["perm", "polytech"],
            domain="pstu.ru",
            limit=10,
            fetcher=fetcher,
        )
        self.assertEqual([item.url for item in captures], [row["url"]])
        _, kwargs = fetcher.calls[0]
        self.assertEqual(kwargs["matchType"], "domain")
        self.assertIn("=status:200", kwargs["filter"])

    def test_warcio_and_beautifulsoup_extract_html(self):
        warc = make_warc(
            b"<html><head><title>Perm Polytech</title><style>hidden</style></head>"
            b"<body>Hello <b>archive</b><script>hidden</script></body></html>"
        )
        title, text = cc.parse_warc_record(warc)
        self.assertEqual(title, "Perm Polytech")
        self.assertEqual(text, "Perm Polytech Hello archive")

    def test_range_header_uses_offset_and_length(self):
        warc = make_warc(b"<title>T</title><p>Archive text</p>")
        capture = cc.Capture(
            url="https://example.org",
            timestamp="20250801123045",
            filename="crawl-data/CC-MAIN-2025-30/x/file.warc.gz",
            offset=100,
            length=len(warc),
        )
        session = FakeSession(FakeResponse(warc))
        result = cc.enrich_with_text(capture, session=session)
        self.assertEqual(session.headers["Range"], f"bytes=100-{99 + len(warc)}")
        self.assertEqual(result.title, "T")
        self.assertIn("Archive text", result.snippet)

    def test_full_text_matches_russian_word_forms(self):
        capture = cc.Capture(
            url="https://example.org",
            timestamp="20250801123045",
            filename="crawl-data/CC-MAIN-2025-30/x/file.warc.gz",
            offset=0,
            length=1,
            content="История Перми и Пермского политехнического университета",
        )
        self.assertTrue(
            cc._capture_contains_keywords(capture, ["Пермь", "Пермский Политех"])
        )

    def test_university_aliases(self):
        msu = cc.Capture(
            url="https://msu.ru",
            timestamp="20250801123045",
            filename="file.warc.gz",
            offset=0,
            length=1,
            title="Московский государственный университет",
        )
        self.assertTrue(cc._capture_contains_keywords(msu, ["МГУ"]))

    def test_url_scopes_cdx_but_is_not_required_in_text(self):
        capture = cc.Capture(
            url="https://teatr.example/doktor-zhivago",
            timestamp="20250801123045",
            filename="file.warc.gz",
            offset=0,
            length=1,
            content="Борис Пастернак бывал в Перми.",
        )
        keywords = ["teatr.example/doktor-zhivago", "Борис Пастернак", "Пермь"]
        self.assertTrue(cc._capture_contains_keywords(capture, keywords))
        self.assertEqual(
            cc._query_specs(keywords, None, content_search=True),
            [("teatr.example/doktor-zhivago", "prefix", None)],
        )

    def test_relevant_snippet_starts_near_keyword(self):
        text = "Навигация " * 50 + "Кафедра ИТАС проводит набор студентов."
        snippet = cc._relevant_snippet(text, ["ИТАС"])
        self.assertIn("ИТАС", snippet[:60])

    def test_pandas_table_contains_required_columns(self):
        capture = cc.Capture(
            url="https://example.org",
            timestamp="20250801123045",
            filename="crawl-data/CC-MAIN-2025-30/x/file.warc.gz",
            offset=0,
            length=1,
        )
        table = cc.render_table([capture], show_text=False)
        self.assertIn("URL", table)
        self.assertIn("Дата архивации", table)
        self.assertIn("Заголовок страницы", table)


if __name__ == "__main__":
    unittest.main()
