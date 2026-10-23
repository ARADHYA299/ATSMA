"""Optional live collection (RSS + polite HTML text extraction). Core app works without it.
Usage: python news_collector.py <rss_url> [<rss_url> ...]   -> data/live_articles.json"""
import hashlib, json, sys, time
import urllib.robotparser as robotparser
from urllib.parse import urlparse
import feedparser, requests
from bs4 import BeautifulSoup
from text_preprocessing import ROOT, COLUMNS

HEADERS = {"User-Agent": "college-nlp-lab/1.0 (educational project)"}


def allowed_by_robots(url) -> bool:
    """Conservative: if robots.txt cannot be read we do NOT fetch."""
    try:
        p = urlparse(url)
        rp = robotparser.RobotFileParser(f"{p.scheme}://{p.netloc}/robots.txt")
        rp.read()
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return False


def fetch_article_text(url):
    """Return (text, status). No paywall/login/anti-bot bypassing."""
    if not allowed_by_robots(url):
        return "", "blocked_or_unreachable"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        text = " ".join(p.get_text(" ") for p in BeautifulSoup(r.text, "html.parser").find_all("p"))
        return text, "ok" if len(text) > 200 else "text_unavailable"
    except requests.RequestException as e:
        return "", f"error: {type(e).__name__}"


def fetch_rss(feed_url, limit=10, with_text=False):
    rows = []
    try:
        feed = feedparser.parse(feed_url)
    except Exception:
        return rows
    for e in feed.entries[:limit]:
        url = e.get("link", "")
        text, status = (e.get("summary", ""), "summary_only")
        if with_text and url:
            time.sleep(1)                                   # rate limit
            text, status = fetch_article_text(url)
        rows.append({"article_id": "L" + hashlib.md5(url.encode()).hexdigest()[:8], "title": e.get("title", ""),
                     "source": feed.feed.get("title", feed_url), "url": url,
                     "publication_date": e.get("published", ""), "text": text, "collection_status": status})
    return rows


def collect(feed_urls, out=ROOT / "data" / "live_articles.json", **kw):
    seen, rows = set(), []
    for f in feed_urls:
        for r in fetch_rss(f, **kw):
            if r["url"] and r["url"] not in seen:           # duplicate URLs dropped
                seen.add(r["url"]); rows.append({c: r.get(c, "") for c in COLUMNS})
    out.write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    print(f"saved {len(collect(sys.argv[1:], with_text=True))} articles")
