import os
import httpx
from urllib.parse import urljoin
from bs4 import BeautifulSoup

UA = os.getenv("USER_AGENT","SemanticScraper/1.1")
MAX_HTML_CHARS = int(os.getenv("MAX_HTML_CHARS","160000"))

def http_get(url: str, timeout: float = 20.0) -> str:
    headers = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        r = client.get(url)
        r.raise_for_status()
        html = r.text
        if len(html) > MAX_HTML_CHARS:
            html = html[:MAX_HTML_CHARS]
        return html

def absolutize(base_url: str, maybe_link: str) -> str:
    if not maybe_link:
        return base_url
    return urljoin(base_url, maybe_link)

def strip_boilerplate(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","noscript"]):
        tag.decompose()
    return str(soup)
