# crawler.py

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def crawl_website(base_url, max_pages=10):
    """Crawls a website and returns a list of (url, text content)"""
    visited = set()
    to_visit = [base_url]
    results = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited or urlparse(url).netloc != urlparse(base_url).netloc:
            continue

        try:
            resp = requests.get(url, timeout=5)
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            results.append((url, text))
            visited.add(url)

            for link in soup.find_all("a", href=True):
                full_url = urljoin(url, link['href'])
                if full_url.startswith(base_url) and full_url not in visited:
                    to_visit.append(full_url)

        except Exception as e:
            print(f"Failed to fetch {url}: {e}")

    return results