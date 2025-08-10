# lise/crawler.py

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import trafilatura 

def crawl_website(base_url, must_include=None, must_exclude=None, max_pages=10):
    """
    Crawls a website, extracts only the MAIN content from each page using
    trafilatura, and returns a list of (url, main_text_content).
    """
    visited = set()
    to_visit = [base_url]
    results = []
    print(f"Starting intelligent crawl at {base_url}...")

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited or urlparse(url).netloc != urlparse(base_url).netloc:
            continue

        try:
            # It's better to let trafilatura work on the raw downloaded content
            downloaded = trafilatura.fetch_url(url)
            
            # If download fails, skip this URL
            if downloaded is None:
                print(f"-> Failed to download {url}")
                continue

            # Use trafilatura to extract the main content, stripping out menus, ads, etc.
            main_text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False, # Often tables are not useful prose
                no_fallback=True      # Do not fall back to generic extraction if main content not found
            )

            if main_text:
                results.append((url, main_text))
                # print(f"-> Successfully extracted main content from {url}")
            
            visited.add(url)

            # Use BeautifulSoup just for finding links to follow
            soup = BeautifulSoup(downloaded, "html.parser")
            for link in soup.find_all("a", href=True):
                full_url = urljoin(url, link['href'])
                # Stay on the same domain and avoid already visited URLs
                if full_url.startswith(base_url) and full_url not in visited and len(to_visit) < max_pages * 2:
                    to_visit.append(full_url)

        except Exception as e:
            print(f"-> Failed to process {url}: {e}")

    print(f"Crawl complete. Extracted main content from {len(results)} pages.")
    return results