import socket
import ssl
import time
import sqlite3
import pickle
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import urllib.robotparser

DB_FILE = "wikipedia_spider.sqlite"
UNVISITED_FILE = "unvisited.pkl"

contactInfo=input("Contact Info: ")

# Create SQLite database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS pages (
            url TEXT PRIMARY KEY,
            title TEXT,
            content TEXT
        )
    ''')
    conn.commit()
    return conn

def is_allowed(url, user_agent="MyWikipediaSpider/1.0 (your-email@example.com)"):
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url("https://en.wikipedia.org/robots.txt")
    rp.read()
    return rp.can_fetch(user_agent, url)

def get_page(url):
    print(f"[+] Fetching: {url}")
    parsed = urlparse(url)
    host = parsed.netloc or "en.wikipedia.org"
    path = parsed.path or "/wiki/Main_Page"

    sock = socket.create_connection((host, 443))
    context = ssl.create_default_context()
    ssock = context.wrap_socket(sock, server_hostname=host)

    request = f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: MyWikipediaSpider/1.0 {contactInfo}\r\nConnection: close\r\n\r\n"
    ssock.sendall(request.encode())

    response = b""
    while True:
        data = ssock.recv(4096)
        if not data:
            break
        response += data
    ssock.close()

    headers, body = response.split(b'\r\n\r\n', 1)
    header_lines = headers.split(b'\r\n')
    status_line = header_lines[0].decode(errors='ignore')
    status_code = int(status_line.split()[1])

    if status_code == 301 or status_code == 302:
        print(f"[!] Redirected — skipping for now: {url}")
        return ""
    elif status_code != 200:
        print(f"[!] Non-200 response ({status_code}) — skipping: {url}")
        return ""

    return body.decode(errors='ignore')


def parse_and_store(url, html, conn):
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1", {"id": "firstHeading"})
    content_div = soup.find("div", {"id": "mw-content-text"})

    if not title_tag or not content_div:
        return []

    title = title_tag.get_text(strip=True)
    content = content_div.get_text(separator="\n", strip=True)

    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO pages (url, title, content) VALUES (?, ?, ?)", (url, title, content))
        conn.commit()
    except sqlite3.IntegrityError:
        pass

    links = []
    for a in content_div.find_all("a", href=True):
        href = a['href']
        if href.startswith("/wiki/") and not any(prefix in href for prefix in [":", "#"]):
            full_url = urljoin("https://en.wikipedia.org", href)
            links.append(full_url)
    return set(links)

def load_unvisited():
    if os.path.exists(UNVISITED_FILE):
        with open(UNVISITED_FILE, "rb") as f:
            return pickle.load(f)
    return set()

def save_unvisited(unvisited):
    with open(UNVISITED_FILE, "wb") as f:
        pickle.dump(unvisited, f)

def crawl(start_url, n_pages, resume=False):
    conn = init_db()
    visited = set()
    unvisited = load_unvisited() if resume else set()
    if not resume:
        unvisited.add(start_url)

    while unvisited and len(visited) < n_pages:
        url = unvisited.pop()
        if url in visited:
            continue
        if not is_allowed(url):
            print(f"[!] Skipping disallowed URL: {url}")
            continue
        try:
            html = get_page(url)
            new_links = parse_and_store(url, html, conn)
            visited.add(url)
            unvisited.update(link for link in new_links if link not in visited)
            save_unvisited(unvisited)
            time.sleep(2)
        except Exception as e:
            print(f"[!] Error processing {url}: {e}")

    print(f"[*] Finished crawling {len(visited)} pages.")
    save_unvisited(unvisited)
    conn.close()

if __name__ == "__main__":
    resume_choice = input("Resume from previous session? (y/n): ").strip().lower()
    resume = resume_choice == "y"
    if not resume:
        start_url = input("Enter start URL (e.g., https://en.wikipedia.org/wiki/Web_scraping): ").strip()
    else:
        start_url = None
    n = int(input("How many pages to crawl this session? "))
    crawl(start_url, n, resume=resume)
