import csv
from csv import QUOTE_ALL
import os
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from openai import OpenAI
import faiss
import numpy as np
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import xml.etree.ElementTree as ET

# --- COMMAND-LINE ARGUMENTS ---
parser = argparse.ArgumentParser(
    description="Build FAISS index from financial articles CSV."
)
parser.add_argument("-v", "--verbose", action="store_true", help="Print log messages")
parser.add_argument("-c", "--chunk_size", type=int, default=500, help="Words per chunk")
parser.add_argument("-o", "--overlap", type=int, default=125, help="Words overlap per chunk")
parser.add_argument("-t", "--threads", type=int, default=5, help="Parallel threads")
parser.add_argument("-b", "--batch_size", type=int, default=20, help="Chunks per embedding batch")
parser.add_argument("-m", "--model", type=str, default="text-embedding-3-small", help="Embedding model to use")
parser.add_argument("-a", "--articles_csv", type=str, default="articles.csv", help="Path to articles CSV file")
parser.add_argument("-i", "--index_file", type=str, default="financial_articles.index", help="Output FAISS index file")
parser.add_argument("-md", "--metadata_file", type=str, default="financial_articles_metadata.json", help="Output index metadata json file")
parser.add_argument("-l", "--log_file", type=str, default="fetch.log", help="Log file name")
parser.add_argument("-f", "--failure_csv", type=str, default="failures.csv", help="Failed Fetch CSV file")
parser.add_argument("-s", "--success_csv", type=str, default="successes.csv", help="Successful Fetch CSV file")
args = parser.parse_args()

VERBOSE = args.verbose
CHUNK_SIZE = args.chunk_size
OVERLAP = args.overlap
MAX_THREADS = args.threads
BATCH_SIZE = args.batch_size

# --- PATH CONFIG ---
CSV_FILE = os.path.join("articles", args.articles_csv)
OUTPUT_DIR = os.path.join("..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


FAISS_INDEX_FILE = os.path.join(OUTPUT_DIR, args.index_file)
METADATA_FILE = os.path.join(OUTPUT_DIR, args.metadata_file)

LOG_SUBDIR = "logs"
os.makedirs(LOG_SUBDIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_SUBDIR, args.log_file)
FAILURE_CSV = os.path.join(LOG_SUBDIR, args.failure_csv)
SUCCESS_CSV = os.path.join(LOG_SUBDIR, args.success_csv)

EMBEDDING_MODEL = args.model
MAX_RETRIES = 3
RETRY_DELAY = 5
GET_TIMEOUT = 60
EMBEDDING_TIMEOUT = 60
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36"
}

client = OpenAI()  # assumes OPENAI_API_KEY is set in env
lock = threading.Lock()


# --- LOGGING ---
def log(message):
    with lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")
        if VERBOSE:
            print(message)


# --- SAVE URLS ---
def save_urls(urls, filename):
    """Save list of dicts (articles) to CSV. Handles empty list."""
    if not urls:
        log(f"No rows to save for {filename}")
        return
    # Ensure urls is a list of dicts
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=urls[0].keys(), quoting=QUOTE_ALL)
        writer.writeheader()
        writer.writerows(urls)
    log(f"Saved {len(urls)} rows to {filename}")


# --- TEXT EXTRACTION ---
def is_mediawiki_page(soup):
    return bool(soup.select_one("body.mediawiki"))


def clean_mediawiki(soup: BeautifulSoup) -> str:
    """Cleans a MediaWiki page HTML."""
    for selector in [
        ".navbox", ".metadata", ".reflist", ".infobox",
        ".hatnote", ".mw-editsection", "#footer", ".sidebar", ".toc"
    ]:
        for elem in soup.select(selector):
            elem.decompose()
    main_content = soup.select_one("#content") or soup.select_one("#mw-content-text") or soup.body
    if not main_content:
        main_content = soup.body
    text = main_content.get_text(" ", strip=True)
    return " ".join(text.split())


def extract_text_from_pdf(path_or_url):
    """Extract text from local PDF or download if URL."""
    if os.path.exists(path_or_url):
        reader = PdfReader(path_or_url)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    else:
        # download
        r = requests.get(path_or_url, headers=HEADERS, timeout=GET_TIMEOUT)
        r.raise_for_status()
        with open("temp.pdf", "wb") as f:
            f.write(r.content)
        reader = PdfReader("temp.pdf")
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        os.remove("temp.pdf")
        return text


def extract_text_from_html(path_or_url):
    if os.path.exists(path_or_url):
        with open(path_or_url, "r", encoding="utf-8") as f:
            html_content = f.read()
    else:
        r = requests.get(path_or_url, headers=HEADERS, timeout=GET_TIMEOUT)
        r.raise_for_status()
        html_content = r.text

    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()
    if is_mediawiki_page(soup):
        return clean_mediawiki(soup)
    text = "\n".join([line.strip() for line in soup.get_text().splitlines() if line.strip()])
    return text

def extract_text_from_mediawiki_xml(file_path):
    """
    Parses a MediaWiki XML dump and returns a list of dicts:
    [{"title": page_title, "text": page_text}, ...]
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}  # MediaWiki XML namespace
        pages_list = []

        for page in root.findall("mw:page", ns):
            title_elem = page.find("mw:title", ns)
            revision = page.find("mw:revision", ns)
            if title_elem is None or revision is None:
                continue
            text_elem = revision.find("mw:text", ns)
            if text_elem is None or not text_elem.text:
                continue
            page_text = " ".join(text_elem.text.split())
            pages_list.append({
                "title": title_elem.text,
                "text": page_text
            })

        return pages_list

    except Exception as e:
        log(f"Failed to parse MediaWiki XML {file_path}: {e}")
        return []


# --- CHUNKING ---
def chunk_text_with_overlap(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += (chunk_size - overlap)
    return chunks


# --- EMBEDDING ---
def embed_texts(texts):
    embeddings = []
    for i in range(0, len(texts), 10):
        batch = texts[i:i+10]
        try:
            resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL, timeout=EMBEDDING_TIMEOUT)
            embeddings.extend([e.embedding for e in resp.data])
        except Exception as e:
            log(f"Batch embedding error: {e}")
    return embeddings


# --- FETCH SINGLE ARTICLE ---
def fetch_article(row):
    url_or_path = row["url"]
    title = row["title"]
    category = row["primary_category"]
    note = row["notes"]

    try:
        chunks_result = []

        # Handle PDFs
        if url_or_path.lower().endswith(".pdf"):
            text = extract_text_from_pdf(url_or_path)
            chunks = chunk_text_with_overlap(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
            for idx, chunk in enumerate(chunks):
                chunks_result.append({
                    "chunk": chunk,
                    "metadata": {
                        "title": title,
                        "url": url_or_path,
                        "category": category,
                        "note": note,
                        "chunk_index": idx
                    }
                })
            log(f"SUCCESS: PDF '{url_or_path}' processed with {len(chunks)} chunks")

        # Handle MediaWiki XML
        elif url_or_path.lower().endswith(".xml"):
            pages_text = extract_text_from_mediawiki_xml(url_or_path)
            if not pages_text:
                raise ValueError(f"No pages extracted from XML: {url_or_path}")
            
            for page in pages_text:
                page_text = page["text"]
                page_title = page["title"]
                chunks = chunk_text_with_overlap(page_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
                for idx, chunk in enumerate(chunks):
                    chunks_result.append({
                        "chunk": chunk,
                        "metadata": {
                            "title": page_title,
                            "url": url_or_path,
                            "category": category,
                            "note": note,
                            "chunk_index": idx
                        }
                    })
                log(f"SUCCESS: XML page '{page_title}' processed with {len(chunks)} chunks")

        # Handle HTML/web pages
        else:
            text = extract_text_from_html(url_or_path)
            chunks = chunk_text_with_overlap(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
            for idx, chunk in enumerate(chunks):
                chunks_result.append({
                    "chunk": chunk,
                    "metadata": {
                        "title": title,
                        "url": url_or_path,
                        "category": category,
                        "note": note,
                        "chunk_index": idx
                    }
                })
            log(f"SUCCESS: URL '{url_or_path}' processed with {len(chunks)} chunks")

        return chunks_result, None

    except Exception as e:
        log(f"FAIL: {url_or_path} - {e}")
        return None, row

# --- PARALLEL FETCH ---
def fetch_articles_parallel(articles, success_urls=[]):
    documents = []
    metadata = []
    failed_urls = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_row = {executor.submit(fetch_article, row): row for row in articles}
        for future in as_completed(future_to_row):
            chunks_result, failed_row = future.result()
            if chunks_result:
                success_urls.append(future_to_row[future])
                for r in chunks_result:
                    documents.append(r["chunk"])
                    metadata.append(r["metadata"])
            if failed_row:
                failed_urls.append(failed_row)

    return documents, metadata, failed_urls, success_urls


# --- MAIN ---
with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    articles = list(reader)

log(f"Using CHUNK_SIZE={CHUNK_SIZE}, OVERLAP={OVERLAP}, VERBOSE={VERBOSE}, THREADS={MAX_THREADS}, BATCH_SIZE={BATCH_SIZE}")

# First pass
documents, metadata, failed_urls, success_urls = fetch_articles_parallel(articles)

# Retry logic
for attempt in range(1, MAX_RETRIES + 1):
    if not failed_urls:
        break
    log(f"--- Retry attempt {attempt} for {len(failed_urls)} failed URLs ---")
    time.sleep(RETRY_DELAY)
    documents_retry, metadata_retry, failed_urls_retry, success_urls = fetch_articles_parallel(failed_urls, success_urls)
    documents.extend(documents_retry)
    metadata.extend(metadata_retry)
    failed_urls = failed_urls_retry

save_urls(failed_urls, FAILURE_CSV)
save_urls(success_urls, SUCCESS_CSV)

log(f"Total documents/chunks: {len(documents)}")
log(f"Total failed fetches after retries: {len(failed_urls)}")
log(f"Total successful fetches after retries: {len(success_urls)}")

# --- INCREMENTAL EMBEDDING AND FAISS BUILD ---
dimension = None
index = None
all_metadata = []

log("Starting incremental embedding and FAISS index build...")

for batch_start in range(0, len(documents), BATCH_SIZE):
    batch_chunks = documents[batch_start:batch_start + BATCH_SIZE]
    batch_metadata = metadata[batch_start:batch_start + BATCH_SIZE]

    batch_embeddings = embed_texts(batch_chunks)
    batch_embeddings = np.array(batch_embeddings).astype("float32")

    if index is None:
        dimension = batch_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)

    index.add(batch_embeddings)
    all_metadata.extend(batch_metadata)
    log(f"Processed batch {batch_start // BATCH_SIZE + 1} / {(len(documents) + BATCH_SIZE - 1) // BATCH_SIZE}")
    

# --- SAVE INDEX AND METADATA ---
for global_index, meta_item in enumerate(all_metadata):
    # This line OVERWRITES the local, repeating 'chunk_index' 
    # with the correct global, unique index ID (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...)
    meta_item["chunk_index"] = global_index
    
faiss.write_index(index, FAISS_INDEX_FILE)
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)

log(f"FAISS index saved to {FAISS_INDEX_FILE}")
log(f"Metadata saved to {METADATA_FILE}")
