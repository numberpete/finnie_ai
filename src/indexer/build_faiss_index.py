import csv
from csv import QUOTE_ALL
import os
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from openai import OpenAI
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import xml.etree.ElementTree as ET
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- COMMAND-LINE ARGUMENTS ---
parser = argparse.ArgumentParser(
    description="Build FAISS index from financial articles CSV."
)
parser.add_argument("-v", "--verbose", action="store_true", help="Print log messages")
parser.add_argument("-c", "--chunk_size", type=int, default=500, help="Words per chunk")
parser.add_argument("-o", "--overlap", type=int, default=125, help="Words overlap per chunk")
parser.add_argument("-t", "--threads", type=int, default=5, help="Parallel threads")
parser.add_argument("-b", "--batch_size", type=int, default=100, help="Documents per batch for FAISS")
parser.add_argument("-m", "--model", type=str, default="text-embedding-3-small", help="Embedding model to use")
parser.add_argument("-a", "--articles_csv", type=str, default="articles.csv", help="Path to articles CSV file")
parser.add_argument("-i", "--index_dir", type=str, default="financial_articles", help="Output FAISS index directory")
parser.add_argument("-l", "--log_file", type=str, default="fetch.log", help="Log file name")
parser.add_argument("-f", "--failure_csv", type=str, default="failures.csv", help="Failed Fetch CSV file")
parser.add_argument("-s", "--success_csv", type=str, default="successes.csv", help="Successful Fetch CSV file")
args = parser.parse_args()

VERBOSE = args.verbose
CHUNK_SIZE = args.chunk_size
OVERLAP = args.overlap
MAX_THREADS = args.threads
BATCH_SIZE = args.batch_size
EMBEDDING_MODEL = args.model

# --- PATH CONFIG ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_SCRIPT_DIR))
CSV_FILE = os.path.join(CURRENT_SCRIPT_DIR, "articles", args.articles_csv)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FAISS_INDEX_DIR = os.path.join(OUTPUT_DIR, args.index_dir)

LOG_SUBDIR = os.path.join(CURRENT_SCRIPT_DIR, "logs")
os.makedirs(LOG_SUBDIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_SUBDIR, args.log_file)
FAILURE_CSV = os.path.join(LOG_SUBDIR, args.failure_csv)
SUCCESS_CSV = os.path.join(LOG_SUBDIR, args.success_csv)

MAX_RETRIES = 3
RETRY_DELAY = 5
GET_TIMEOUT = 60
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36",
    # -------------------------------------------------------------
    # NEW LINE: Request the server only use Gzip or Deflate,
    # thereby avoiding the problematic 'br' (Brotli) encoding.
    "Accept-Encoding": "gzip, deflate"
    # -------------------------------------------------------------
}

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
        ns = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}
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


# --- FETCH SINGLE ARTICLE (returns LangChain Documents) ---
def fetch_article(row):
    url_or_path = row["url"]
    page_url = row.get("page_url", url_or_path)
    title = row["title"]
    category = row["primary_category"]
    note = row["notes"]

    try:
        documents = []

        # Handle PDFs
        if url_or_path.lower().endswith(".pdf"):
            text = extract_text_from_pdf(url_or_path)
            chunks = chunk_text_with_overlap(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "title": title,
                        "url": page_url,
                        "category": category,
                        "note": note,
                        "chunk_index": idx
                    }
                )
                documents.append(doc)
            log(f"SUCCESS: PDF '{url_or_path}' processed with {len(chunks)} chunks")

        # Handle MediaWiki XML
        elif url_or_path.lower().endswith(".xml"):
            pages_text = extract_text_from_mediawiki_xml(url_or_path)
            if not pages_text:
                raise ValueError(f"No pages extracted from XML: {url_or_path}")
            
            for page in pages_text:
                page_text = page["text"]
                page_title = page["title"]
                page_url ="http://bogleheads.org/wiki/" + page_title.replace(" ", "_")
                chunks = chunk_text_with_overlap(page_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
                for idx, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "title": page_title,
                            "url": page_url,
                            "category": category,
                            "note": note,
                            "chunk_index": idx
                        }
                    )
                    documents.append(doc)
                log(f"SUCCESS: XML page '{page_title}' processed with {len(chunks)} chunks")

        # Handle HTML/web pages
        else:
            text = extract_text_from_html(url_or_path)
            chunks = chunk_text_with_overlap(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "title": title,
                        "url": page_url,
                        "category": category,
                        "note": note,
                        "chunk_index": idx
                    }
                )
                documents.append(doc)
            log(f"SUCCESS: URL '{url_or_path}' processed with {len(chunks)} chunks")

        return documents, None

    except Exception as e:
        log(f"FAIL: {url_or_path} - {e}")
        return None, row


# --- PARALLEL FETCH ---
def fetch_articles_parallel(articles, success_urls=[]):
    all_documents = []
    failed_urls = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_row = {executor.submit(fetch_article, row): row for row in articles}
        for future in as_completed(future_to_row):
            documents, failed_row = future.result()
            if documents:
                success_urls.append(future_to_row[future])
                all_documents.extend(documents)
            if failed_row:
                failed_urls.append(failed_row)

    return all_documents, failed_urls, success_urls


# --- MAIN ---
with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    articles = list(reader)

log(f"Using CHUNK_SIZE={CHUNK_SIZE}, OVERLAP={OVERLAP}, VERBOSE={VERBOSE}, THREADS={MAX_THREADS}, BATCH_SIZE={BATCH_SIZE}")

# First pass
all_documents, failed_urls, success_urls = fetch_articles_parallel(articles)

# Retry logic
for attempt in range(1, MAX_RETRIES + 1):
    if not failed_urls:
        break
    log(f"--- Retry attempt {attempt} for {len(failed_urls)} failed URLs ---")
    time.sleep(RETRY_DELAY)
    documents_retry, failed_urls_retry, success_urls = fetch_articles_parallel(failed_urls, success_urls)
    all_documents.extend(documents_retry)
    failed_urls = failed_urls_retry

save_urls(failed_urls, FAILURE_CSV)
save_urls(success_urls, SUCCESS_CSV)

log(f"Total documents/chunks: {len(all_documents)}")
log(f"Total failed fetches after retries: {len(failed_urls)}")
log(f"Total successful fetches after retries: {len(success_urls)}")

# --- BUILD FAISS INDEX WITH LANGCHAIN ---
log("Starting FAISS index build with LangChain...")

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL,max_retries=3)

log(f"Using embedding model: {embeddings.model}")
log(f"Embedding dimension: {len(embeddings.embed_query('test'))}")


# Build index in batches to manage memory
if len(all_documents) <= BATCH_SIZE:
    # Small enough to do in one go
    vector_store = FAISS.from_documents(all_documents, embeddings)
    log(f"Built FAISS index with {len(all_documents)} documents")
else:
    # Build incrementally
    log(f"Building index incrementally with batch size {BATCH_SIZE}")
    vector_store = FAISS.from_documents(all_documents[:BATCH_SIZE], embeddings)
    log(f"Initial batch: {BATCH_SIZE} documents")
    
    for i in range(BATCH_SIZE, len(all_documents), BATCH_SIZE):
        batch = all_documents[i:i+BATCH_SIZE]
        batch_store = FAISS.from_documents(batch, embeddings)
        vector_store.merge_from(batch_store)
        log(f"Processed batch {i//BATCH_SIZE + 1} / {(len(all_documents) + BATCH_SIZE - 1) // BATCH_SIZE}")
        time.sleep(1)  # Wait 1 second between batches to avoid reate limits

# --- SAVE INDEX ---
vector_store.save_local(
    FAISS_INDEX_DIR
)
log(f"FAISS index saved to {FAISS_INDEX_DIR}")
