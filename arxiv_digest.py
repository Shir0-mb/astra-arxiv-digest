"""
arXiv SST Weekly Digest
Cerca i paper degli ultimi 7 giorni su arXiv con keyword SST/SSA,
produce un digest strutturato via Groq API e lo invia via Telegram bot.
"""

import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_API_KEY       = os.environ["GROQ_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"

KEYWORDS = [
    "space surveillance tracking",
    "space situational awareness",
    "LEO optical",
    "space debris tracking",
    "space surveillance",
    "satellite tracking",
    "resident space objects",
]

MAX_RESULTS_PER_QUERY = 15   # risultati arXiv per keyword
MAX_PAPERS_TO_LLM     = 30   # cap totale mandato a Groq


# ── arXiv fetch ───────────────────────────────────────────────────────────────

def build_arxiv_query(keyword: str) -> str:
    """Costruisce query arXiv API v1 per titolo+abstract."""
    kw = keyword.replace(" ", "+AND+")
    return (
        f"http://export.arxiv.org/api/query?"
        f"search_query=ti:{kw}+OR+abs:{kw}"
        f"&sortBy=submittedDate&sortOrder=descending"
        f"&max_results={MAX_RESULTS_PER_QUERY}"
    )


def fetch_arxiv(keyword: str) -> list[dict]:
    """Scarica e parsa i paper arXiv per una keyword. Ritorna lista di dict."""
    url = build_arxiv_query(keyword)
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[WARN] arXiv request failed for '{keyword}': {e}")
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    papers = []

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)

    for entry in root.findall("atom:entry", ns):
        # Data pubblicazione
        published_str = entry.findtext("atom:published", default="", namespaces=ns)
        try:
            published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if published < cutoff:
            continue

        arxiv_id = entry.findtext("atom:id", default="", namespaces=ns).strip()
        title    = entry.findtext("atom:title", default="").strip().replace("\n", " ")
        abstract = entry.findtext("atom:summary", default="").strip().replace("\n", " ")
        authors  = [
            a.findtext("atom:name", namespaces=ns, default="")
            for a in entry.findall("atom:author", ns)
        ]
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."

        papers.append({
            "id":       arxiv_id,
            "title":    title,
            "authors":  author_str,
            "abstract": abstract[:600],   # tronca per non sprecare token
            "url":      arxiv_id,
        })

    return papers


def collect_papers() -> list[dict]:
    """Raccoglie paper da tutte le keyword, deduplica per arXiv ID."""
    seen_ids = set()
    all_papers = []

    for kw in KEYWORDS:
        print(f"[INFO] Fetching arXiv: '{kw}'")
        papers = fetch_arxiv(kw)
        for p in papers:
            if p["id"] not in seen_ids:
                seen_ids.add(p["id"])
                all_papers.append(p)

    print(f"[INFO] Unique papers found: {len(all_papers)}")
    return all_papers[:MAX_PAPERS_TO_LLM]


# ── Groq digest ───────────────────────────────────────────────────────────────

def build_prompt(papers: list[dict]) -> str:
    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(
            f"{i}. TITLE: {p['title']}\n"
            f"   AUTHORS: {p['authors']}\n"
            f"   URL: {p['url']}\n"
            f"   ABSTRACT: {p['abstract']}\n"
        )
    papers_block = "\n".join(lines)

    return f"""You are a Space Surveillance and Tracking (SST) research assistant.
Below are {len(papers)} arXiv papers published in the last 7 days, retrieved with SST/SSA-related keywords.

Your task:
1. Group the papers into thematic categories (e.g. Optical Observations, Orbit Determination, Space Debris, Radar/RF, Conjunction Analysis, Machine Learning for SST, Other).
2. For each paper write a single concise bullet point (~60 words) summarizing the key contribution, followed by the URL.
3. Skip papers that are clearly off-topic for SST/SSA.
4. End with a short "Week Highlights" section (2-3 sentences) summarizing the most notable trends.

Format the output in clean Markdown. Write in English.

PAPERS:
{papers_block}"""


def call_groq(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":      GROQ_MODEL,
        "max_tokens": 2000,
        "messages":   [{"role": "user", "content": prompt}],
    }
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Telegram send ─────────────────────────────────────────────────────────────

def send_telegram(text: str) -> None:
    """Invia un messaggio Telegram, spezzando se > 4096 caratteri."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunk_size = 4000

    # Spezza il testo in chunk rispettando i paragrafi
    chunks = []
    while len(text) > chunk_size:
        split_at = text.rfind("\n", 0, chunk_size)
        if split_at == -1:
            split_at = chunk_size
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    chunks.append(text)

    for i, chunk in enumerate(chunks):
        prefix = f"📡 *arXiv SST Weekly Digest*\n_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}_\n\n" if i == 0 else ""
        payload = {
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       prefix + chunk,
            "parse_mode": "Markdown",
        }
        resp = requests.post(url, json=payload, timeout=20)
        if not resp.ok:
            print(f"[WARN] Telegram send failed: {resp.text}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    papers = collect_papers()

    if not papers:
        send_telegram("📡 *arXiv SST Weekly Digest*\n\nNessun paper trovato negli ultimi 7 giorni per le keyword SST/SSA configurate.")
        return

    print(f"[INFO] Building digest for {len(papers)} papers via Groq...")
    prompt = build_prompt(papers)
    digest = call_groq(prompt)

    send_telegram(digest)
    print("[INFO] Digest sent.")


if __name__ == "__main__":
    main()
