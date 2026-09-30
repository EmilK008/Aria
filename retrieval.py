"""
retrieval.py -- key-free web/knowledge search for grounding Aria's answers.

For facts, a small model shouldn't rely on memory -- it should READ. This module
fetches real text from Wikipedia (no API key needed) so we can drop it into the
prompt and let Aria answer from it (retrieval-augmented generation).

  needs_search(msg) -> bool     # is this a factual question worth looking up?
  search(query)     -> (context_text, source_title) | (None, None)
"""

import html
import json
import os
import re
import urllib.parse
import urllib.request

UA = {"User-Agent": "AriaChatbot/1.0 (from-scratch learning project)"}
WIKI = "https://en.wikipedia.org/w/api.php"

# The grounded-answer template. CRITICAL: training (prepare_aria.py) and inference
# (chat.py / api.py) must build the user turn with this exact same function, so the
# model sees the format it was trained on.
GROUNDED_INSTRUCTION = "Answer the question using the context below."


def grounded_user_text(context, question):
    return f"{GROUNDED_INSTRUCTION}\nContext: {context}\nQuestion: {question}"


_STOP = set("a an the is are was were be been being of to in on at for and or but with "
            "as by from that this these those it its who what when where why which how "
            "do does did has have had can could will would should may might about into "
            "your you i we they he she them his her their our".split())


def _content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


def focus_context(question, context, max_chars=1100):
    """Rerank: keep the opening (definition) + the sentences most relevant to the
    question, in original order. Ensures the answer sentence is actually present
    even when it's buried deep in a long article."""
    sents = re.split(r"(?<=[.!?]) ", context)
    if len(sents) <= 3:
        return context[:max_chars]
    qwords = _content_words(question)
    if not qwords:
        return context[:max_chars]
    keep, used = {0}, len(sents[0])           # always keep the first (defining) sentence
    scored = sorted(
        ((len(_content_words(s) & qwords), i) for i, s in enumerate(sents)),
        key=lambda x: -x[0],
    )
    for score, i in scored:
        if score <= 0 or used > max_chars:
            break
        if i not in keep:
            keep.add(i)
            used += len(sents[i])
    return " ".join(sents[i] for i in sorted(keep))[:max_chars]

# things that look like questions but should NOT trigger a web lookup
# (identity, small talk, feelings -- Aria handles these herself)
# conversational / personal -> never search (greetings, feelings, identity, opinions,
# and anything addressed at Aria herself: "do/did/have you...", "your day", etc.)
_CHITCHAT = re.compile(
    r"\b(hi|hello|hey|yo|sup|thanks|thank you|bye|goodbye|good (morning|night|evening)|"
    r"your name|who are you|how are you|what'?s up|how'?s it going|how'?s your|"
    r"what do you (think|like|feel|want|do for fun)|do you (like|think|feel|want|have|remember)|"
    r"did you|have you|would you|can you help|what should we|let'?s|"
    r"made you|created you|built you|are you (a|an|real|human|ok|there)|how do you feel|"
    r"favorite|your (day|weekend|favorite|hobby|opinion)|tell me a (joke|story))\b",
    re.I,
)
# clearly a lookup: starts with a question word, or a "tell me / describe" opener
_QUESTION_WORD = re.compile(
    r"^\s*(what|what'?s|who|who'?s|when|where|why|which|whose|how)\b|"
    r"^\s*(tell me about|tell me|describe|define|explain|give me|list|name the|"
    r"look up|search for|search)\b",
    re.I,
)
# fact-ish words anywhere -> probably wants a lookup
_FACT_CUE = re.compile(
    r"\b(capital|population|born|died|invented|discovered|founded|located|distance|"
    r"height|tall|deep|wide|weigh|largest|smallest|biggest|longest|oldest|"
    r"author|wrote|painted|composed|directed|president|king|queen|located|"
    r"definition|meaning of|what year|how many|how much)\b",
    re.I,
)
_FIRST_PERSON = re.compile(r"\b(i|i'?m|im|me|my|we|our|you|your|let'?s)\b", re.I)


def needs_search(message):
    """Heuristic: factual lookup -> yes; greeting/identity/small talk -> no.

    Deliberately fairly eager -- if it over-triggers, search() returns nothing for
    irrelevant queries (the relevance gate), so we fall back to a normal chat reply.
    """
    m = message.strip()
    if len(m) < 3 or _CHITCHAT.search(m):
        return False
    if _QUESTION_WORD.match(m):
        return True
    if _FACT_CUE.search(m):
        return True
    # any question that isn't about Aria/the user (no "you/I/we") -> try a lookup
    if m.endswith("?") and not _FIRST_PERSON.search(m):
        return True
    return False


def _get(params):
    url = WIKI + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=12) as r:
        return json.loads(r.read())


# strip the question scaffolding so we search Wikipedia for the actual TOPIC,
# e.g. "What is the capital of France?" -> "capital of France"
_STRIP_LEAD = re.compile(
    r"^\s*(please\s+)?(can you\s+|could you\s+|do you know\s+)?"
    r"(tell me about|what'?s|what is|what are|what was|what were|who is|who are|"
    r"who was|who were|who|when is|when was|when did|when do|where is|where are|"
    r"where was|where did|why is|why are|why did|which is|which are|whose is|"
    r"how many|how much|how tall|how old|how long|how far|how high|how big|"
    r"how large|how|define|explain|what)\b",
    re.I,
)
# common "who/what VERBed X" verbs -- strip so we search for X, not the verb
_STRIP_VERB = re.compile(
    r"^\s*(wrote|authored|painted|invented|discovered|created|built|founded|"
    r"designed|composed|directed|produced|developed|made|named|won|said|"
    r"sang|plays?|owns?|runs?|leads?|wrote)\b",
    re.I,
)
_STRIP_FILLER = re.compile(r"^\s*(the|a|an|is|are|was|were|did|does|do|of)\b", re.I)


def _clean_query(message):
    q = message.strip().rstrip("?.! ")
    q = _STRIP_LEAD.sub("", q).strip()
    q = _STRIP_VERB.sub("", q).strip()
    # peel a couple of leading filler words (the/is/are/of...) left after stripping
    for _ in range(3):
        new = _STRIP_FILLER.sub("", q).strip()
        if new == q:
            break
        q = new
    return q or message.strip().rstrip("?.! ")


def search(query, max_chars=1600):
    """Find the most relevant Wikipedia article and return its intro text."""
    try:
        cleaned = _clean_query(query)
        qwords = _content_words(cleaned)
        # 1. search for the best-matching article titles (a few, in case some are empty)
        res = _get({"action": "query", "list": "search", "srsearch": cleaned,
                    "srlimit": 5, "format": "json"})
        hits = res.get("query", {}).get("search", [])
        if not hits:
            return None, None

        # 2. walk the hits and return the first article with a real, RELEVANT intro
        for hit in hits:
            title = hit["title"]
            res = _get({"action": "query", "prop": "extracts", "exintro": 1,
                        "explaintext": 1, "titles": title, "redirects": 1,
                        "format": "json"})
            pages = res.get("query", {}).get("pages", {})
            page = next(iter(pages.values()))
            extract = (page.get("extract") or "").strip()
            if len(extract) < 40:        # skip empty/disambiguation stubs
                continue
            extract = re.sub(r"\s+", " ", extract)
            # relevance gate: the article must share a word with the query, else the
            # trigger mis-fired -> return nothing so we fall back to a chat reply
            if qwords and not (qwords & _content_words(title + " " + extract[:200])):
                continue
            if len(extract) > max_chars:
                extract = extract[:max_chars].rsplit(".", 1)[0] + "."
            return extract, title
        return None, None
    except Exception:
        return None, None


_DDG = "https://html.duckduckgo.com/html/"
_BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120 Safari/537.36"}


def _strip(t):
    return html.unescape(re.sub(r"<[^>]+>", "", t)).strip()


_KEYS_FILE = os.path.join(os.path.dirname(__file__), "search_keys.json")


def _load_keys():
    """Search API keys from env vars, falling back to search_keys.json (user-set)."""
    keys = {"brave": os.environ.get("BRAVE_API_KEY", ""),
            "tavily": os.environ.get("TAVILY_API_KEY", "")}
    try:
        with open(_KEYS_FILE, encoding="utf-8") as f:
            fk = json.load(f)
        for k in ("brave", "tavily"):
            if not keys[k] and fk.get(k):
                keys[k] = fk[k]
    except Exception:
        pass
    return {k: v for k, v in keys.items() if v}


def _brave_search(query, key, max_chars, n):
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
        {"q": query, "count": n})
    req = urllib.request.Request(url, headers={"X-Subscription-Token": key,
                                               "Accept": "application/json"})
    data = json.loads(urllib.request.urlopen(req, timeout=12).read())
    results = data.get("web", {}).get("results", [])
    if not results:
        return None, None, None
    ctx = re.sub(r"\s+", " ", " ".join(_strip(r.get("description", "")) for r in results[:n]))
    return ctx[:max_chars], results[0].get("title", "web result"), results[0].get("url")


def _tavily_search(query, key, max_chars, n):
    body = json.dumps({"api_key": key, "query": query, "max_results": n,
                       "search_depth": "basic"}).encode()
    req = urllib.request.Request("https://api.tavily.com/search", data=body,
                                 headers={"Content-Type": "application/json"})
    data = json.loads(urllib.request.urlopen(req, timeout=15).read())
    results = data.get("results", [])
    if not results:
        return None, None, None
    ctx = re.sub(r"\s+", " ", " ".join((r.get("content") or "") for r in results[:n]))
    return ctx[:max_chars], results[0].get("title", "web result"), results[0].get("url")


def web_search(query, max_chars=900, n=3):
    """General web search. Best backend available: Brave/Tavily (if a key is set) ->
    DuckDuckGo scraping. Returns (context, title, url) or Nones."""
    keys = _load_keys()
    for name, fn in (("brave", _brave_search), ("tavily", _tavily_search)):
        if keys.get(name):
            try:
                r = fn(query, keys[name], max_chars, n)
                if r[0]:
                    return r
            except Exception:
                pass
    return _ddg_search(query, max_chars, n)


def _ddg_search(query, max_chars=900, n=3):
    """Free fallback: scrape DuckDuckGo HTML (rate-limits under heavy use)."""
    try:
        url = _DDG + "?" + urllib.parse.urlencode({"q": query})
        req = urllib.request.Request(url, headers=_BROWSER)
        h = urllib.request.urlopen(req, timeout=12).read().decode("utf-8", "replace")
        titles = [_strip(t) for t in re.findall(r'class="result__a"[^>]*>(.*?)</a>', h, re.S)]
        hrefs = re.findall(r'class="result__a"[^>]*href="([^"]+)"', h, re.S)
        snips = [_strip(s) for s in re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', h, re.S)]

        def real_url(href):
            m = re.search(r"uddg=([^&]+)", href)
            return urllib.parse.unquote(m.group(1)) if m else href

        # zip results and drop ads (those still point at a duckduckgo.com redirect)
        results = []
        for t, u, s in zip(titles, hrefs, snips):
            ru = real_url(u)
            if ru.startswith("http") and "duckduckgo.com" not in ru and s:
                results.append((t, ru, s))
        if not results:
            return None, None, None
        context = re.sub(r"\s+", " ", " ".join(r[2] for r in results[:n]))[:max_chars]
        return context, results[0][0], results[0][1]
    except Exception:
        return None, None, None


def _wiki(query):
    ctx, title = search(query)
    if not ctx:
        return None, None, None
    ctx = focus_context(query, ctx)
    url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
    return ctx, f"Wikipedia — {title}", url


def _web(query):
    ctx, title, url = web_search(query)
    if not ctx:
        return None, None, None
    domain = re.sub(r"^https?://(www\.)?", "", url or "").split("/")[0] or "web"
    return ctx, f"{title[:60]} ({domain})", url


# these queries want the live web (how-tos, current, opinion) rather than an encyclopedia
_WEB_FIRST = re.compile(
    r"\b(how (to|do|can|does)|best|top \d|cheapest|recommend|review|vs|versus|"
    r"near me|today|tonight|latest|current|recent|news|202\d|price|buy|download|"
    r"recipe|tutorial|guide|fix|error|update|release date|schedule)\b", re.I)


def retrieve(query):
    """Unified retrieval. Encyclopedic -> Wikipedia (clean text); how-to/current -> web.
    Falls back to the other source. Returns (context, source_label, source_url)."""
    order = (_web, _wiki) if _WEB_FIRST.search(query) else (_wiki, _web)
    for fn in order:
        r = fn(query)
        if r[0]:
            return r
    return None, None, None


if __name__ == "__main__":
    for q in ["What is the capital of France?", "How tall is Mount Everest?",
              "Who wrote Romeo and Juliet?", "hi how are you?"]:
        if needs_search(q):
            ctx, src = search(q)
            print(f"[SEARCH] {q}\n  source: {src}\n  {ctx[:160] if ctx else None}...\n")
        else:
            print(f"[CHAT  ] {q}  (no search)\n")
