"""Query-language router for the bilingual NL->DSL prompt stacks.

The package ships two independently hardened translation paths:

  en  ->  nl2sl_hybrid_en.py  (English prompts / few-shots / reflect turns)
  zh  ->  nl2sl_hybrid_zh.py  (Chinese prompts / few-shots / reflect turns)

The routing signal is the LANGUAGE OF THE QUERY TEXT itself (per user
decision: instruction language is neutral, so each query gets the prompt
stack written in its own language).  Detection is a CJK-character ratio over
the letters of ``nature_language``:

  ratio >= _CJK_RATIO_THRESHOLD          -> zh
  any letters but ratio below threshold  -> en
  no letters at all (empty / degenerate) -> the runner's lang, then default

The runner's ``--lang`` is honoured only as the fallback for degenerate
text: it names the ENVIRONMENT (database) language, which normally matches
the query language, but when the two disagree the prompts must follow the
query text, not the database.
"""
import re

# CJK Unified Ideographs (+ Extension A) and the CJK punctuation / fullwidth
# blocks commonly present in Chinese travel queries.
_CJK_RE = re.compile(u"[㐀-䶿一-鿿豈-﫿]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# One CJK character carries roughly a word of signal, so even a modest share
# marks a Chinese query; English queries citing a Chinese POI name or ticket
# term (e.g. one '机票' span) stay far below this share of total letters.
_CJK_RATIO_THRESHOLD = 0.20


def detect_query_lang(nature_language, runner_lang=None, default="en"):
    """'zh' or 'en' for a query's nature_language (see module docstring)."""
    text = nature_language if isinstance(nature_language, str) else ""
    cjk = len(_CJK_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    total = cjk + latin
    if total == 0:
        fallback = str(runner_lang or "").strip().lower()
        return "zh" if fallback.startswith("zh") else (
            "en" if fallback.startswith("en") else default)
    return "zh" if cjk / total >= _CJK_RATIO_THRESHOLD else "en"
