import re
from typing import List, Dict

AUTHORITY_PATTERNS = [
    # Regulations
    (r"\bRegulation\s*\(\s?(?:EU|EC)\s?\)\s*No\s*\d+/\d{4}", "REGULATION"),
    (r"\bRegulation\s*\(\s?(?:EU|EC)\s?\)\s*\d+/\d{4}", "REGULATION"),
    # Directives
    (r"\bDirective\s*\d{4}/\d{2,4}/?(?:EU|EC)?", "DIRECTIVE"),
    (r"\bDirective\s*(?:EU|EC)\s*\d{4}/\d{2,4}", "DIRECTIVE"),
    # Decisions
    (r"\bDecision\s*\(\s?(?:EU|EC)\s?\)\s*\d+/\d{4}", "DECISION"),
    # Articles
    (r"\bArticle\s+\d+[a-zA-Z]?", "ARTICLE"),
    (r"\bArt\.\s*\d+[a-zA-Z]?", "ARTICLE"),
    # Cases (simplified)
    (r"\bCase\s+[A-Z]\s*\d+/\d{2}", "CASE"),
]


def classify_level(auth_type: str) -> float:
    """
    Assign approximate lex superior levels:
    Higher = more authoritative.
    """
    if auth_type in {"REGULATION", "DIRECTIVE"}:
        return 3.0
    if auth_type in {"DECISION"}:
        return 2.5
    if auth_type in {"ARTICLE"}:
        return 2.0
    if auth_type in {"CASE"}:
        return 2.0
    return 1.0


def extract_authorities(text: str) -> List[Dict]:
    """
    Extract legal authority mentions with type and level.

    Returns list of:
      { "text": str, "type": str, "start": int, "end": int, "level": float }
    """
    results = []
    for pattern, auth_type in AUTHORITY_PATTERNS:
        for match in re.finditer(pattern, text):
            span_text = match.group(0)
            start, end = match.span()
            results.append({
                "text": span_text,
                "type": auth_type,
                "start": start,
                "end": end,
                "level": classify_level(auth_type),
            })
    return results