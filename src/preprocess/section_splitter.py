import re
from typing import List, Dict

SECTION_MARKERS = {
    "PREAMBLE": [
        r"(?i)^whereas",
        r"(?i)^having regard",
        r"(?i)^the council",
        r"(?i)^the european parliament",
    ],
    "DEFINITIONS": [
        r"(?i)for the purposes of this",
        r"(?i)in this (regulation|directive)",
        r"(?i)'[^']+' means",
    ],
    "PROVISIONS": [
        r"(?i)^article\s+\d+",
        r"(?i)^section\s+\d+",
        r"^\d+\.\s+[A-Z]",
    ],
    "PENALTIES": [
        r"(?i)(penalt|sanction|fine|infringement)",
    ],
    "ANNEX": [
        r"(?i)^annex\s+(i|ii|iii|iv|v|\d+)",
    ],
}


def detect_section_type(paragraph: str) -> str:
    for section_type, patterns in SECTION_MARKERS.items():
        for pattern in patterns:
            if re.search(pattern, paragraph[:200]):
                return section_type
    return "PROVISIONS"


def split_into_sections(text: str) -> List[Dict]:
    """
    Split a legal document into sections (paragraph groups).

    Returns a list of dicts:
      { "type": str, "text": str, "idx": int }
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    sections = []
    for i, para in enumerate(paragraphs):
        if len(para) < 20:
            continue
        section_type = detect_section_type(para)
        sections.append({"type": section_type, "text": para, "idx": i})

    if not sections:
        sections = [{"type": "PROVISIONS", "text": text, "idx": 0}]

    return sections