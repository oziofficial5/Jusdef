import re
from typing import List, Tuple

OPERATOR_PATTERNS = {
    "OVR": [
        r"notwithstanding\b",
        r"without prejudice to",
        r"takes? precedence over",
        r"shall prevail over",
        r"by way of derogation from",
        r"in derogation of",
        r"overrides?",
    ],
    "EXC": [
        r"except (?:where|when|in|for|as provided)",
        r"unless\s+(?:the|a|it|otherwise)",
        r"save (?:where|for|as)",
        r"other than\s+(?:in|where|when)",
        r"with the exception of",
        r"excluding\s+(?:the|any|cases)",
        r"subject to (?:the|article|paragraph)",
        r"provided that",
        r"on condition that",
    ],
    "NEG": [
        r"shall not (?:apply|be|have)",
        r"does not apply",
        r"is not applicable",
        r"shall be excluded",
        r"no\s+\w+\s+shall",
        r"not\s+(?:be\s+)?(?:required|permitted|allowed)",
        r"prohibited",
        r"is excluded from",
    ],
}


def detect_operator(sentence: str) -> str:
    """
    Classify a sentence as one of:
    - OVR (override)
    - EXC (exception)
    - NEG (negation)
    - AFF (affirmation / default)
    """
    sentence_lower = sentence.lower()
    for operator in ["OVR", "EXC", "NEG"]:
        for pattern in OPERATOR_PATTERNS[operator]:
            if re.search(pattern, sentence_lower):
                return operator
    return "AFF"


def detect_operators_in_section(
    section_text: str,
    concept_spans: List[Tuple[int, int]],
) -> List[str]:
    """
    Given a section and list of concept spans (start,end),
    return one operator label per span.
    """
    sentences = re.split(r"(?<=[.;])\s+", section_text)
    operators = []

    for (start, end) in concept_spans:
        char_count = 0
        governing_sentence = section_text
        for sent in sentences:
            char_count += len(sent) + 1
            if char_count >= start:
                governing_sentence = sent
                break
        operators.append(detect_operator(governing_sentence))

    return operators