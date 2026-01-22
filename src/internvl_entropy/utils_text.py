import re


_whitespace_re = re.compile(r"\s+")


def clean_text(text: str) -> str:
    cleaned = text.replace("Answer:", "")
    cleaned = _whitespace_re.sub(" ", cleaned).strip()
    return cleaned
