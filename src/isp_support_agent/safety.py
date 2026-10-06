import re

# Token/measurement validation, not a phrase blacklist. Customer templates contain no infrastructure.
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
ADDRESS = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|(?:[\da-fA-F]{0,4}:){2,}[\da-fA-F:]*")
HOST = re.compile(r"\b[\w-]+(?:\.[\w-]+)+\b|https?://", re.UNICODE)
IDENTIFIER = re.compile(r"\b[a-z0-9]+(?:[-_][a-z0-9]+)+\b|\b[a-z]+\d+[a-z\d]*\b", re.I)
TECHNICAL = re.compile(
    r"\b(?:hops?|saltos?|rtt|ccq|dbm|mbps|packet[-_ ]?loss)\b|\d\s*(?:%|ms\b)", re.I
)


def valid_rewrite(text: str, allowed_numbers: set[str]) -> bool:
    if not text or len(text) > 2000 or any(ord(c) < 32 and c not in "\n\t" for c in text):
        return False
    if (
        ADDRESS.search(text)
        or HOST.search(text)
        or TECHNICAL.search(text)
        or any(
            not re.fullmatch(r"\d{4}-\d{2}-\d{2}", match.group())
            for match in IDENTIFIER.finditer(text)
        )
    ):
        return False
    return set(NUMBER.findall(text)) <= allowed_numbers


async def safe_rewrite(template: str, model, enabled: bool) -> str:
    if not enabled:
        return template
    numbers = set(NUMBER.findall(template))
    try:
        candidate = await model.rewrite(
            template, {"customer_template": template, "allowed_numbers": sorted(numbers)}
        )
        return candidate if valid_rewrite(candidate, numbers) else template
    except Exception:
        return template
