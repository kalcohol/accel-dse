"""0.189: provenance tags.  A tag is one base label or a compound of distinct base labels joined by '/', e.g. 「推导/二手」
(derived from a secondary source) or 「二手/假设」."""
BASE = ("来源", "推导", "二手", "假设")


def valid_tag(t) -> bool:
    if not isinstance(t, str) or not t:
        return False
    parts = t.split("/")
    return all(p in BASE for p in parts) and len(set(parts)) == len(parts)
