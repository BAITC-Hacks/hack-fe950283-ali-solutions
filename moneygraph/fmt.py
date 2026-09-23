"""Человекочитаемые числа для evidence и карточек."""


def kzt(x: float) -> str:
    x = float(x)
    if x >= 1e6:
        return f"{x / 1e6:.1f} млн ₸".replace(".0 ", " ")
    if x >= 1e3:
        return f"{x / 1e3:.0f} тыс ₸"
    return f"{x:.0f} ₸"


def pct(x: float) -> str:
    return f"{100 * float(x):.0f}%"


def clip_text(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip(" ;,") + "…"


def short(gid) -> str:
    """Короткая метка для схемы: 1000000XXXXXXXX100 → XXXXXXXX (уникальна в датасете)."""
    s = str(gid)
    return s[7:-3] if len(s) == 18 and s.startswith("1000000") else s
