"""
report.py
---------
Stata-style text output: a regression table with z, p, confidence interval
and significance stars, an esttab-like comparison across estimators, a
block of Wald tests, and a horizon table for impulse responses.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Optional, Sequence


def stars(p: float) -> str:
    """*** p<0.01, ** p<0.05, * p<0.10."""
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def regression_table(names: Sequence[str], coef: np.ndarray, se: np.ndarray, title: str = "",
                     depvar: str = "y", header: Optional[dict] = None, level: float = 0.95,
                     width: int = 14, skip: Optional[Sequence[str]] = None) -> str:
    """One estimator, Stata layout: Coef. Std.Err. z P>|z| [conf. interval] stars."""
    z = stats.norm.ppf(0.5 + level / 2)
    lines = []
    head = header or {}
    keys = list(head)
    if title:
        lines.append(title)
    for k in keys:
        lines.append(" " * 48 + f"{k:<20}= {head[k]:>10}")
    lines.append("-" * 82)
    lines.append(f"{depvar:>{width}} |      Coef.   Std. Err.       z    P>|z|     [{int(level*100)}% Conf. Interval]")
    lines.append("-" * (width + 1) + "+" + "-" * 70)
    skip = set(skip or [])
    for nm, b, s in zip(names, coef, se):
        if any(nm.startswith(sk) for sk in skip):
            continue
        if s > 0 and np.isfinite(s):
            zz = b / s
            p = 2 * stats.norm.sf(abs(zz))
            lines.append(f"{nm:>{width}} | {b:10.4f} {s:10.4f} {zz:8.2f} {p:8.3f} {b - z*s:11.4f} {b + z*s:11.4f} {stars(p):>4}")
        else:
            lines.append(f"{nm:>{width}} | {b:10.4f} {'':>10} {'':>8} {'':>8} {'':>11} {'':>11}")
    lines.append("-" * 82)
    return "\n".join(lines)


def comparison_table(columns: Sequence[dict], rows: Sequence[str], labels: Optional[dict] = None,
                     title: str = "", stats_rows: Optional[Sequence[tuple]] = None, width: int = 16) -> str:
    """
    esttab-style: coefficient with stars on one line, standard error in
    parentheses below. Each column is {'title', 'names', 'coef', 'se'}.
    stats_rows: list of (label, [value per column]) printed at the bottom.
    """
    labels = labels or {}
    ncol = len(columns)
    sep = "-" * (26 + (width + 1) * ncol)
    lines = [title, "=" * len(sep)]
    lines.append(f"{'':<26}" + "".join(f"{c['title']:>{width}} " for c in columns))
    lines.append(f"{'':<26}" + "".join(f"{'(' + str(i + 1) + ')':>{width}} " for i in range(ncol)))
    lines.append(sep)
    for r in rows:
        top, bot = f"{labels.get(r, r):<26}", f"{'':<26}"
        for c in columns:
            if r in c["names"]:
                j = c["names"].index(r)
                b, s = c["coef"][j], c["se"][j] if c["se"] is not None else np.nan
                if np.isfinite(s) and s > 0:
                    p = 2 * stats.norm.sf(abs(b / s))
                    top += f"{f'{b:.4f}{stars(p)}':>{width}} "
                    bot += f"{f'({s:.4f})':>{width}} "
                else:
                    top += f"{f'{b:.4f}':>{width}} "
                    bot += f"{'':>{width}} "
            else:
                top += f"{'':>{width}} "
                bot += f"{'':>{width}} "
        lines += [top, bot]
    if stats_rows:
        lines.append(sep)
        for lab, vals in stats_rows:
            lines.append(f"{lab:<26}" + "".join(f"{str(v):>{width}} " for v in vals))
    lines.append("=" * len(sep))
    lines.append("Standard errors in parentheses. * p<0.10, ** p<0.05, *** p<0.01")
    return "\n".join(lines)


def tests_block(tests: dict, title: str = "Wald tests") -> str:
    lines = [title, "-" * 70, f"{'hypothesis':<44}{'chi2':>9}{'df':>5}{'p':>8}   "]
    for k, v in tests.items():
        lines.append(f"{k:<44}{v['stat']:9.2f}{v['df']:5d}{v['p']:8.3f} {stars(v['p']):>4}")
    lines.append("-" * 70)
    return "\n".join(lines)


def horizon_table(h: np.ndarray, se: np.ndarray, title: str = "", label: str = "h") -> str:
    lines = [title, "-" * 56, f"{label:>6} {'effect':>12} {'std.err.':>12} {'z':>8} {'p':>8}"]
    for i, (b, s) in enumerate(zip(h, se)):
        z = b / s if s > 0 else np.nan
        p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
        lines.append(f"{i:>6} {b:12.4f} {s:12.4f} {z:8.2f} {p:8.3f} {stars(p):>4}")
    lines.append("-" * 56)
    return "\n".join(lines)
