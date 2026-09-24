"""Shared Matplotlib style for the figure scripts in this directory.

One colour per role, used identically in every figure:

    reference / wild type   #0072B2   blue
    variant / mutant        #D55E00   vermillion
    compensated / treated   #009E73   bluish green
    gating m / h / n        #7B3294 / #56B4E9 / #E69F00

The role colours are taken from the Okabe-Ito colourblind-safe palette.

Usage:
    from wo7_style import apply_style, C, save
    apply_style()
    fig, ax = plt.subplots(...)
    ...
    save(fig, "figures/<content_name>")     # writes .pdf AND .png at 300 dpi
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- the one colour per role -------------------------------------------------
class C:
    WT = "#0072B2"            # reference / wild type
    VARIANT = "#D55E00"       # variant / mutant
    COMPENSATED = "#009E73"   # compensated / treated variant
    M = "#7B3294"             # gating variable m
    H = "#56B4E9"             # gating variable h
    N = "#E69F00"             # gating variable n
    INK = "#1A1A1A"           # primary text
    MUTED = "#5A5A5A"         # secondary text, annotation rules
    GRID = "#D9D9D9"
    SURFACE = "#FFFFFF"

    # region shading for the three-region excitability figure, kept pale so the
    # data marks stay dominant
    BAND_LESS = "#D55E00"
    BAND_SAME = "#8A8A8A"
    BAND_MORE = "#0072B2"


def apply_style():
    """Install the shared rcParams.  Call once before creating any figure."""
    plt.rcParams.update({
        # font: one family, one small set of sizes
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,

        # recessive grid and axes, no chartjunk
        "axes.grid": True,
        "grid.color": C.GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "axes.edgecolor": C.MUTED,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": C.INK,
        "text.color": C.INK,
        "xtick.color": C.MUTED,
        "ytick.color": C.MUTED,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # marks
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "lines.solid_capstyle": "round",

        # legend: present whenever >= 2 series, no frame
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.4,

        # output
        "figure.facecolor": C.SURFACE,
        "axes.facecolor": C.SURFACE,
        "savefig.facecolor": C.SURFACE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,      # embed TrueType, editable text in the PDF
        "ps.fonttype": 42,
    })


# ---- automatic style check ---------------------------------------------------
# audit_figure() checks three rules:
#   1. every quantitative axis label carries a unit
#   2. every panel title that states a value carries a unit
#   3. a legend never repeats the same label
# A trailing \b would be wrong in _UNIT.  Python's \w is Unicode-aware and "²" is
# alphanumeric, so "mS/cm²" has NO word boundary after "cm" and \b would reject a
# correctly-labelled axis.  A negative lookahead for an ASCII letter is what is
# actually meant.
_UNIT = re.compile(r"(?<![A-Za-z])(ms|mV|nA|pA|Hz|mS/cm|mho/cm|GB/s|s|count|"
                   r"dimensionless|sets per second|per second)(?![A-Za-z])|%|µ")
_HASNUM = re.compile(r"\d")


def _categorical(ax, which):
    """True if this axis carries text categories rather than a quantity.  Such an
    axis has no unit to state, so the unit rule does not apply to it."""
    ticks = (ax.get_xticklabels() if which == "x" else ax.get_yticklabels())
    texts = [t.get_text() for t in ticks if t.get_text().strip()]
    if not texts:
        return False
    def numeric(t):
        try:
            float(t.replace("\u2212", "-").replace(",", "").replace("%", ""))
            return True
        except ValueError:
            return False
    return sum(not numeric(t) for t in texts) > len(texts) / 2


def _categorical_group(ax, which):
    """True if this axis, or any axis sharing it, carries text categories.
    Shared axes hide the tick labels on all but one member, so asking only the
    axes in front of you gives the wrong answer."""
    group = (ax.get_shared_x_axes() if which == "x"
             else ax.get_shared_y_axes()).get_siblings(ax)
    return any(_categorical(a, which) for a in group)


def _labelled(ax, which):
    """True if this axis is labelled, or shares its axis with one that is."""
    get = (lambda a: a.get_xlabel()) if which == "x" else (lambda a: a.get_ylabel())
    group = (ax.get_shared_x_axes() if which == "x"
             else ax.get_shared_y_axes()).get_siblings(ax)
    return [get(a) for a in group if get(a).strip()]


def audit_figure(fig):
    """Return a list of style violations.  Empty list means the figure passes."""
    bad = []
    for i, ax in enumerate(fig.get_axes()):
        if not (ax.lines or ax.patches or ax.collections or ax.images):
            continue                                    # decorative axes
        if not ax.axison:
            # The axis is switched off, so there is no axis to label: an embedded
            # bitmap, or a rendered table drawn as text and rules.  The unit
            # rules apply only to data axes.
            continue
        name = (ax.get_title(loc="left") or ax.get_title() or ax.get_label()
                or f"axes[{i}]")
        for which in ("x", "y"):
            if _categorical_group(ax, which):
                continue
            labels = _labelled(ax, which)
            if not labels:
                bad.append(f"{name}: {which} axis has no label")
            elif not any(_UNIT.search(l) for l in labels):
                bad.append(f"{name}: {which} label {labels[0]!r} states no unit")
        # get_title() returns the CENTRE title only; a left-aligned title is
        # invisible to it, so every location must be asked for explicitly.
        for loc in ("center", "left", "right"):
            t = ax.get_title(loc=loc)
            if t and _HASNUM.search(t) and not _UNIT.search(t):
                bad.append(f"panel title {t!r} states a value with no unit")
        leg = ax.get_legend()
        if leg is not None:
            texts = [x.get_text() for x in leg.get_texts()]
            if len(texts) != len(set(texts)):
                bad.append(f"{name}: legend repeats a label {texts}")
    return bad


def save(fig, stem, dpi=300, audit=True, allow=()):
    """Write `stem`.pdf (vector) and `stem`.png (300 dpi).

    Never overwrites an existing file silently -- it raises instead.  Runs the
    style check and prints any violation; `allow` is a list of substrings for
    violations that are deliberate and should not be printed."""
    if audit:
        bad = [b for b in audit_figure(fig)
               if not any(a in b for a in allow)]
        if bad:
            print(f"  STYLE PASS -- {len(bad)} violation(s) in {stem}:")
            for b in bad:
                print(f"     ! {b}")
        else:
            print(f"  STYLE PASS -- clean: {stem}")
    for ext in ("pdf", "png"):
        path = f"{stem}.{ext}"
        if os.path.exists(path):
            raise FileExistsError(
                f"{path} already exists; save() does not overwrite a figure. "
                f"Choose a new name or delete deliberately.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return [f"{stem}.pdf", f"{stem}.png"]
