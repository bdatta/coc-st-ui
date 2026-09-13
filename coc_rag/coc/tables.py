"""
Table reconstruction from pdfplumber with rowspan / colspan support.

pdfplumber's `extract_tables()` returns a flat list-of-lists and silently loses
merged-cell structure. Instead we work from `page.find_tables()`, which exposes
each detected cell as a bounding box. By clustering all cell edges into a global
set of column/row boundaries we can recover, for every cell, the *span* of grid
columns and rows it covers -- i.e. real colspan/rowspan.

Rendering targets:
  * `to_markdown_pipe()`  -- merged values propagated into every covered cell.
                             Lossy structurally, but each row is self-contained,
                             which is what you want for embedding + retrieval.
  * `to_html()`           -- true <td rowspan=.. colspan=..>, full fidelity.
  * `to_row_sentences()`  -- "Col: val | Col: val" per row, very strong signal
                             for semantic search over benefit grids.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BBox = Tuple[float, float, float, float]  # (x0, top, x1, bottom)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Cell:
    r0: int  # first grid row covered
    r1: int  # one past last grid row covered
    c0: int
    c1: int
    text: str = ""

    @property
    def rowspan(self) -> int:
        return max(1, self.r1 - self.r0)

    @property
    def colspan(self) -> int:
        return max(1, self.c1 - self.c0)


@dataclass
class Grid:
    cells: List[Cell]
    n_rows: int
    n_cols: int
    col_edges: List[float] = field(default_factory=list)
    row_edges: List[float] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    header_rows: int = 1
    #: Leading rows that are a single full-width cell -- a caption or lead-in
    #: paragraph enclosed by the table's border rather than tabular data.
    caption_rows: int = 0
    table_id: str = ""

    @property
    def caption(self) -> str:
        if not self.caption_rows:
            return ""
        m = to_matrix(self, propagate=False)
        return " ".join(" ".join(m[r][0].split()) for r in range(self.caption_rows)).strip()

    @property
    def has_spans(self) -> bool:
        return any(c.rowspan > 1 or c.colspan > 1 for c in self.cells)

    @property
    def page_start(self) -> int:
        return min(self.pages) if self.pages else 0

    @property
    def page_end(self) -> int:
        return max(self.pages) if self.pages else 0


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def cluster_edges(values: Iterable[float], tol: float) -> List[float]:
    """Collapse near-identical coordinates into a single ordered edge list."""
    vals = sorted(float(v) for v in values)
    out: List[float] = []
    bucket: List[float] = []
    for v in vals:
        if bucket and v - bucket[0] > tol:
            out.append(sum(bucket) / len(bucket))
            bucket = []
        bucket.append(v)
    if bucket:
        out.append(sum(bucket) / len(bucket))
    return out


def nearest_index(edges: Sequence[float], value: float) -> int:
    best, best_d = 0, abs(value - edges[0])
    for i, e in enumerate(edges):
        d = abs(value - e)
        if d < best_d:
            best, best_d = i, d
    return best


def _bucket_words(
    words: Sequence[dict], xs: Sequence[float], ys: Sequence[float]
) -> Dict[Tuple[int, int], List[dict]]:
    """Assign each word to the atomic grid cell containing its centre point."""
    buckets: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    x_lo, x_hi = xs[0], xs[-1]
    y_lo, y_hi = ys[0], ys[-1]
    for w in words:
        cx = (w["x0"] + w["x1"]) / 2.0
        cy = (w["top"] + w["bottom"]) / 2.0
        if cx < x_lo - 1 or cx > x_hi + 1 or cy < y_lo - 1 or cy > y_hi + 1:
            continue
        ci = min(max(bisect_right(xs, cx) - 1, 0), len(xs) - 2)
        ri = min(max(bisect_right(ys, cy) - 1, 0), len(ys) - 2)
        buckets[(ri, ci)].append(w)
    return buckets


def _join_words(words: Sequence[dict], line_tol: float = 2.5) -> str:
    """Order words into visual lines and join. Newlines preserved as \\n."""
    if not words:
        return ""
    ws = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines: List[List[dict]] = []
    for w in ws:
        if lines and abs(w["top"] - lines[-1][0]["top"]) <= line_tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    out = []
    for ln in lines:
        ln.sort(key=lambda w: w["x0"])
        out.append(" ".join(w["text"] for w in ln).strip())
    return "\n".join(t for t in out if t)


# --------------------------------------------------------------------------- #
# Grid construction
# --------------------------------------------------------------------------- #
def grid_from_table(
    table,
    page_words: Sequence[dict],
    page_number: int,
    x_tol: float = 6.0,
    y_tol: float = 4.0,
    max_cells: int = 4000,
) -> Optional[Grid]:
    """Build a Grid from a pdfplumber Table object plus the page's words."""
    raw: List[BBox] = [tuple(c) for c in (table.cells or []) if c]
    if len(raw) < 2 or len(raw) > max_cells:
        return None

    xs = cluster_edges([c[0] for c in raw] + [c[2] for c in raw], x_tol)
    ys = cluster_edges([c[1] for c in raw] + [c[3] for c in raw], y_tol)
    if len(xs) < 2 or len(ys) < 2:
        return None

    buckets = _bucket_words(page_words, xs, ys)

    cells: List[Cell] = []
    for (x0, top, x1, bottom) in raw:
        c0, c1 = nearest_index(xs, x0), nearest_index(xs, x1)
        r0, r1 = nearest_index(ys, top), nearest_index(ys, bottom)
        if c1 <= c0:
            c1 = c0 + 1
        if r1 <= r0:
            r1 = r0 + 1
        c1 = min(c1, len(xs) - 1)
        r1 = min(r1, len(ys) - 1)
        words: List[dict] = []
        for r in range(r0, r1):
            for c in range(c0, c1):
                words.extend(buckets.get((r, c), ()))
        cells.append(Cell(r0, r1, c0, c1, _join_words(words)))

    grid = Grid(
        cells=cells,
        n_rows=len(ys) - 1,
        n_cols=len(xs) - 1,
        col_edges=list(xs),
        row_edges=list(ys),
        pages=[page_number],
    )
    grid.caption_rows = guess_caption_rows(grid)
    grid.header_rows = guess_header_rows(grid, grid.caption_rows)
    return grid


def _row_is_full_width(grid: Grid, r: int) -> bool:
    anchored = [c for c in grid.cells if c.r0 == r]
    return (len(anchored) == 1 and anchored[0].rowspan == 1
            and anchored[0].colspan >= grid.n_cols and grid.n_cols > 1)


def guess_caption_rows(grid: Grid, max_rows: int = 3) -> int:
    """Leading rows that are one full-width cell.

    Benefit schedules routinely enclose their lead-in paragraph inside the
    table's top border ("Amounts which you are required to pay as shown below
    are based on Allowed Amounts..."). Geometrically it is row zero of the
    table; semantically it is a caption. Left alone it becomes the header, and
    every column label turns into a sentence fragment of that paragraph.
    """
    count = 0
    while count < min(max_rows, grid.n_rows - 1) and _row_is_full_width(grid, count):
        count += 1
    return count


def guess_header_rows(grid: Grid, start: int = 0) -> int:
    """How many rows, beginning at `start`, form the header.

    Two signals. A cell anchored at the first header row that spans several rows
    (typically the stub column, e.g. "Covered Health Care Service") reveals the
    header's depth directly. Failing that, a horizontal merge in that row is a
    banner sitting above a second row of real column labels.
    """
    remaining = grid.n_rows - start
    if remaining <= 1:
        return 1
    deepest = max((c.rowspan for c in grid.cells if c.r0 == start), default=1)
    if deepest > 1:
        return min(deepest, max(1, remaining - 1), 4)
    row_spans = any(c.r0 == start and 1 < c.colspan < grid.n_cols for c in grid.cells)
    if row_spans and remaining >= 3:
        return 2
    return 1


def to_matrix(grid: Grid, propagate: bool = True) -> List[List[str]]:
    """Dense n_rows x n_cols matrix.

    When `propagate`, a merged cell's text is written into every position it
    covers -- that is what makes each row self-contained for retrieval.

    Exception: a cell spanning the *entire* width is a banner or section-label
    row ("Ambulance Services"), not data. Repeating it across every column
    multiplies tokens with no added meaning, so it is written once.
    """
    m = [["" for _ in range(grid.n_cols)] for _ in range(grid.n_rows)]
    for cell in grid.cells:
        full_width = cell.colspan >= grid.n_cols and grid.n_cols > 1
        for r in range(cell.r0, min(cell.r1, grid.n_rows)):
            for c in range(cell.c0, min(cell.c1, grid.n_cols)):
                anchor = (r == cell.r0 and c == cell.c0)
                if anchor or (propagate and not full_width):
                    m[r][c] = cell.text
    return m


def header_labels(grid: Grid) -> List[str]:
    """Flatten a possibly multi-row header into one label per column."""
    m = to_matrix(grid, propagate=True)
    start = min(grid.caption_rows, max(grid.n_rows - 1, 0))
    hr = min(max(grid.header_rows, 1), max(grid.n_rows - start, 1))
    labels: List[str] = []
    for c in range(grid.n_cols):
        parts: List[str] = []
        for r in range(start, start + hr):
            t = " ".join(m[r][c].split())
            if t and (not parts or parts[-1] != t):
                parts.append(t)
        labels.append(" - ".join(parts) if parts else f"Column {c + 1}")
    return labels


def body_matrix(grid: Grid) -> List[List[str]]:
    m = to_matrix(grid, propagate=True)
    start = min(grid.caption_rows, max(grid.n_rows - 1, 0))
    hr = min(max(grid.header_rows, 1), max(grid.n_rows - start, 1))
    return m[start + hr:]


# --------------------------------------------------------------------------- #
# Cross-page stitching
# --------------------------------------------------------------------------- #
def columns_align(a: Grid, b: Grid, tol: float = 8.0) -> bool:
    if a.n_cols != b.n_cols:
        return False
    if not a.col_edges or not b.col_edges:
        return True
    return all(abs(x - y) <= tol for x, y in zip(a.col_edges, b.col_edges))


def _row_text(m: List[List[str]], r: int) -> str:
    return "|".join(" ".join(v.split()).lower() for v in m[r])


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _repeats_header(a_labels: Sequence[str], a_rows: Sequence[str], row: List[str],
                    min_ratio: float = 0.6) -> bool:
    """Is `row` a repetition of the parent table's header on a continuation page?

    Exact equality is not enough: a two-row header flattened to
    'Cost Share - In-Network' will not equal the continuation page's plain
    'In-Network', so a containment test is used per column instead.
    """
    sig = "|".join(_norm(v) for v in row)
    if sig in a_rows:
        return True
    matches = 0
    counted = 0
    for label, value in zip(a_labels, row):
        lv, vv = _norm(label), _norm(value)
        if not vv:
            continue
        counted += 1
        if lv and (vv in lv or lv in vv):
            matches += 1
    return counted > 0 and matches / counted >= min_ratio


def merge_grids(a: Grid, b: Grid) -> Grid:
    """Append `b` beneath `a`, dropping `b`'s header if it repeats `a`'s."""
    ma, mb = to_matrix(a, True), to_matrix(b, True)
    drop = 0
    hr = min(max(a.header_rows, 1), a.n_rows)
    if mb and ma:
        a_labels = header_labels(a)
        a_start = min(a.caption_rows, max(a.n_rows - 1, 0))
        a_rows = {_row_text(ma, r) for r in range(a_start, a_start + hr)}
        a_caption = _norm(a.caption)
        limit = min(b.caption_rows + max(hr, 1) + 1, b.n_rows)
        for i in range(limit):
            row_text = _norm(" ".join(mb[i]))
            caption_repeat = bool(a_caption) and (
                row_text[:60] == a_caption[:60] or _row_is_full_width(b, i))
            if caption_repeat or _repeats_header(a_labels, a_rows, mb[i]):
                drop = i + 1
            else:
                break

    offset = a.n_rows
    cells = list(a.cells)
    for cell in b.cells:
        if cell.r1 <= drop:
            continue
        r0 = max(cell.r0 - drop, 0) + offset
        r1 = max(cell.r1 - drop, 1) + offset
        cells.append(Cell(r0, r1, cell.c0, cell.c1, cell.text))

    return Grid(
        cells=cells,
        n_rows=offset + (b.n_rows - drop),
        n_cols=a.n_cols,
        col_edges=a.col_edges,
        row_edges=[],
        pages=sorted(set(a.pages) | set(b.pages)),
        header_rows=a.header_rows,
        caption_rows=a.caption_rows,
        table_id=a.table_id,
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _esc_pipe(text: str) -> str:
    return " ".join(text.replace("|", "\\|").replace("\n", " <br> ").split())


def to_markdown_pipe(grid: Grid) -> str:
    labels = [_esc_pipe(x) or " " for x in header_labels(grid)]
    lines = ["| " + " | ".join(labels) + " |",
             "|" + "|".join([" --- "] * grid.n_cols) + "|"]
    for row in body_matrix(grid):
        lines.append("| " + " | ".join(_esc_pipe(v) or " " for v in row) + " |")
    return "\n".join(lines)


def _esc_html(text: str) -> str:
    out = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return out.replace("\n", "<br/>")


def to_html(grid: Grid) -> str:
    anchors: Dict[Tuple[int, int], Cell] = {(c.r0, c.c0): c for c in grid.cells}
    covered = set()
    for c in grid.cells:
        for r in range(c.r0, c.r1):
            for cc in range(c.c0, c.c1):
                covered.add((r, cc))

    hr = min(max(grid.header_rows, 1), grid.n_rows)
    out = ["<table>"]
    for r in range(grid.n_rows):
        out.append("  <tr>")
        for c in range(grid.n_cols):
            cell = anchors.get((r, c))
            if cell is None:
                if (r, c) in covered:
                    continue  # consumed by a span
                out.append("    <td></td>")
                continue
            tag = "th" if r < hr else "td"
            attrs = ""
            if cell.rowspan > 1:
                attrs += f' rowspan="{cell.rowspan}"'
            if cell.colspan > 1:
                attrs += f' colspan="{cell.colspan}"'
            out.append(f"    <{tag}{attrs}>{_esc_html(cell.text)}</{tag}>")
        out.append("  </tr>")
    out.append("</table>")
    return "\n".join(out)


def is_banner_row(row: Sequence[str]) -> bool:
    """A row carrying text only in its first column: a group label, not data."""
    if len(row) < 2:
        return False
    first = " ".join(row[0].split())
    rest = [" ".join(v.split()) for v in row[1:]]
    return bool(first) and not any(rest)


def to_row_sentences(grid: Grid, max_rows: int = 400) -> str:
    """One line per data row: 'Group - Label: value | Label: value'.

    Dense, self-describing, and embeds far better than a raw pipe grid because
    every value stays glued to its column name.

    Banner rows carry the benefit name ("Acupuncture") while the rows beneath
    ask "What Is the Coinsurance You Pay?". Read alone, those rows never mention
    the benefit -- so the running banner is carried onto each one. Without this,
    a query about acupuncture cannot match the row holding its coinsurance.
    """
    labels = header_labels(grid)
    out: List[str] = []
    group = ""
    for row in body_matrix(grid)[:max_rows]:
        if is_banner_row(row):
            group = " ".join(row[0].split())
            continue
        parts = []
        for lbl, val in zip(labels, row):
            v = " ".join(val.split())
            if v:
                parts.append(f"{lbl}: {v}")
        if parts:
            out.append((f"{group} - " if group else "") + " | ".join(parts))
    return "\n".join(out)


def looks_degenerate(grid: Grid, min_fill: float = 0.15) -> bool:
    """Reject 'tables' that are really just whitespace-aligned prose."""
    if grid.n_rows < 2 or grid.n_cols < 2:
        return True
    filled = sum(1 for c in grid.cells if c.text.strip())
    return filled / max(len(grid.cells), 1) < min_fill
