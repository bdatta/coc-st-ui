"""
Layout analysis: words -> lines -> headings / paragraphs, plus detection of
running headers and footers.

Everything here is heuristic and deliberately tunable from the Streamlit UI,
because COC formatting varies a lot between carriers. The two knobs that matter
most in practice are `heading_size_ratio` and `treat_allcaps_as_heading`.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

#: Bullet markers, including the Unicode Private Use Area. Symbol and Wingdings
#: glyphs embed as PUA code points (U+F0A7 is the Wingdings square bullet, seen
#: 176 times in one real SPD). Unrecognised, each bulleted exclusion merges into
#: the preceding paragraph, so fifteen separate rules become one 700-character
#: blob that is useless to quote in a letter.
#: Bullet glyphs, including the Unicode Private Use Area and the geometric
#: shapes carriers use for nested levels. Real documents mix them freely: one
#: SPD uses U+F0A7 (Wingdings) throughout, another uses U+2022 for top-level
#: items and U+2666 for a third level. An unrecognised glyph merges every item
#: beneath it into one paragraph, so a specific exclusion stops being citable
#: on its own.
BULLET_CHARS = (
    r"\u2022\u25cf\u25aa\u25ab\u25e6\u25c6\u25c7\u2666\u2665\u2663\u2660"
    r"\u25a0\u25a1\u25b8\u25b9\u2023\u2043\u00b7\u2013\u2014\-\*\u00a7"
    r"\uE000-\uF8FF"
)
BULLET_RE = re.compile(
    rf"^\s*(?:[{BULLET_CHARS}]\s*|\(?[a-zA-Z0-9]{{1,3}}[\.\)]\s+)"
)

#: Same markers, wherever they appear in a line -- used to split a run-on
#: paragraph that was assembled before the marker was recognised.
INLINE_BULLET_RE = re.compile(
    rf"[\u2022\u25cf\u25aa\u25ab\u25e6\u25c6\u25c7\u2666\u2665\u2663\u2660"
    rf"\u25a0\u25a1\u25b8\u25b9\u2023\u2043\uE000-\uF8FF]\s*")
NUMBERED_RE = re.compile(r"^\s*((?:\d+|[IVXLCivxlc]+)(?:\.\d+)*)[\.\)]?\s+\S")
#: Requires a designator after the keyword ("Section 1", "Appendix A",
#: "Article IV"). Without it, ordinary prose beginning "Schedule of Benefits
#: will tell you..." gets promoted to a heading -- observed on real COCs.
KEYWORD_RE = re.compile(
    r"^\s*(SECTION|ARTICLE|PART|APPENDIX|SCHEDULE|EXHIBIT|CHAPTER|RIDER|AMENDMENT)"
    r"\s+([0-9]+|[IVXLC]+|[A-Z])\b",
    re.IGNORECASE,
)
SENTENCE_END_RE = re.compile(r"[\.;,]\s*$")

#: A division heading that trails off mid-clause is a cross-reference caught by
#: a line break, not a title: "Section 1: Covered Health Care Services; and".
DIVISION_CONTINUATION_RE = re.compile(
    r"([.;,]\s*$|\b(and|or|of|for|in|to|the|that|which|will|shall|as|is|are)\s*$)",
    re.IGNORECASE)
BOLD_HINT_RE = re.compile(r"(bold|black|heavy|semib|demi)", re.IGNORECASE)


@dataclass
class Line:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    bold: bool
    page: int
    #: Fraction of characters at the dominant size. A line mixing a large bold
    #: cross-reference into body prose ("See Appendix A - Clinical Programs for
    #: more details.") scores low here, which is what keeps it out of the
    #: heading set.
    size_frac: float = 1.0

    @property
    def height(self) -> float:
        return self.bottom - self.top


def _is_bold(fontname: str) -> bool:
    return bool(BOLD_HINT_RE.search(fontname or ""))


def words_to_lines(words: Sequence[dict], page_number: int, y_tol: float = 2.5) -> List[Line]:
    """Cluster words into visual lines, carrying font size and boldness."""
    if not words:
        return []
    ws = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    groups: List[List[dict]] = []
    for w in ws:
        if groups and abs(w["top"] - groups[-1][0]["top"]) <= y_tol:
            groups[-1].append(w)
        else:
            groups.append([w])

    lines: List[Line] = []
    for g in groups:
        g.sort(key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in g).strip()
        if not text:
            continue
        # Dominant size weighted by characters, not the median: a heading is
        # whatever size most of the line's ink is set in.
        by_size: Counter = Counter()
        for w in g:
            s = round(float(w.get("size", 0) or 0), 1)
            if s > 0:
                by_size[s] += len(w["text"])
        total_chars = sum(by_size.values()) or 1
        dominant, dom_chars = (by_size.most_common(1)[0] if by_size else (0.0, 0))
        bold_chars = sum(len(w["text"]) for w in g if _is_bold(str(w.get("fontname", ""))))
        lines.append(
            Line(
                text=text,
                x0=min(w["x0"] for w in g),
                x1=max(w["x1"] for w in g),
                top=min(w["top"] for w in g),
                bottom=max(w["bottom"] for w in g),
                size=dominant,
                bold=bold_chars / total_chars > 0.6,
                page=page_number,
                size_frac=dom_chars / total_chars,
            )
        )
    lines.sort(key=lambda ln: (ln.top, ln.x0))
    return lines


# --------------------------------------------------------------------------- #
# Running headers / footers
# --------------------------------------------------------------------------- #
_DIGITS_RE = re.compile(r"\d+")


def normalize(text: str) -> str:
    return _DIGITS_RE.sub("#", " ".join(text.split()).lower()).strip()


def find_running_lines(
    pages_lines: Sequence[Sequence[Line]],
    page_heights: Sequence[float],
    band: float = 0.10,
    min_fraction: float = 0.45,
    max_threshold: int = 10,
) -> Set[str]:
    """Normalized text of lines that repeat in the top/bottom band of most pages.

    The threshold is capped rather than purely proportional. Long benefit
    documents change their footer per section -- "39 Schedule of Benefits" runs
    for thirty pages, then "72 Section 1: Covered Health Care Services" takes
    over. A percentage-of-all-pages rule never fires for any of them, and the
    surviving footer text lands between a table and its continuation on the next
    page, silently breaking cross-page stitching.
    """
    counts: Counter = Counter()
    n_pages = max(len(pages_lines), 1)
    for lines, height in zip(pages_lines, page_heights):
        seen = set()
        for ln in lines:
            in_band = ln.top <= height * band or ln.bottom >= height * (1 - band)
            if not in_band:
                continue
            key = normalize(ln.text)
            if key and key not in seen:
                seen.add(key)
                counts[key] += 1
    threshold = max(3, min(int(n_pages * min_fraction), max_threshold))
    return {k for k, c in counts.items() if c >= threshold}


#: Form/version codes printed in footers, e.g. "COC25.INS.2018.LG.CO".
#: Composite policy documents stamp a different one on each bound instrument,
#: so any single code may appear on only a handful of pages -- too few for
#: frequency-based running-line detection to catch.
FORM_CODE_RE = re.compile(r"^[A-Z]{2,5}\d{0,4}(\.[A-Z0-9]{1,12}){2,}$")


def is_page_number(text: str) -> bool:
    """Page furniture: arabic numerals, roman numerals, or a form code.

    Roman numerals matter for policy binders, where front matter and appended
    notices are often numbered i, ii, iii while the body restarts at 1. Digit-
    only matching leaves those footers behind, and they surface as one-word
    paragraphs scattered through the output."""
    t = " ".join(text.split())
    if not t:
        return False
    if re.fullmatch(r"(page\s*)?[-–\s]*\d{1,4}([-–/ ]+(of\s*)?\d{1,4})?[-–\s]*", t, re.I):
        return True
    if re.fullmatch(r"[-–\s]*[IVXLCivxlc]{1,7}[-–\s]*", t) and len(t.strip(" -–")) <= 7:
        return True
    # "COC25.INS.2018.LG.CO 46" -- code with or without a trailing page number
    stripped = re.sub(r"\s+\d{1,4}$", "", t).strip()
    return bool(FORM_CODE_RE.match(stripped))


# --------------------------------------------------------------------------- #
# Headings
# --------------------------------------------------------------------------- #
def modal_body_size(pages_lines: Sequence[Sequence[Line]]) -> float:
    counts: Counter = Counter()
    for lines in pages_lines:
        for ln in lines:
            counts[ln.size] += len(ln.text)
    if not counts:
        return 10.0
    return counts.most_common(1)[0][0]


def build_size_levels(
    pages_lines: Sequence[Sequence[Line]],
    body_size: float,
    size_ratio: float,
    max_levels: int = 6,
) -> Dict[float, int]:
    """Map each distinct 'large' font size to a heading level (1 = largest)."""
    counts: Counter = Counter()
    for lines in pages_lines:
        for ln in lines:
            if ln.size >= body_size * size_ratio and len(ln.text) <= 160:
                counts[ln.size] += 1
    sizes = sorted((s for s in counts if counts[s] >= 1), reverse=True)
    return {s: min(i + 1, max_levels) for i, s in enumerate(sizes[:max_levels])}


def heading_level(
    line: Line,
    body_size: float,
    size_levels: Dict[float, int],
    *,
    max_len: int = 160,
    treat_allcaps_as_heading: bool = True,
    treat_bold_as_heading: bool = True,
    min_size_frac: float = 0.75,
    force_patterns: Sequence[re.Pattern] = (),
) -> Optional[int]:
    """Return a heading level (1..6) or None if the line is body text."""
    text = line.text.strip()
    if not text or len(text) > max_len:
        return None
    if BULLET_RE.match(text) and not KEYWORD_RE.match(text):
        return None

    for pat in force_patterns:
        if pat.search(text):
            return 1

    letters = sum(ch.isalpha() for ch in text)
    if letters < 3:
        return None

    # A named division ("Section 5: How to File a Claim", "Appendix B: ...") is
    # structurally top level whatever its point size. Ranking purely by font
    # size is fragile: if a subsection happens to rank at the same level as its
    # parent, the child replaces the parent on the heading stack and the section
    # vanishes from every breadcrumb beneath it.
    #
    # But plan documents cross-reference their own sections constantly, and a
    # line break can leave "Section 1: Covered Health Care Services; and" or
    # "... Services will continue until directed by the Commissioner" starting a
    # line. Promoting one of those replaces the real heading and every exclusion
    # beneath it is then cited under the wrong section. So a division heading
    # must also look like a title: set in larger type than the body, short, and
    # not trailing off into the next clause.
    if (KEYWORD_RE.match(text)
            and line.size > body_size
            and len(text) <= 70
            and not DIVISION_CONTINUATION_RE.search(text)
            and len(KEYWORD_RE.findall(text)) <= 1):
        return 1

    # A line whose large type is only part of it is prose containing a
    # cross-reference, not a heading.
    if line.size_frac < min_size_frac:
        return None

    # A line that names a document division but trails off mid-clause is a
    # cross-reference caught by a line break, never a heading -- whatever size
    # it is set in. This has to be checked before the font-size path, not only
    # on the division-promotion path: carriers set cross-references in the same
    # style as headings, so "Section 1: Covered Health Care Services." (40
    # characters, so the sentence-end rule does not catch it either) was being
    # promoted on size alone. Each one replaces the real section on the heading
    # stack, and every exclusion beneath it is then cited under the wrong
    # section -- or, because it is out of scope, not found at all.
    if KEYWORD_RE.match(text) and (
            DIVISION_CONTINUATION_RE.search(text)
            or len(text) > 70
            or len(KEYWORD_RE.findall(text)) > 1):
        return None

    ends_like_sentence = bool(SENTENCE_END_RE.search(text)) and len(text) > 70
    size_level = size_levels.get(line.size)

    if size_level is not None and not ends_like_sentence:
        return size_level

    deepest = (max(size_levels.values()) if size_levels else 1) + 1
    deepest = min(deepest, 5)

    # A keyword heading set in body type, in a regular weight, ending in a
    # full stop is a cross-reference inside a sentence -- not a heading.
    if KEYWORD_RE.match(text) and len(text) < 120:
        if (line.size > body_size or line.bold) and not text.rstrip().endswith("."):
            return min(deepest, 2)
        return None

    upper_ratio = sum(ch.isupper() for ch in text if ch.isalpha()) / max(letters, 1)
    if treat_allcaps_as_heading and upper_ratio > 0.9 and len(text) < 110 and not ends_like_sentence:
        return deepest

    if treat_bold_as_heading and line.bold and len(text) < 110 and not ends_like_sentence:
        return deepest

    if NUMBERED_RE.match(text) and (line.bold or line.size > body_size) and len(text) < 110:
        return min(deepest + 1, 6)

    return None


# --------------------------------------------------------------------------- #
# Printed page numbering
# --------------------------------------------------------------------------- #
PAGE_NUM_TOKEN_RE = re.compile(r"^(\d{1,4})$")
ROMAN_TOKEN_RE = re.compile(r"^(?=[ivxlcdm]{1,7}$)[ivxlcdm]+$", re.I)
ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _roman_to_int(text: str) -> Optional[int]:
    total = prev = 0
    for ch in reversed(text.lower()):
        value = ROMAN_VALUES.get(ch)
        if value is None:
            return None
        total = total - value if value < prev else total + value
        prev = max(prev, value)
    return total or None


@dataclass
class PageNumbering:
    """Where printed page numbering starts, and how it maps to PDF positions."""
    first_numbered_page: Optional[int]      # 1-based PDF position
    offset: Optional[int]                   # pdf_page - printed_page
    confidence: float                       # share of pages agreeing
    consistent_pages: int
    pages_examined: int
    roman_pages: List[int] = field(default_factory=list)
    samples: List[Tuple[int, int]] = field(default_factory=list)  # (pdf, printed)

    @property
    def found(self) -> bool:
        return self.first_numbered_page is not None and self.confidence >= 0.5


def detect_page_numbering(pdf, band: float = 0.12,
                          sample_limit: Optional[int] = None) -> PageNumbering:
    """Find the first page carrying printed page numbering.

    Works from agreement rather than from a single lucky match. Every integer in
    a page's header/footer band is a candidate; for each, the implied offset
    (pdf position minus printed number) is recorded. The offset the whole
    document agrees on is the real one, and the first page consistent with it is
    where numbering begins.

    That agreement test is what makes this trustworthy. A cover page carrying
    "Group Number: 168504" or a footer reading "Form 12-B" produces integers
    too, but they imply nonsense offsets that no other page repeats.

    Roman-numeral front matter is reported separately: those pages are numbered,
    but they are front matter, and starting conversion there defeats the purpose.
    """
    pages = pdf.pages if sample_limit is None else pdf.pages[:sample_limit]
    per_page: Dict[int, Set[int]] = {}
    roman_pages: List[int] = []
    offsets: Counter = Counter()

    for idx, page in enumerate(pages, start=1):
        height = float(page.height)
        candidates: Set[int] = set()
        try:
            words = page.extract_words()
        except Exception:
            words = []
        for w in words:
            if not (w["top"] <= height * band or w["bottom"] >= height * (1 - band)):
                continue
            token = w["text"].strip().strip(".)(-")
            m = PAGE_NUM_TOKEN_RE.match(token)
            if m:
                value = int(m.group(1))
                if 1 <= value <= 9999:
                    candidates.add(value)
            elif ROMAN_TOKEN_RE.match(token):
                value = _roman_to_int(token)
                if value and value <= 50 and idx not in roman_pages:
                    roman_pages.append(idx)
        per_page[idx] = candidates
        for value in candidates:
            delta = idx - value
            if 0 <= delta <= 60:          # a plausible amount of front matter
                offsets[delta] += 1
        try:
            page.flush_cache()
        except Exception:
            pass

    if not offsets:
        return PageNumbering(None, None, 0.0, 0, len(per_page), roman_pages, [])

    offset, agree = offsets.most_common(1)[0]
    consistent = sorted(idx for idx, vals in per_page.items() if (idx - offset) in vals)
    first = consistent[0] if consistent else None

    # Ignore an isolated early match: require the run to actually continue.
    if first is not None and len(consistent) > 2:
        for candidate in consistent:
            following = [p for p in consistent if candidate <= p <= candidate + 4]
            if len(following) >= 3:
                first = candidate
                break

    return PageNumbering(
        first_numbered_page=first,
        offset=offset,
        confidence=round(agree / max(len(per_page), 1), 3),
        consistent_pages=len(consistent),
        pages_examined=len(per_page),
        roman_pages=roman_pages,
        samples=[(p, p - offset) for p in consistent[:8]],
    )


# --------------------------------------------------------------------------- #
# Form-coded document sections
# --------------------------------------------------------------------------- #
#: A footer such as "SBN25.CHCSELDP.I.2018.LG.CO 14": an identifier followed by
#: the page number within that form. Requires the number LAST and at least two
#: dots in the identifier, which is what separates it from a footer like
#: "19 Schedule of Benefits" -- number first, no dots -- used by documents that
#: number straight through.
FORM_FOOTER_RE = re.compile(r"^(.{5,80}?)\s+(\d{1,4})$")
MIN_FORM_DOTS = 2


@dataclass
class FormSection:
    """A run of pages sharing one form code, with its own page numbering."""
    form_code: str
    pdf_start: int
    pdf_end: int
    printed_start: int
    printed_end: int

    def printed_for(self, pdf_page: int) -> Optional[int]:
        if not (self.pdf_start <= pdf_page <= self.pdf_end):
            return None
        return self.printed_start + (pdf_page - self.pdf_start)


@dataclass
class FormLayout:
    sections: List[FormSection] = field(default_factory=list)
    pages_with_code: int = 0
    pages_examined: int = 0

    @property
    def coverage(self) -> float:
        return self.pages_with_code / max(self.pages_examined, 1)

    @property
    def found(self) -> bool:
        """Only treat the document as form-coded when most of it agrees.

        A handful of incidental matches must not switch on a different citation
        scheme for a document that numbers straight through.
        """
        return len(self.sections) >= 2 and self.coverage >= 0.30

    def for_page(self, pdf_page: int) -> Optional[FormSection]:
        for section in self.sections:
            if section.pdf_start <= pdf_page <= section.pdf_end:
                return section
        return None


def _form_code_of(footer: str) -> Optional[Tuple[str, int]]:
    """Pull a trailing form code and page number out of a footer line.

    The code is read as a *suffix*, not as the whole line: the header/footer
    band often catches the last line of body text as well, so the raw footer can
    read "every three years. Repair and/or SBN25.CHCSELDP.I.2018.LG.CO 5".
    Tokens are taken from the right while they still look like part of an
    identifier -- dotted, not sentence-final -- which keeps "RID25.One
    Pass.I.2018.LG.CO" intact while stopping before "Prosthetic Devices".
    """
    tokens = footer.strip().split()
    if len(tokens) < 2 or not tokens[-1].isdigit():
        return None
    number = int(tokens[-1])
    if not (1 <= number <= 9999):
        return None

    code_tokens: List[str] = []
    for token in reversed(tokens[:-1]):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._\-]*", token):
            break
        if token.endswith("."):          # sentence-final, not part of a code
            break
        if code_tokens and "." not in token:
            break                        # only dotted tokens extend a code
        code_tokens.insert(0, token)
        if len(code_tokens) >= 3:
            break

    code = " ".join(code_tokens).strip()
    if code.count(".") < MIN_FORM_DOTS:
        return None
    if not any(ch.isdigit() or ch.isupper() for ch in code):
        return None
    return code, number


def detect_form_sections(pdf, band: float = 0.12,
                         page_range: Optional[Tuple[int, int]] = None) -> FormLayout:
    """Split a bound compilation into its constituent form-coded documents.

    A single PDF may carry a medical schedule, a certificate, a drug schedule and
    several riders, each restarting at printed page 1. Page numbers alone cannot
    then identify a provision -- "Schedule of Benefits, p. 12" is ambiguous
    across four documents. The form code in the footer resolves it, and it is
    printed on every page.
    """
    lo, hi = page_range or (1, len(pdf.pages))
    layout = FormLayout()
    runs: List[FormSection] = []

    for pno in range(max(lo, 1), min(hi, len(pdf.pages)) + 1):
        page = pdf.pages[pno - 1]
        height = float(page.height)
        try:
            words = page.extract_words()
        except Exception:
            words = []
        footer = " ".join(w["text"] for w in words if w["bottom"] >= height * (1 - band))
        layout.pages_examined += 1
        parsed = _form_code_of(footer)
        try:
            page.flush_cache()
        except Exception:
            pass
        if not parsed:
            continue
        code, printed = parsed
        layout.pages_with_code += 1
        if runs and runs[-1].form_code == code and pno == runs[-1].pdf_end + 1:
            runs[-1].pdf_end = pno
            runs[-1].printed_end = printed
        else:
            runs.append(FormSection(code, pno, pno, printed, printed))

    layout.sections = runs
    return layout


# --------------------------------------------------------------------------- #
# Language-assistance pages
# --------------------------------------------------------------------------- #
#: Common English function words. Their share of a page separates English prose
#: from Spanish, Portuguese, Vietnamese or Haitian Creole, which share the Latin
#: alphabet and so cannot be told apart by script alone. Measured across three
#: real plan documents, English pages score 0.27-0.42 and translated pages
#: 0.03-0.08 -- a wide enough gap to threshold safely.
ENGLISH_FUNCTION_WORDS = frozenset("""
the of and to a in is are for you your that this be with or not as on at by
from will may we our if it any all no other than when which have has been
""".split())

WORD_RE = re.compile(r"[A-Za-z\u00C0-\u024F]+")


@dataclass
class PageLanguage:
    page: int
    words: int
    non_latin_ratio: float
    english_ratio: float
    foreign: bool
    reason: str = ""


def profile_page_language(text: str, page: int,
                          min_words: int = 40,
                          non_latin_threshold: float = 0.15,
                          english_threshold: float = 0.12) -> PageLanguage:
    """Judge whether a page is a translated notice rather than plan content.

    Two signals. A page carrying a substantial share of non-Latin characters is
    Chinese, Korean, Russian, Arabic and so on. A page in a Latin-script
    language instead shows almost no English function words.

    Short pages are never flagged: a heading or a table fragment can score low
    on both measures without being foreign, and skipping plan content is far
    worse than indexing a page of Spanish.
    """
    letters = [c for c in text if c.isalpha()]
    words = WORD_RE.findall(text.lower())
    if len(words) < min_words or not letters:
        return PageLanguage(page, len(words), 0.0, 1.0, False, "too short to judge")

    non_latin = sum(1 for c in letters if ord(c) > 0x02C0) / len(letters)
    english = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS) / len(words)

    if non_latin >= non_latin_threshold:
        return PageLanguage(page, len(words), round(non_latin, 3),
                            round(english, 3), True,
                            f"{non_latin:.0%} non-Latin characters")
    if english < english_threshold:
        return PageLanguage(page, len(words), round(non_latin, 3),
                            round(english, 3), True,
                            f"only {english:.0%} English function words")
    return PageLanguage(page, len(words), round(non_latin, 3),
                        round(english, 3), False, "")


def detect_foreign_pages(pdf, page_range: Optional[Tuple[int, int]] = None,
                         **kwargs) -> List[PageLanguage]:
    """Language-assistance pages within the converted range.

    Plan documents carry one or two pages of translated notices. Indexed, they
    produce chunks that match weakly against almost any query while carrying no
    plan content -- pure retrieval noise.
    """
    lo, hi = page_range or (1, len(pdf.pages))
    out: List[PageLanguage] = []
    for pno in range(max(lo, 1), min(hi, len(pdf.pages)) + 1):
        page = pdf.pages[pno - 1]
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        result = profile_page_language(text, pno, **kwargs)
        if result.foreign:
            out.append(result)
        try:
            page.flush_cache()
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- #
# Table of contents
# --------------------------------------------------------------------------- #
#: "About this Booklet ......... 1" and the variant without leaders,
#: "About this Booklet          1". Roman numerals are accepted because front
#: matter is often numbered that way.
TOC_LEADER_RE = re.compile(r"^\s*(.+?)\s*[\.\u2026\u00b7_-]{3,}\s*([ivxlcdm]+|\d{1,4})\s*$",
                           re.IGNORECASE)
TOC_PLAIN_RE = re.compile(r"^\s*(.{4,90}?)\s{2,}([ivxlcdm]+|\d{1,4})\s*$",
                          re.IGNORECASE)
TOC_TITLE_RE = re.compile(r"^\s*(table of contents|contents|what.s inside)\s*$",
                          re.IGNORECASE)


@dataclass
class TocPage:
    page: int
    entries: List[Tuple[str, str]] = field(default_factory=list)
    titled: bool = False


def _toc_entries(text: str) -> Tuple[List[Tuple[str, str]], bool]:
    entries: List[Tuple[str, str]] = []
    titled = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if TOC_TITLE_RE.match(line):
            titled = True
            continue
        m = TOC_LEADER_RE.match(line) or TOC_PLAIN_RE.match(line)
        if m:
            title = " ".join(m.group(1).split())
            if len(title) >= 4 and any(c.isalpha() for c in title):
                entries.append((title, m.group(2)))
    return entries, titled


def detect_toc_pages(pdf, search_limit: int = 30,
                     min_entries: int = 5) -> List[TocPage]:
    """Find the document's first table of contents.

    Returns the first contiguous run of pages that read as a contents list, so a
    later per-section index does not get mixed in. A page qualifies on the shape
    of its lines -- a title followed by a page number, with or without dot
    leaders -- rather than on a heading, since not every contents page is
    labelled.
    """
    found: List[TocPage] = []
    for pno in range(1, min(search_limit, len(pdf.pages)) + 1):
        page = pdf.pages[pno - 1]
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        entries, titled = _toc_entries(text)
        lines = [l for l in text.splitlines() if l.strip()]
        dense = bool(lines) and len(entries) / len(lines) >= 0.5
        if entries and (titled or (dense and len(entries) >= min_entries)):
            found.append(TocPage(pno, entries, titled))
        elif found:
            break          # the run has ended; keep only the first contents
        try:
            page.flush_cache()
        except Exception:
            pass
    return found
