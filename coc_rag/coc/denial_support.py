"""
Denial support: find the plan provision that governs a denied service.

Intended use
------------
A clinical reviewer supplies a procedure/diagnosis code for a service that a
claim adjudication engine denied. This module locates the plan language that
governs that service and returns it verbatim, with the section breadcrumb and
page, for inclusion in an adverse benefit determination.

Three design decisions worth understanding before you rely on this
---------------------------------------------------------------------------

1. CODE DESCRIPTORS ARE NEVER INFERRED.
   The descriptor for a procedure code must be supplied from your licensed code
   set. This module will not guess what a code means and no code descriptions
   are bundled. CPT is AMA-copyrighted and requires a license; ICD-10-CM is
   published by CMS/NCHS and is freely redistributable. Letting a language model
   recall a descriptor from memory is both a licensing problem and an accuracy
   problem -- a misremembered descriptor produces a confidently wrong citation.

2. "NO SUPPORTING PROVISION FOUND" IS A FIRST-CLASS RESULT.
   The workflow starts from a denial that already happened and looks for the
   language supporting it. That direction of reasoning invites post-hoc
   justification: a system that always returns something will eventually return
   something that does not actually fit. When nothing clears the confidence
   threshold this module says so, and that outcome should trigger review of the
   denial itself rather than a search for text that is merely close.

3. EXCEPTION CLAUSES ARE SURFACED, NOT BURIED.
   Plan exclusions routinely carry carve-backs -- "unless Medically Necessary",
   "except when provided as part of treatment for documented obstructive sleep
   apnea". An exclusion whose exception applies does not support a denial.
   Every returned provision has its exception clause extracted and flagged.

Scope note: not every prior authorization denial is an exclusion. Medical
necessity, missing clinical documentation, network status, and exhausted benefit
maximums are all common bases and live elsewhere in the document (or outside it
entirely). Searching only the exclusions section for a denial that rests on
medical necessity will find nothing -- correctly.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Code handling
# --------------------------------------------------------------------------- #
CPT_RE = re.compile(r"^\d{4}[0-9A-Z]$")           # 99213, 0510T
HCPCS_RE = re.compile(r"^[A-CEGHJ-MP-V]\d{4}$")   # J1745, E0601
ICD10_RE = re.compile(r"^[A-TV-Z]\d{2}(\.[0-9A-Z]{1,4})?$", re.I)  # E66.01, Z68.41


def classify_code(code: str) -> str:
    """Return 'cpt', 'hcpcs', 'icd10', or 'unknown' from format alone."""
    c = (code or "").strip().upper().replace(" ", "")
    if CPT_RE.match(c):
        return "cpt"
    if HCPCS_RE.match(c):
        return "hcpcs"
    if ICD10_RE.match(c):
        return "icd10"
    return "unknown"


def normalize_code(code: str) -> str:
    return (code or "").strip().upper().replace(" ", "")


@dataclass
class CodeEntry:
    code: str
    system: str
    descriptor: str
    synonyms: List[str] = field(default_factory=list)


class CodeResolver:
    """Descriptor lookup backed by a file you supply.

    Expected CSV columns: code, system, descriptor, synonyms (optional,
    pipe-separated). Point this at an export from your licensed CPT/HCPCS
    source, or at the free CMS ICD-10-CM release.
    """

    def __init__(self, entries: Optional[Dict[str, CodeEntry]] = None):
        self.entries: Dict[str, CodeEntry] = entries or {}

    @classmethod
    def _from_rows(cls, rows: Iterable[dict]) -> "CodeResolver":
        entries: Dict[str, CodeEntry] = {}
        for row in rows:
            code = normalize_code(row.get("code", ""))
            if not code:
                continue
            raw_syn = (row.get("synonyms") or "").strip()
            entries[code] = CodeEntry(
                code=code,
                system=(row.get("system") or classify_code(code)).lower(),
                descriptor=(row.get("descriptor") or "").strip(),
                synonyms=[s.strip() for s in raw_syn.split("|") if s.strip()],
            )
        return cls(entries)

    @classmethod
    def from_csv(cls, path: str) -> "CodeResolver":
        with open(path, newline="", encoding="utf-8-sig") as fh:
            return cls._from_rows(csv.DictReader(fh))

    @classmethod
    def from_bytes(cls, data: bytes) -> "CodeResolver":
        """Parse an uploaded CSV in memory.

        Avoids writing to a temp path, which is both unnecessary and
        platform-specific -- a hardcoded POSIX temp directory fails on Windows.
        """
        import io

        text = data.decode("utf-8-sig", errors="replace")
        return cls._from_rows(csv.DictReader(io.StringIO(text)))

    def get(self, code: str) -> Optional[CodeEntry]:
        return self.entries.get(normalize_code(code))

    def require(self, code: str) -> CodeEntry:
        entry = self.get(code)
        if entry is None or not entry.descriptor:
            raise LookupError(
                f"No descriptor loaded for {normalize_code(code)}. Supply it from your "
                f"licensed code set -- descriptors are never inferred."
            )
        return entry


# --------------------------------------------------------------------------- #
# Query construction
# --------------------------------------------------------------------------- #
STOPWORDS = {
    "the", "a", "an", "of", "or", "and", "for", "to", "in", "on", "with", "by",
    "at", "as", "is", "are", "be", "not", "any", "all", "other", "than", "when",
    "this", "that", "these", "those", "from", "each", "per", "may", "will",
    "unspecified", "without", "site", "encounter", "initial", "subsequent",
}

# Clinical shorthand worth expanding.
#
# Code-set vocabulary and plan vocabulary rarely overlap. HCPCS calls E0240 a
# "bath/shower chair"; the plan excludes "Chairs, bath chairs, feeding chairs".
# The shared word is "chair" -- so if a description reaches the search without
# it, only "bath" matches and the evidence is too thin to cite. Expansions
# bridge that gap.
#
# The terms below were read off the exclusion language of two real UnitedHealth
# plan documents rather than invented, so they map toward wording the documents
# actually use. Extend this with your own carriers' phrasing; it is the cheapest
# lever on recall in the whole pipeline.
TERM_EXPANSIONS: Dict[str, List[str]] = {
    # --- surgical / cosmetic ---
    "abdominoplasty": ["hanging skin", "plastic surgery", "cosmetic"],
    "panniculectomy": ["hanging skin", "abdominoplasty"],
    "brachioplasty": ["hanging skin", "plastic surgery"],
    "rhytidectomy": ["face lift", "cosmetic"],
    "hyperhidrosis": ["excessive sweating"],
    "rhinoplasty": ["cosmetic", "physical appearance"],
    "blepharoplasty": ["cosmetic", "eyelid"],
    "gastric": ["obesity", "bariatric"],
    "bariatric": ["obesity", "weight loss"],
    "obesity": ["weight loss", "bariatric"],
    "cosmetic": ["physical appearance", "reconstructive"],
    "varicose": ["physical appearance", "cosmetic"],
    "septoplasty": ["snoring", "sleep apnea"],
    "uvulopalatopharyngoplasty": ["snoring", "sleep apnea"],
    "palatopharyngoplasty": ["snoring", "sleep apnea"],

    # --- alternative / therapy ---
    "acupuncture": ["alternative treatments"],
    # A plan describes a service in its own words, not the code set's.
    # "hippotherapy" and "equestrian" appear nowhere in either test document;
    # the governing exclusion reads "animal-assisted therapy".
    "hippotherapy": ["animal-assisted therapy", "alternative treatment"],
    "equestrian": ["animal-assisted therapy", "alternative treatment"],
    "equine": ["animal-assisted therapy"],
    "aquatic": ["alternative treatment", "recreational therapy"],
    "recreational": ["recreational therapy", "alternative treatment"],
    "massage": ["massage therapy", "alternative treatments"],
    "biofeedback": ["alternative treatments"],
    "naturopath": ["alternative treatments"],
    "homeopath": ["alternative treatments"],
    "hypnosis": ["alternative treatments", "hypnotherapy"],
    "art": ["art therapy", "alternative treatment"],
    "music": ["music therapy", "alternative treatment"],
    "dance": ["dance therapy", "alternative treatment"],
    "chiropractic": ["manipulative treatment"],
    "infertility": ["reproduction", "fertility"],
    "fertility": ["reproduction", "infertility"],
    "orthodontic": ["dental", "teeth"],
    "custodial": ["personal care", "custodial care"],
    "experimental": ["investigational", "unproven"],
    "investigational": ["experimental", "unproven"],

    # --- DME: seating, mobility, transfer ---
    # "chair" is the pivot: the plan excludes "Chairs, bath chairs, feeding
    # chairs, toddler chairs, chair lifts and recliners", so any seating device
    # needs that word to reach the right line.
    "bath": ["bath chairs", "chairs"],
    "shower": ["bath chairs", "chairs"],
    "commode": ["chairs", "toilet"],
    "recliner": ["recliners", "chairs"],
    "stairlift": ["chair lifts", "elevators", "home modifications"],
    "wheelchair": ["mobility device", "manual wheelchair", "electric wheelchair",
                   "transfer chair", "scooter"],
    "scooter": ["mobility device", "wheelchair"],
    "walker": ["mobility device"],
    "seat": ["chairs", "seat lift"],
    "rail": ["handrails", "home modifications"],
    "grab": ["handrails", "home modifications"],

    # --- DME: home and environment ---
    "ramp": ["home modifications", "handrails", "elevators"],
    "elevator": ["home modifications", "elevators", "chair lifts"],
    "humidifier": ["air purifiers", "dehumidifiers"],
    "dehumidifier": ["air conditioners", "air purifiers"],
    "purifier": ["air conditioners", "air purifiers", "filters"],
    "conditioner": ["air conditioners"],
    "whirlpool": ["hot tubs"],
    "tub": ["hot tubs"],
    "treadmill": ["exercise equipment"],
    "exercise": ["exercise equipment"],
    "mattress": ["beds"],

    # --- DME: respiratory, monitoring, supplies ---
    "cpap": ["tubings and masks", "sleep apnea"],
    "nebulizer": ["tubings and masks"],
    "mask": ["tubings and masks"],
    "tubing": ["tubings and masks"],
    "monitor": ["monitoring equipment", "publicly available devices"],
    "glucose": ["continuous glucose monitors", "diabetic supplies"],
    "diabetic": ["diabetic supplies"],
    "catheter": ["urinary catheters", "urologic supplies"],
    "ostomy": ["ostomy supplies"],
    "dressing": ["gauze and dressings"],
    "gauze": ["gauze and dressings"],
    "compress": ["hot and cold compresses"],

    # --- DME: orthotics, prosthetics, hearing ---
    "orthotic": ["shoe orthotics", "orthotics"],
    "orthosis": ["orthotics", "brace"],
    "brace": ["orthotics"],
    "prosthesis": ["prosthetic devices", "prosthetics"],
    "prosthetic": ["prosthetic devices", "prosthetics"],
    "shoe": ["shoe orthotics", "orthotics"],
    "hearing": ["hearing aid", "bone anchored hearing aid"],
    "cochlear": ["hearing aid", "bone anchored hearing aid"],
    "wig": ["wigs", "hair replacement"],

    # --- setting / level of care ---
    "respite": ["respite care", "hospice"],
    "lodging": ["travel", "lodging"],
}


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(t) > 2 and t not in STOPWORDS]


def build_core_terms(descriptors: Sequence[str],
                     extra_terms: Sequence[str] = ()) -> List[str]:
    """Terms the user actually supplied: the descriptor and any synonyms typed
    in. These carry the evidential weight."""
    terms: List[str] = []
    for d in descriptors:
        terms.extend(tokenize(d))
    for t in extra_terms:
        terms.extend(tokenize(t))
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_query_terms(descriptors: Sequence[str],
                      extra_terms: Sequence[str] = ()) -> List[str]:
    """Core terms plus table-driven expansions.

    Expansions widen recall but must not, on their own, justify a citation.
    Mapping "nebulizer" toward "durable medical equipment" helps surface the
    right region of the document, but those words also appear in every DME
    definition and boilerplate sentence -- so a provision matching only
    expansion terms is a lead to follow, not governing language. `assess`
    enforces that by requiring a core-term match before reporting support.
    """
    core = build_core_terms(descriptors, extra_terms)
    terms = list(core)
    seen = set(core)
    for t in core:
        for exp in TERM_EXPANSIONS.get(t, []):
            for token in tokenize(exp):
                if token not in seen:
                    seen.add(token)
                    terms.append(token)
    return terms


# --------------------------------------------------------------------------- #
# Provision model
# --------------------------------------------------------------------------- #
EXCEPTION_RE = re.compile(
    r"\b(unless|except(?:\s+(?:as|when|for|to the extent))?|other than|"
    r"provided that|does not apply)\b", re.IGNORECASE
)
CROSSREF_RE = re.compile(
    r"\b(Section\s+\d+[^.,;]*|Appendix\s+[A-Z][^.,;]*|Schedule of Benefits)", re.IGNORECASE
)


#: Qualifier conflicts: (pattern in the plan provision, pattern in the service
#: descriptor). A hit means the provision governs the *opposite* of the service
#: performed -- "Non-surgical treatment of obesity" does not support denying a
#: sleeve gastrectomy. Term-overlap scoring cannot see this on its own, because
#: the conflicting provision shares almost all its vocabulary with the service.
QUALIFIER_CONFLICTS: List[Tuple[str, str, str]] = [
    (r"\bnon-?surgical\b",
     r"\b(surgery|surgical|surgic|ectomy|plasty|otomy|oscopy|laparoscop|excision|resection|implant)\b",
     "provision addresses NON-surgical treatment; the service is surgical"),
    (r"\bnon-?prescription\b|\bover-the-counter\b",
     r"\bprescription\b",
     "provision addresses non-prescription items; the service is a prescription"),
    (r"\boral appliance",
     r"\b(surgery|surgical|plasty|ectomy)\b",
     "provision addresses an oral appliance; the service is a surgical procedure"),
    (r"\boutpatient only\b",
     r"\binpatient\b",
     "provision is limited to outpatient; the service is inpatient"),
    (r"\bcosmetic\b",
     r"\breconstructive\b",
     "provision addresses cosmetic services; the service is described as reconstructive"),
]


def detect_conflicts(provision_text: str, descriptors: Sequence[str]) -> List[str]:
    desc = " ".join(descriptors).lower()
    prov = (provision_text or "").lower()
    out: List[str] = []
    for prov_pat, desc_pat, message in QUALIFIER_CONFLICTS:
        if re.search(prov_pat, prov) and re.search(desc_pat, desc):
            out.append(message)
    return out


@dataclass
class Provision:
    """One candidate plan provision, quoted verbatim."""
    text: str                       # the precise sentence / bullet
    breadcrumb: str
    section_path: List[str]
    page_start: int
    page_end: int
    score: float
    matched_terms: List[str] = field(default_factory=list)
    context: str = ""               # surrounding paragraph, for the reviewer
    exception_clause: Optional[str] = None
    cross_references: List[str] = field(default_factory=list)
    source_type: str = "exclusion"  # exclusion | limitation | definition | benefit
    distinctive_terms: List[str] = field(default_factory=list)
    core_matched: List[str] = field(default_factory=list)
    #: Cross-encoder relevance, 0-1, when reranking ran. Ordering only: it never
    #: overrides a gate.
    rerank_score: Optional[float] = None
    lexical_rank: Optional[int] = None
    #: The operative text above this line: the section lead-in and any parent
    #: list item. A nested exclusion often reads as a bare list of nouns; the
    #: parent supplies the verb that makes it a denial basis.
    lead_in: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    form_code: Optional[str] = None
    printed_page_start: Optional[int] = None
    printed_page_end: Optional[int] = None

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicts)

    @property
    def citation(self) -> str:
        """Reference as it should appear in a letter.

        Where the PDF binds several form-coded documents together, the form code
        is part of the citation -- four documents in one compilation each have a
        page 1, so the page number alone is not a reference."""
        if self.printed_page_start:
            pages = (f"p. {self.printed_page_start}"
                     if self.printed_page_start == self.printed_page_end
                     else f"pp. {self.printed_page_start}-{self.printed_page_end}")
        else:
            pages = (f"p. {self.page_start}" if self.page_start == self.page_end
                     else f"pp. {self.page_start}-{self.page_end}")
        form = f"{self.form_code}, " if self.form_code else ""
        return f"{self.breadcrumb} ({form}{pages})"

    @property
    def has_exception(self) -> bool:
        return bool(self.exception_clause)

    @property
    def quoted_provision(self) -> str:
        """Full text as it should appear in a letter: context chain, then the
        specific line."""
        if not self.lead_in:
            return self.text
        chain = "\n".join(f"{'  ' * i}{part}" for i, part in enumerate(self.lead_in))
        return f"{chain}\n{'  ' * len(self.lead_in)}{self.text}"


@dataclass
class DenialSupport:
    """Result of a provision search. `outcome` drives what happens next."""
    codes: List[str]
    descriptors: List[str]
    query_terms: List[str]
    provisions: List[Provision]
    outcome: str                    # supported | weak | not_found
    notes: List[str] = field(default_factory=list)

    @property
    def needs_human_review(self) -> bool:
        return (self.outcome != "supported"
                or any(p.has_exception for p in self.provisions[:1]))


# --------------------------------------------------------------------------- #
# Sentence-level extraction
# --------------------------------------------------------------------------- #
SENT_SPLIT_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


def split_sentences(text: str) -> List[str]:
    """Split a block into citable units. Bullets are already discrete
    provisions, so they are kept whole rather than split further."""
    out: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("-", "*", "\u2022")):
            out.append(stripped.lstrip("-*\u2022 ").strip())
            continue
        for part in SENT_SPLIT_RE.split(stripped):
            part = part.strip()
            if len(part) > 15:
                out.append(part)
    return out


def score_sentence(sentence: str, terms: Sequence[str],
                   idf: Optional[Dict[str, float]] = None) -> Tuple[float, List[str]]:
    """Term overlap weighted by inverse document frequency, length-normalised.

    Deliberately transparent rather than clever: a reviewer must be able to see
    why a provision was surfaced.
    """
    tokens = set(tokenize(sentence))
    if not tokens or not terms:
        return 0.0, []
    matched = [t for t in terms if t in tokens]
    if not matched:
        return 0.0, []
    weight = sum((idf or {}).get(t, 1.0) for t in matched)
    coverage = len(matched) / max(len(set(terms)), 1)
    brevity = 1.0 / (1.0 + math.log1p(max(len(tokens) - 12, 0)))
    return round(weight * (0.5 + coverage) * brevity, 4), matched


def extract_exception(sentence: str) -> Optional[str]:
    m = EXCEPTION_RE.search(sentence)
    if not m:
        return None
    return sentence[m.start():].strip().rstrip(".") or None


# --------------------------------------------------------------------------- #
# Scope: where denial-relevant language lives
# --------------------------------------------------------------------------- #
#: Denials are not only supported by the exclusions section. Benefit limits sit
#: in the Schedule of Benefits, and terms like "Experimental or Investigational"
#: are governed by their definition in the defined-terms section.
SCOPE_PATTERNS: Dict[str, List[str]] = {
    "exclusion":  [r"exclusion", r"not covered", r"limitations"],
    "limitation": [r"schedule of benefits"],
    "definition": [r"defined terms"],
}
DEFAULT_SCOPES = ("exclusion", "limitation", "definition")

#: Language that actually expresses a benefit limit. The Schedule of Benefits is
#: mostly narrative -- eligibility, network rules, how to file, programme
#: descriptions -- and on a real SPD only 5 of 193 sentences in that section
#: state a limit. Without this gate the other 188 compete with genuine
#: exclusions, which is how a query about bath chairs surfaced "The Plan offers
#: a Medicare Crossover program ... for Durable Medical Equipment (DME) claims".
LIMIT_LANGUAGE_RE = re.compile(
    r"(limited to|limit(s)? (of|to|apply)|maximum|max\.|not to exceed|no more than|"
    r"up to \d|per (calendar |plan )?year|per (visit|day|admission|lifetime)|"
    r"\b\d+\s*(visits?|days?|treatments?|sessions?|hours?)\b|"
    r"benefit limit|combined limit|does not (include|apply to)|"
    r"only (covered|available|payable)|no benefits)", re.IGNORECASE)

#: Process and programme narration. Describes how the plan operates rather than
#: what it covers, so it can never be the basis for a denial.
ADMINISTRATIVE_RE = re.compile(
    r"^(the plan (offers|provides|has|uses|may)|we (offer|provide|will send|may contact)|"
    r"you (may|can|should|must) (call|contact|write|visit|register|log)|"
    r"to (register|enroll|contact|reach)|for more information|"
    r"this (program|programme|service) )", re.IGNORECASE)


def is_citable_provision(scope: str, sentence: str) -> bool:
    """Could this sentence be the stated basis for a denial?

    Exclusions and definitions qualify by virtue of where they sit. Sentences
    drawn from a Schedule of Benefits must additionally express an actual limit:
    that section is overwhelmingly narrative, and its prose otherwise crowds out
    the exclusions that do govern.
    """
    if ADMINISTRATIVE_RE.match(sentence.strip()):
        return False
    if scope == "limitation":
        return bool(LIMIT_LANGUAGE_RE.search(sentence))
    return True


#: Sections that explain how to *read* the exclusions rather than stating any.
#: Their text is full of exclusion vocabulary and matches strongly, so it must
#: be excluded from citable units or it will be quoted in a denial letter.
#: Wording varies between carriers -- "How Are Headings Used in this Section?"
#: and "How Do We Use Headings in this Section?" are the same section. Match the
#: shape rather than an exact phrase.
META_HEADING_RE = re.compile(
    r"(how\s+\w+\s+\w+\s+use\s+headings|how\s+are\s+headings\s+used|"
    r"headings?\s+(are\s+)?used\s+(in|to)|where\s+are\s+benefit\s+limitations\s+shown|"
    r"plan\s+does\s+not\s+pay\s+benefits\s+for\s+exclusions|"
    r"how\s+do\s+(you|we)\s+use\s+this|about\s+this\s+booklet|table\s+of\s+contents)",
    re.IGNORECASE)

#: A citable provision is a sentence, not a fragment -- but shortness alone is a
#: poor test. Real exclusions are often two words ("Hot tubs.", "Car seats.",
#: "Television."), while a cross-reference tail like "Alternative Treatments
#: below)." is longer. Length sets only a floor; shape does the real filtering.
MIN_PROVISION_CHARS = 8

#: Mid-sentence debris: opens lower-case, or closes a parenthesis it never
#: opened. Either means the split landed inside a sentence.
FRAGMENT_RE = re.compile(r"^[a-z]")


def is_fragment(sentence: str) -> bool:
    text = sentence.strip()
    if len(text) < MIN_PROVISION_CHARS:
        return True
    if FRAGMENT_RE.match(text):
        return True
    if text.count(")") > text.count("("):
        return True
    return False

#: Self-referential prose that describes the document instead of stating a rule.
META_SENTENCE_RE = re.compile(
    r"(this section contains|to help you find|are used (only )?to help|"
    r"for example[,]? [A-Z]|please review|as (described|shown|stated) (in|under)\s+Section|"
    r"refer to)", re.IGNORECASE)


#: Language that makes a sentence a rule rather than a description of the
#: document. Bullets in an exclusions list are rules even when the heading above
#: them reads like navigation.
RULE_LIKE_RE = re.compile(
    r"(\b(not (a )?covered|are not|is not|excluded?|exclusion|no benefits|"
    r"will not pay|does not (cover|pay|apply)|we do not|is limited to|"
    r"limited to)\b|^\s*[-\u2022]?\s*(removal|treatment|services|"
    r"procedures|surgery|drugs|devices|items|supplies|charges|therapy)\b)",
    re.IGNORECASE)


def is_meta_unit(breadcrumb: str, sentence: str) -> bool:
    """Is this sentence about the document rather than a rule in it?

    Judged from the sentence alone. Heading text turned out to be an unreliable
    signal: when sub-headings inside an exclusions section are not detected, the
    bullets are parented to whatever heading preceded them, so "Where Are
    Benefit Limitations Shown?" ends up sitting above genuine exclusions. Two
    versions of this function excluded content by heading and silently dropped
    the hanging-skin and excessive-sweating exclusions as a result.

    A sentence that states a rule is always kept; otherwise self-referential
    phrasing ("this section contains", "to help you find") marks it as
    navigation.
    """
    if RULE_LIKE_RE.search(sentence):
        return False
    return bool(META_SENTENCE_RE.search(sentence))


#: Components that name a division of the document rather than a topic inside one.
DIVISION_RE = re.compile(
    r"^\s*(section|appendix|schedule|part|article|exhibit|rider|addendum)\b", re.I)


def scope_of(breadcrumb: str) -> Optional[str]:
    """Which searchable category this breadcrumb belongs to, if any.

    Scope is decided by the *deepest division* the breadcrumb names -- the last
    "Section N" / "Appendix X" / "Schedule of ..." component -- not by matching
    keywords anywhere in the path.

    The difference matters when heading levels come out wrong. A conversion that
    nests "Appendix A: Clinical Programs and Resources" underneath "Section 9:
    Defined Terms" would, under a loose match, give every wellness-programme
    page the "definition" scope, flooding the candidate pool and pushing the
    real exclusions out of the results. Judging by the deepest division puts
    Appendix A out of scope where it belongs, even while the nesting is wrong.
    """
    components = [c.strip() for c in (breadcrumb or "").split(">") if c.strip()]
    division = None
    for component in components:
        low = component.lower()
        names_division = bool(DIVISION_RE.match(component)) or any(
            re.search(p, low) for patterns in SCOPE_PATTERNS.values() for p in patterns
        )
        if names_division:
            division = low

    target = division if division is not None else (breadcrumb or "").lower()
    for scope, patterns in SCOPE_PATTERNS.items():
        if any(re.search(p, target) for p in patterns):
            return scope
    return None


# --------------------------------------------------------------------------- #
# Markdown-backed index (no database required)
# --------------------------------------------------------------------------- #
class MarkdownProvisionIndex:
    """Lexical provision search straight off the converted Markdown.

    Exists so this workflow can be exercised and audited before anything is
    embedded, and so results stay explainable: every hit traces to specific
    matched terms rather than to a similarity score.
    """

    def __init__(self, markdown: str, scopes: Sequence[str] = DEFAULT_SCOPES):
        from .chunker import _pipe_row_sentences, parse_markdown

        blocks, _ = parse_markdown(markdown)
        self.units: List[dict] = []
        df: Counter = Counter()

        # Track the nesting context so a leaf can carry its parents.
        intro: str = ""
        #: Depth of the first item that followed the intro. An item shallower
        #: than this starts a new group, so the intro no longer applies. Without
        #: it, "The following infertility treatment-related services:" would be
        #: attached to the sibling exclusions that follow its sub-list -- naming
        #: gestational-carrier costs as infertility treatment, which is wrong.
        intro_depth: Optional[int] = None
        parents: Dict[int, str] = {}

        for blk in blocks:
            breadcrumb = " > ".join(blk.section_path)

            if blk.kind == "table":
                # Real benefit limits live in the Schedule of Benefits grid
                # ("Limited to 10 visits per calendar year"), not in its prose.
                # Index only rows that state a limit, carrying the column labels
                # so the row reads as a rule rather than a fragment.
                scope = scope_of(breadcrumb)
                if scope is None or scope not in scopes:
                    continue
                rows = _pipe_row_sentences(blk.text).splitlines()
                for row in rows:
                    if not LIMIT_LANGUAGE_RE.search(row):
                        continue
                    toks = set(tokenize(row))
                    if not toks:
                        continue
                    df.update(toks)
                    self.units.append({
                        "lead_in": [],
                        "text": " ".join(row.split()),
                        "breadcrumb": breadcrumb,
                        "section_path": list(blk.section_path),
                        "page_start": blk.page_start,
                        "page_end": blk.page_end,
                        "scope": "limitation",
                        "context": blk.text[:1200],
                        "form_code": blk.form_code,
                        "printed_start": blk.printed_start,
                        "printed_end": blk.printed_end,
                    })
                continue

            if blk.kind != "para":
                continue
            scope = scope_of(breadcrumb)
            if scope is None or scope not in scopes:
                intro, intro_depth, parents = "", None, {}
                continue

            first_line = blk.text.splitlines()[0] if blk.text else ""
            stripped = first_line.lstrip()
            depth = (len(first_line) - len(stripped)) // 2
            is_item = stripped.startswith(("- ", "* "))

            if not is_item:
                # A lead-in ends with a colon ("The following are not Covered
                # Health Care Services:"); ordinary prose resets the context.
                intro = " ".join(blk.text.split()) if blk.text.rstrip().endswith(":") else ""
                intro_depth = None
                parents = {}
            else:
                item_text = " ".join(stripped[2:].split())
                if intro:
                    if intro_depth is None:
                        intro_depth = depth
                    elif depth < intro_depth:
                        intro, intro_depth = "", None
                parents = {d: t for d, t in parents.items() if d < depth}
                parents[depth] = item_text
            chain: List[str] = []
            if intro:
                chain.append(intro)
            chain.extend(parents[d] for d in sorted(parents) if d < depth)

            for sent in split_sentences(blk.text):
                toks = set(tokenize(sent))
                if (not toks or is_fragment(sent) or is_meta_unit(breadcrumb, sent)
                        or not is_citable_provision(scope, sent)):
                    continue
                df.update(toks)
                self.units.append({
                    "lead_in": list(chain),
                    "text": sent,
                    "breadcrumb": breadcrumb,
                    "section_path": list(blk.section_path),
                    "page_start": blk.page_start,
                    "page_end": blk.page_end,
                    "scope": scope,
                    "context": blk.text,
                    "form_code": blk.form_code,
                    "printed_start": blk.printed_start,
                    "printed_end": blk.printed_end,
                })

        n = max(len(self.units), 1)
        self.idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}

    #: IDF above which a matched term counts as clinically distinctive rather
    #: than boilerplate that appears throughout the document.
    DISTINCTIVE_IDF = 4.0

    def search(self, terms: Sequence[str], limit: int = 8,
               min_score: float = 0.0,
               descriptors: Sequence[str] = (),
               core_terms: Sequence[str] = ()) -> List[Provision]:
        scored: List[Provision] = []
        for u in self.units:
            score, matched = score_sentence(u["text"], terms, self.idf)
            if score <= min_score:
                continue
            distinctive = [t for t in matched
                           if self.idf.get(t, 0.0) >= self.DISTINCTIVE_IDF]
            core_set = set(core_terms) if core_terms else set(terms)
            scored.append(Provision(
                distinctive_terms=distinctive,
                core_matched=[t for t in matched if t in core_set],
                lead_in=list(u.get("lead_in") or []),
                conflicts=detect_conflicts(u["text"], descriptors),
                text=u["text"],
                breadcrumb=u["breadcrumb"],
                section_path=u["section_path"],
                page_start=u["page_start"],
                page_end=u["page_end"],
                score=score,
                matched_terms=matched,
                context=u["context"],
                exception_clause=extract_exception(u["text"]),
                cross_references=[m.group(0).strip()
                                  for m in CROSSREF_RE.finditer(u["text"])],
                source_type=u["scope"],
                form_code=u.get("form_code"),
                printed_page_start=u.get("printed_start"),
                printed_page_end=u.get("printed_end"),
            ))
        # A definition explains how an exclusion applies; it is rarely itself
        # the basis for a denial. Damping keeps "Durable Medical Equipment -
        # medical equipment that is all of the following" from outranking the
        # exclusion that actually governs.
        for prov in scored:
            if prov.source_type == "definition":
                prov.score = round(prov.score * DEFINITION_WEIGHT, 4)
        scored.sort(key=lambda p: p.score, reverse=True)
        return scored[:limit]


# --------------------------------------------------------------------------- #
# Atlas-backed search (semantic recall, then sentence-level precision)
# --------------------------------------------------------------------------- #
#: Corpus IDF per (collection, document). Built once per process.
_CORPUS_IDF_CACHE: Dict[str, Dict[str, float]] = {}


def corpus_idf(coll, doc_id: Optional[str] = None,
               scopes: Sequence[str] = DEFAULT_SCOPES,
               max_chunks: int = 2000) -> Dict[str, float]:
    """Term rarity across the whole document, not the retrieved candidates.

    Deriving IDF from the result set inverts the signal it is meant to carry.
    Vector search returns semantically similar passages by construction, so the
    query's own terms are common *within* that set: searching "bath chair"
    returns chunks about bath chairs, "bath" then looks unremarkable, and the
    distinctiveness gate rejects the correct provision. The better the
    retrieval, the worse the effect.

    Measured against the corpus, "bath" is rare and the gate behaves as intended
    -- and matches what the Markdown path computes, so the two paths agree.
    """
    key = f"{getattr(coll, 'full_name', 'coll')}::{doc_id or '*'}"
    cached = _CORPUS_IDF_CACHE.get(key)
    if cached is not None:
        return cached

    query: Dict = {"doc_id": doc_id} if doc_id else {}
    df: Counter = Counter()
    sentences = 0
    try:
        cursor = coll.find(query, {"_id": 0, "text": 1, "breadcrumb": 1}).limit(max_chunks)
        for row in cursor:
            scope = scope_of(row.get("breadcrumb", ""))
            if scope is None or scope not in scopes:
                continue
            for sent in split_sentences(row.get("text") or ""):
                toks = set(tokenize(sent))
                if toks:
                    df.update(toks)
                    sentences += 1
    except Exception:
        return {}

    n = max(sentences, 1)
    idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}
    _CORPUS_IDF_CACHE[key] = idf
    return idf


def scoped_section_values(coll, scopes: Sequence[str] = DEFAULT_SCOPES) -> List[str]:
    """Section-path values whose name matches one of the requested scopes.

    Used to build a pre-filter for $vectorSearch. Without it the search has to
    retrieve broadly and discard afterwards, and exclusions are a small slice of
    a plan document -- around 12% of chunks in a typical SPD -- so the top
    semantic hits are routinely all definitions and general prose, leaving
    nothing in scope to score. That reads as "no supporting provision found"
    when the provision is sitting right there in the index.
    """
    try:
        values = coll.distinct("section_path")
    except Exception:
        return []
    wanted = []
    for value in values:
        if not isinstance(value, str):
            continue
        low = value.lower()
        for scope in scopes:
            if any(re.search(p, low) for p in SCOPE_PATTERNS.get(scope, [])):
                wanted.append(value)
                break
    return wanted


def search_provisions_vector(coll, embedder, query: str, terms: Sequence[str],
                             doc_id: Optional[str] = None, k: int = 40,
                             scopes: Sequence[str] = DEFAULT_SCOPES,
                             limit: int = 8,
                             descriptors: Sequence[str] = (),
                             core_terms: Sequence[str] = (),
                             idf_out: Optional[Dict[str, float]] = None,
                             ) -> List[Provision]:
    """Retrieve candidate chunks semantically, then pick the precise sentence.

    Chunk-level recall alone is not enough here: a denial letter must quote the
    governing sentence, not a 650-token passage that happens to contain it.

    Scope is applied as a pre-filter inside $vectorSearch where possible, so the
    k results returned are already exclusions/limits/definitions rather than
    whatever the whole document offered.
    """
    from .vectorstore import vector_search

    clauses: List[Dict] = []
    if doc_id:
        clauses.append({"doc_id": {"$eq": doc_id}})
    sections = scoped_section_values(coll, scopes)
    if sections:
        clauses.append({"section_path": {"$in": sections}})

    if len(clauses) > 1:
        filters: Optional[Dict] = {"$and": clauses}
    elif clauses:
        filters = clauses[0]
    else:
        filters = None

    hits = vector_search(coll, embedder.embed_query(query), k=k, filters=filters)
    if not hits and filters is not None:
        # Pre-filter matched nothing (unusual section naming); fall back to a
        # broad search and filter afterwards rather than returning nothing.
        hits = vector_search(coll, embedder.embed_query(query), k=max(k, 60),
                             filters={"doc_id": {"$eq": doc_id}} if doc_id else None)

    # Split every retrieved chunk into sentences first, then derive an IDF over
    # that pool. Without it this path scored each matched term at weight 1.0
    # while the Markdown path weighted by rarity -- roughly a twelvefold
    # difference on the same provision, against thresholds calibrated for the
    # weighted scale. The correct exclusion would surface as the top candidate
    # and still be reported as "no supporting provision found".
    candidates: List[tuple] = []
    sentence_df: Counter = Counter()
    for hit in hits:
        breadcrumb = hit.get("breadcrumb", "")
        scope = scope_of(breadcrumb)
        if scope is None or scope not in scopes:
            continue
        body = hit.get("text") or ""
        for sent in split_sentences(body):
            if is_fragment(sent) or is_meta_unit(breadcrumb, sent):
                continue
            if not is_citable_provision(scope, sent):
                continue
            toks = set(tokenize(sent))
            if not toks:
                continue
            sentence_df.update(toks)
            candidates.append((hit, breadcrumb, scope, sent))

    # Corpus IDF where available; the candidate pool only as a fallback.
    idf = corpus_idf(coll, doc_id, scopes)
    if not idf:
        n_sentences = max(len(candidates), 1)
        idf = {t: math.log(1 + n_sentences / (1 + c)) for t, c in sentence_df.items()}
    if idf_out is not None:
        idf_out.clear()
        idf_out.update(idf)
    distinctive_floor = 4.0
    core_set = set(core_terms) if core_terms else set(terms)

    out: List[Provision] = []
    for hit, breadcrumb, scope, sent in candidates:
            body = hit.get("text") or ""
            score, matched = score_sentence(sent, terms, idf)
            if score <= 0:
                continue
            out.append(Provision(
                core_matched=[t for t in matched if t in core_set],
                distinctive_terms=[t for t in matched
                                   if idf.get(t, 0.0) >= distinctive_floor],
                conflicts=detect_conflicts(sent, descriptors),
                text=sent,
                breadcrumb=breadcrumb,
                section_path=hit.get("section_path", []),
                page_start=int(hit.get("page_start") or 0),
                page_end=int(hit.get("page_end") or 0),
                score=round(score * (0.5 + float(hit.get("score") or 0)), 4),
                matched_terms=matched,
                context=body,
                exception_clause=extract_exception(sent),
                cross_references=[m.group(0).strip()
                                  for m in CROSSREF_RE.finditer(sent)],
                source_type=scope,
                form_code=hit.get("form_code"),
                printed_page_start=hit.get("printed_page_start"),
                printed_page_end=hit.get("printed_page_end"),
            ))
    out.sort(key=lambda p: p.score, reverse=True)
    return out[:limit]


# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #
def rerank_provisions(provisions: Sequence[Provision], query: str,
                      api_key: Optional[str] = None,
                      model: str = "rerank-2.5",
                      top_n: int = 8) -> List[Provision]:
    """Reorder candidates with a cross-encoder.

    Ordering only, and deliberately so. A reranker always returns a ranking: ask
    it about a routinely covered service and it will still nominate a
    best-of-a-bad-set provision, which is exactly the post-hoc justification the
    gates exist to prevent. So the lexical evidence -- which terms matched, how
    distinctive they were, whether the provision contradicts the service -- is
    left untouched and still decides the outcome. Reranking only changes which
    qualifying provision a reviewer sees first.

    It earns its place on vocabulary mismatch, where lexical scoring is weakest:
    a description reading "bath support device" shares one word with "Chairs,
    bath chairs, feeding chairs", and a cross-encoder can tell they are the same
    subject where term overlap cannot.

    Fails soft: any error returns the original order.
    """
    if not provisions or len(provisions) < 2:
        return list(provisions)

    ranked = list(provisions)
    for position, prov in enumerate(ranked, start=1):
        prov.lexical_rank = position

    try:
        import voyageai

        client = voyageai.Client(api_key=api_key or os.environ.get("VOYAGE_API_KEY"))
        documents = [p.quoted_provision[:4000] for p in ranked]
        response = client.rerank(query, documents, model=model,
                                 top_k=min(top_n, len(documents)))
        reordered: List[Provision] = []
        for result in response.results:
            prov = ranked[result.index]
            prov.rerank_score = round(float(result.relevance_score), 4)
            reordered.append(prov)
        remaining = [p for p in ranked if p not in reordered]
        return reordered + remaining
    except Exception:
        return ranked


def rerank_disagreement(provisions: Sequence[Provision]) -> Optional[str]:
    """Flag when the reranker and the term evidence disagree about the top hit.

    Disagreement is informative rather than an error: it usually means the
    wording differs from the plan's, which is worth a reviewer's attention
    either way.
    """
    if not provisions or provisions[0].rerank_score is None:
        return None
    top = provisions[0]
    if top.lexical_rank and top.lexical_rank > 2:
        return (f"The reranker promoted a provision that term matching ranked "
                f"#{top.lexical_rank}. The wording likely differs from the "
                "plan's; confirm it genuinely governs this service.")
    return None


import os  # noqa: E402  (used by rerank_provisions)


# --------------------------------------------------------------------------- #
# Assessment
# --------------------------------------------------------------------------- #
SUPPORTED_THRESHOLD = 6.0
WEAK_THRESHOLD = 2.5
#: A provision must match at least this many terms, at least one of them
#: distinctive, before it can be called supporting. Without this gate a
#: routinely covered service matches an unrelated provision on generic words
#: like "replacement" and is wrongly reported as excluded.
MIN_MATCHED_TERMS = 2
MIN_DISTINCTIVE_TERMS = 1
#: Definitions are damped rather than excluded: they belong in a letter
#: alongside an exclusion, but not in place of one.
DEFINITION_WEIGHT = 0.45

#: A single matched term can still be worth showing, but only when it is rare
#: enough to be clinically specific. Measured on a real SPD: "bath" scores 5.94
#: and identifies the right exclusion, while "replacement" scores 4.56 and
#: matches a dental provision for a knee arthroplasty query. The bar sits above
#: the ordinary distinctiveness threshold for exactly that reason.
HIGHLY_DISTINCTIVE_IDF = 5.5


def assess(codes: Sequence[str], descriptors: Sequence[str],
           terms: Sequence[str], provisions: Sequence[Provision],
           supported_threshold: float = SUPPORTED_THRESHOLD,
           weak_threshold: float = WEAK_THRESHOLD,
           idf: Optional[Dict[str, float]] = None) -> DenialSupport:
    """Decide whether the retrieved language actually supports a denial.

    Three outcomes, and only one of them is a green light. `not_found` is not a
    failure of the search -- it is evidence that the denial may rest on
    something other than plan language (medical necessity, documentation,
    network status), or that it warrants reconsideration.
    """
    notes: List[str] = []
    idf_lookup = idf or {}
    top_p = provisions[0] if provisions else None
    top = top_p.score if top_p else 0.0

    gated = False
    thin_but_distinctive = False
    #: Set when a provision must not be surfaced as usable at all, however well
    #: it scores -- currently only a qualifier conflict, where the provision
    #: governs the opposite of the service performed.
    blocking = False
    #: Every match came from the expansion table. Never enough for "supported",
    #: but withholding it entirely hides correct answers: a plan may describe a
    #: service purely in its own vocabulary, as with hippotherapy appearing only
    #: as "animal-assisted therapy".
    expansion_only = False
    if top_p is not None:
        if len(top_p.matched_terms) < MIN_MATCHED_TERMS:
            gated = True
            # One match on a rare, clinically specific term is weak evidence,
            # not no evidence. "bath" appearing in a short exclusion listing
            # bath chairs is worth showing a reviewer; reporting nothing hides a
            # correct top-ranked hit behind a threshold.
            thin_but_distinctive = any(
                idf_lookup.get(term, 0.0) >= HIGHLY_DISTINCTIVE_IDF
                for term in top_p.matched_terms
            )
            notes.append(
                f"Top provision matched only {len(top_p.matched_terms)} query term(s)"
                + (" — but on a distinctive term, so it is shown for review. "
                   "Add the wording the plan itself uses to confirm."
                   if thin_but_distinctive else
                   "; too thin to treat as governing language.")
            )
        elif len(top_p.distinctive_terms) < MIN_DISTINCTIVE_TERMS:
            gated = True
            notes.append(
                "Top provision matched only common wording, no clinically "
                "distinctive term. Likely a coincidental overlap."
            )
        if not gated and not top_p.core_matched:
            gated = True
            expansion_only = True
            notes.append(
                "Matched through expanded synonyms rather than the service "
                "description itself — the plan describes this service in its own "
                "words. Shown for review: confirm the provision genuinely covers "
                "this service before citing it."
            )
        if top_p.has_conflict:
            blocking = True
            gated = True
            for c in top_p.conflicts:
                notes.append(f"QUALIFIER CONFLICT: {c}. Do not cite this provision.")

    if top_p is not None and top_p.source_type == "definition" and not gated:
        gated = True
        notes.append(
            "The best match is a definition, not an exclusion or benefit limit. "
            "Definitions constrain how a provision applies but do not themselves "
            "deny a service — look for the governing provision before citing."
        )

    if gated:
        if blocking:
            outcome = "not_found"
        elif top >= supported_threshold:
            outcome = "weak"
        elif (thin_but_distinctive or expansion_only) and top >= weak_threshold:
            outcome = "weak"
        else:
            outcome = "not_found"
    elif top >= supported_threshold:
        outcome = "supported"
    elif top >= weak_threshold:
        outcome = "weak"
        notes.append(
            "Match is weak. Confirm the provision genuinely covers this service "
            "before citing it; do not cite on similarity alone."
        )
    else:
        outcome = "not_found"
        notes.append(
            "No plan provision clearly governs this service. Common bases that "
            "live outside the exclusions text: medical necessity criteria, "
            "missing clinical documentation, network status, exhausted benefit "
            "maximums, or eligibility. Route to a qualified reviewer to "
            "establish the actual basis rather than citing an approximate match."
        )

    if provisions and provisions[0].has_exception:
        notes.append(
            "The top provision carries an exception clause. An exclusion whose "
            "exception applies does not support a denial -- confirm the "
            "exception is inapplicable on these clinical facts."
        )
    disagreement = rerank_disagreement(provisions)
    if disagreement:
        notes.append(disagreement)
    if provisions and provisions[0].cross_references:
        notes.append(
            "The provision cross-references "
            f"{', '.join(provisions[0].cross_references[:3])}. Read the "
            "referenced text before relying on this citation."
        )
    if any(p.source_type == "definition" for p in provisions[:2]):
        notes.append(
            "A defined term is among the top matches. Definitions constrain how "
            "an exclusion applies and usually belong in the letter alongside it."
        )

    return DenialSupport(
        codes=[normalize_code(c) for c in codes],
        descriptors=list(descriptors),
        query_terms=list(terms),
        provisions=list(provisions),
        outcome=outcome,
        notes=notes,
    )


def format_citation_block(support: DenialSupport, max_provisions: int = 2) -> str:
    """Reviewer-facing draft. Quotes are verbatim; nothing is paraphrased."""
    if support.outcome == "not_found":
        return ("NO SUPPORTING PLAN PROVISION FOUND.\n\n"
                "Do not draft a denial citing plan language from this search. "
                "Establish the actual basis for the determination first.")

    lines: List[str] = []
    for p in support.provisions[:max_provisions]:
        lines.append(f"Plan provision: {p.citation}")
        if p.lead_in:
            lines.append("Quoted text:")
            for i, part in enumerate(p.lead_in):
                lines.append(f"{'    ' * (i + 1)}{part}")
            lines.append(f"{'    ' * (len(p.lead_in) + 1)}{p.text}")
        else:
            lines.append(f'Quoted text: "{p.text}"')
        if p.has_exception:
            lines.append(f"  ** Exception in this provision: {p.exception_clause}")
        if p.cross_references:
            lines.append(f"  ** Cross-references: {', '.join(p.cross_references)}")
        lines.append("")
    if support.outcome == "weak":
        lines.append("REVIEW REQUIRED: match confidence is low.")
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# LLM-assisted code description  (developer testing)
# --------------------------------------------------------------------------- #
#: Deliberately asks for a plain-language clinical description rather than the
#: official code descriptor. Two reasons: CPT descriptors are AMA-copyrighted
#: and should come from a licensed source, and a model reciting a descriptor
#: from memory produces something that *looks* authoritative while being subtly
#: wrong -- the exact failure this module was built to prevent. A paraphrase is
#: honest about what it is, and it is what the plan-language search needs anyway,
#: since plan documents never use code-set vocabulary.
DESCRIBE_SYSTEM_PROMPT = """You help a benefits engineer search plan documents. \
Given a procedure code and optionally a diagnosis code, describe in plain \
clinical English what service was performed and why.

Rules:
- Do NOT reproduce the official AMA CPT descriptor verbatim. Paraphrase in \
plain clinical language.
- If you are not confident what a code represents, say so: set "confidence" to \
"low" and explain in "caveats". Never guess a plausible-sounding service.
- "search_terms" should be the words a health plan document would use for this \
service, not code-set vocabulary. Plans write "hanging skin", "cosmetic", \
"weight loss surgery" -- not "panniculectomy".
- Note in "caveats" anything that changes whether the service is covered: \
whether it is typically cosmetic vs reconstructive, elective vs medically \
necessary, or commonly subject to prior authorization.

Respond with JSON only, no markdown fencing:
{
  "procedure_description": "...",
  "diagnosis_description": "..." or null,
  "clinical_context": "one or two sentences on why this pairing occurs",
  "search_terms": ["...", "..."],
  "confidence": "high" | "medium" | "low",
  "caveats": ["..."]
}"""


@dataclass
class CodeDescription:
    procedure_code: str
    diagnosis_code: Optional[str]
    procedure_description: str
    diagnosis_description: Optional[str]
    clinical_context: str
    search_terms: List[str]
    confidence: str
    caveats: List[str]
    model: str
    raw: str = ""

    @property
    def is_uncertain(self) -> bool:
        return self.confidence.lower() != "high"


def describe_codes(
    procedure_code: str,
    diagnosis_code: Optional[str] = None,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 900,
) -> CodeDescription:
    """Ask an LLM what a code pairing represents, in plain clinical English.

    For developer testing. A model's recollection of a code is not a substitute
    for a licensed code set: verify the description before relying on any
    provision it leads you to.
    """
    import json as _json
    import os as _os

    proc = normalize_code(procedure_code)
    diag = normalize_code(diagnosis_code) if diagnosis_code else None
    ask = f"Procedure code: {proc} ({classify_code(proc)})"
    if diag:
        ask += f"\nDiagnosis code: {diag} ({classify_code(diag)})"
    ask += "\n\nDescribe these in plain clinical English."

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key or _os.environ.get("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=model or "claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=DESCRIBE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": ask}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        used = model or "claude-sonnet-4-6"
    elif provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key or _os.environ.get("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model or "gpt-4.1-mini",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": DESCRIBE_SYSTEM_PROMPT},
                {"role": "user", "content": ask},
            ],
        )
        text = resp.choices[0].message.content or ""
        used = model or "gpt-4.1-mini"
    else:
        raise ValueError(f"Unknown provider: {provider}")

    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        data = _json.loads(cleaned)
    except Exception:
        data = {
            "procedure_description": cleaned[:500],
            "diagnosis_description": None,
            "clinical_context": "",
            "search_terms": [],
            "confidence": "low",
            "caveats": ["Model response was not valid JSON; treat with caution."],
        }

    return CodeDescription(
        procedure_code=proc,
        diagnosis_code=diag,
        procedure_description=(data.get("procedure_description") or "").strip(),
        diagnosis_description=(data.get("diagnosis_description") or None),
        clinical_context=(data.get("clinical_context") or "").strip(),
        search_terms=[str(t).strip() for t in (data.get("search_terms") or []) if str(t).strip()],
        confidence=str(data.get("confidence") or "low").lower(),
        caveats=[str(c) for c in (data.get("caveats") or [])],
        model=used,
        raw=text,
    )
