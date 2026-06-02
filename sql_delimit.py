#!/usr/bin/env python3
"""sql_delimit.py - validate-and-annotate SQL statement delimiters.

Designed for a manual, file-by-file pass over scripts a bulk auto-repair tool
could not finish. For one input file it:

  * splits the file into top-level statements (respecting strings, comments,
    parentheses, BEGIN/CASE..END blocks, GO batches, and existing ';');
  * inserts a ';' ONLY where sqlglot confirms the resulting split genuinely
    parses (high confidence) or where an opaque-Command segment can be broken
    into clean sub-statements (medium confidence, flagged for review);
  * NEVER guesses: any segment it cannot resolve is left untouched and reported
    with line/column, the parser error, and a context snippet.

"Correct" is defined as "sqlglot can parse it", which approximates the
sqlglot-based downstream parser. It is an approximation, not your exact oracle.

Usage
-----
    python sql_delimit.py file.sql                 # dry-run report to stdout
    python sql_delimit.py file.sql -o fixed.sql    # also write repaired copy
    python sql_delimit.py file.sql --apply-medium  # also apply medium-conf splits
    python sql_delimit.py file.sql --dialect ""    # non-T-SQL file (generic)
    python sql_delimit.py file.sql --json rep.json # machine-readable report

Dependency: pip install sqlglot
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import sqlglot
from sqlglot import expressions as exp
from sqlglot.errors import ParseError, TokenError
from sqlglot.tokens import TokenType

# Keywords that can legitimately start a new statement. Used only to PROPOSE
# candidate boundaries; an actual parse decides whether a split is taken, so an
# over-broad list is safe (false candidates simply fail to validate).
STATEMENT_STARTERS = frozenset(
    {
        "SELECT", "INSERT", "UPDATE", "DELETE", "MERGE", "CREATE", "ALTER",
        "DROP", "TRUNCATE", "EXEC", "EXECUTE", "DECLARE", "SET", "IF", "WHILE",
        "WITH", "PRINT", "RAISERROR", "THROW", "RETURN", "GOTO", "WAITFOR",
        "USE", "GRANT", "REVOKE", "DENY", "COMMIT", "ROLLBACK", "SAVE",
        "BACKUP", "RESTORE", "DBCC", "OPEN", "CLOSE", "FETCH", "DEALLOCATE",
        "BEGIN", "BREAK", "CONTINUE",
    }
)
BLOCK_OPEN = frozenset({"BEGIN", "CASE"})
BLOCK_CLOSE = frozenset({"END"})
MAX_SPLIT_ATTEMPTS = 4000  # guard against pathological large procedures
GO_RE = re.compile(r"^[ \t]*GO\b[ \t]*\d*[ \t]*(--.*)?$", re.IGNORECASE)


@dataclass
class Insertion:
    offset: int          # char offset in ORIGINAL text where ';' is inserted
    line: int
    col: int
    confidence: str      # "high" | "medium"
    note: str
    context: str


@dataclass
class Unresolved:
    line: int
    col: int
    reason: str          # "parse_error" | "opaque_command"
    error: str
    context: str


@dataclass
class FileReport:
    path: str
    dialect: str
    segments: int = 0
    clean_statements: int = 0
    insertions: list[Insertion] = field(default_factory=list)
    unresolved: list[Unresolved] = field(default_factory=list)
    parsable_before: Optional[int] = None
    parsable_after: Optional[int] = None
    error: Optional[str] = None


# ----------------------------- helpers -------------------------------------

def _line_col(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    last_nl = text.rfind("\n", 0, offset)
    col = offset + 1 if last_nl == -1 else offset - last_nl
    return line, col


def _context(text: str, offset: int, size: int = 60) -> str:
    a = max(0, offset - size)
    b = min(len(text), offset + size)
    snippet = text[a:b].replace("\n", " ").strip()
    return ("..." if a > 0 else "") + snippet + ("..." if b < len(text) else "")


def _tokens(sql: str, dialect: str):
    try:
        return sqlglot.tokenize(sql, dialect=dialect or None)
    except (TokenError, Exception):
        return None


def _parse_status(sql: str, dialect: str) -> tuple[str, int]:
    """Return (status, n) where status in {clean, command, multi, error}.

    clean  -> exactly one non-Command statement
    command-> exactly one opaque Command (unsupported syntax fallback)
    multi  -> more than one statement (already internally delimited)
    error  -> raised ParseError/TokenError
    """
    s = sql.strip()
    if not s:
        return "empty", 0
    try:
        stmts = [x for x in sqlglot.parse(s, read=dialect or None) if x is not None]
    except (ParseError, TokenError, Exception):
        return "error", 0
    if not stmts:
        return "empty", 0
    if len(stmts) > 1:
        return "multi", len(stmts)
    return ("command" if isinstance(stmts[0], exp.Command) else "clean"), 1


def _is_clean_single(sql: str, dialect: str) -> bool:
    return _parse_status(sql, dialect)[0] == "clean"


# ------------------------- top-level segmentation ---------------------------

def batch_spans(sql: str) -> list[tuple[int, int]]:
    """Split on GO batch separators LEXICALLY (line-based), before tokenizing.

    GO is a client directive, not SQL; sqlglot can absorb it into a procedure
    body, so it must be handled at the text level. Returns char spans of batch
    CONTENT (GO lines themselves excluded)."""
    spans: list[tuple[int, int]] = []
    start = 0
    pos = 0
    for line in sql.splitlines(keepends=True):
        if GO_RE.match(line.rstrip("\r\n")):
            if pos > start:
                spans.append((start, pos))
            start = pos + len(line)
        pos += len(line)
    if start < len(sql):
        spans.append((start, len(sql)))
    return spans or [(0, len(sql))]


def semicolon_spans(text: str, dialect: str) -> list[tuple[int, int]]:
    """Split a single batch into top-level statement spans on ';' (block/paren
    aware). Offsets are LOCAL to ``text``."""
    toks = _tokens(text, dialect)
    if not toks:
        return [(0, len(text))]
    spans: list[tuple[int, int]] = []
    seg_start = 0
    paren = 0
    block = 0
    for t in toks:
        ttype = t.token_type
        ttext = (t.text or "").upper()
        if ttype == TokenType.L_PAREN:
            paren += 1
            continue
        if ttype == TokenType.R_PAREN:
            paren = max(0, paren - 1)
            continue
        if ttext in BLOCK_OPEN:
            block += 1
            continue
        if ttext in BLOCK_CLOSE:
            block = max(0, block - 1)
            continue
        if ttype == TokenType.SEMICOLON and paren == 0 and block == 0:
            spans.append((seg_start, t.end + 1))
            seg_start = t.end + 1
    if seg_start < len(text):
        spans.append((seg_start, len(text)))
    return spans


def candidate_offsets(seg: str, dialect: str) -> list[int]:
    """Local offsets (>0) of top-level statement-starter keywords in a segment."""
    toks = _tokens(seg, dialect)
    if not toks:
        return []
    offs: list[int] = []
    paren = 0
    block = 0
    for i, t in enumerate(toks):
        ttext = (t.text or "").upper()
        if t.token_type == TokenType.L_PAREN:
            paren += 1
            continue
        if t.token_type == TokenType.R_PAREN:
            paren = max(0, paren - 1)
            continue
        top = paren == 0 and block == 0
        if top and i > 0 and ttext in STATEMENT_STARTERS:
            offs.append(t.start)
        if ttext in BLOCK_OPEN:
            block += 1
        elif ttext in BLOCK_CLOSE:
            block = max(0, block - 1)
    # de-dup, preserve order
    seen: set[int] = set()
    out = []
    for o in offs:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out


def greedy_split(seg: str, dialect: str) -> tuple[list[tuple[int, int]], Optional[int]]:
    """Peel successive clean single statements off the front of ``seg``.

    Returns (spans, unresolved_local_offset). ``spans`` are (start, end) char
    ranges within ``seg`` of statements that each parse cleanly. If a remainder
    cannot be reduced to clean statements, its start offset is returned as the
    unresolved point and not split further.
    """
    spans: list[tuple[int, int]] = []
    base = 0
    attempts = 0
    while True:
        rest = seg[base:]
        if not rest.strip():
            return spans, None
        # MAXIMAL MUNCH: take the LONGEST prefix that parses as one clean
        # statement. Shortest-prefix over-splits (sqlglot accepts the
        # incomplete "INSERT INTO t (a)" as a complete Insert).
        best = None
        if _is_clean_single(rest, dialect):
            best = len(rest)
        else:
            for off in candidate_offsets(rest, dialect):  # ascending
                attempts += 1
                if attempts > MAX_SPLIT_ATTEMPTS:
                    return spans, base
                if _is_clean_single(rest[:off], dialect):
                    best = off  # keep the largest clean prefix
        if best is None:
            return spans, base
        chunk = rest[:best]
        lead_ws = len(chunk) - len(chunk.lstrip())
        spans.append((base + lead_ws, base + len(chunk.rstrip())))
        if best >= len(rest):
            return spans, None
        base += best


# ------------------------------- core --------------------------------------

def analyze(sql: str, path: str, dialect: str, apply_medium: bool) -> tuple[FileReport, list[Insertion]]:
    rep = FileReport(path=path, dialect=dialect or "(generic)")
    applied: list[Insertion] = []

    def already_terminated(end_off: int) -> bool:
        j = end_off
        while j < len(sql) and sql[j] in " \t\r\n":
            j += 1
        return j < len(sql) and sql[j] == ";"

    seg_iter = [
        (bs + ls, bs + le)
        for (bs, be) in batch_spans(sql)
        for (ls, le) in semicolon_spans(sql[bs:be], dialect)
    ]
    for (s, e) in seg_iter:
        seg = sql[s:e]
        if not seg.strip():
            continue
        rep.segments += 1
        # strip a trailing top-level ';' or GO marker from the working text
        status, n = _parse_status(seg.rstrip().rstrip(";"), dialect)

        if status in ("clean", "multi"):
            rep.clean_statements += 1
            end_off = s + len(seg.rstrip().rstrip(";").rstrip())
            if not already_terminated(end_off) and not seg.rstrip().endswith(";"):
                line, col = _line_col(sql, end_off)
                ins = Insertion(end_off, line, col, "high",
                                "clean statement, added terminator", _context(sql, end_off))
                rep.insertions.append(ins)
                applied.append(ins)
            continue

        # status == error or command -> attempt validated split
        work = seg.rstrip().rstrip(";")
        spans, unresolved = greedy_split(work, dialect)
        confidence = "high" if status == "error" else "medium"

        if len(spans) >= 2 or (len(spans) == 1 and unresolved is None and status == "command"):
            # produce an insertion after every sub-statement that lacks one
            for idx, (ls, le) in enumerate(spans):
                end_off = s + le
                is_last = idx == len(spans) - 1 and unresolved is None
                if is_last and (already_terminated(end_off) or seg.rstrip().endswith(";")):
                    continue
                if already_terminated(end_off):
                    continue
                line, col = _line_col(sql, end_off)
                note = ("split glued statements" if status == "error"
                        else "split opaque Command into parseable statements")
                ins = Insertion(end_off, line, col, confidence, note, _context(sql, end_off))
                rep.insertions.append(ins)
                if confidence == "high" or apply_medium:
                    applied.append(ins)
                rep.clean_statements += 1

        if unresolved is not None:
            # the (remaining) text could not be resolved
            uoff = s + (len(seg.rstrip().rstrip(";")) - len(work)) + unresolved
            # recover the sqlglot error message for the unresolved tail
            tail = work[unresolved:]
            try:
                sqlglot.parse(tail.strip(), read=dialect or None)
                err = "parsed only as opaque Command (unsupported syntax)"
                reason = "opaque_command"
            except (ParseError, TokenError) as ex:
                err = str(ex).split("\n")[0]
                reason = "parse_error"
            except Exception as ex:  # noqa: BLE001
                err = f"{type(ex).__name__}: {ex}".split("\n")[0]
                reason = "parse_error"
            line, col = _line_col(sql, uoff)
            rep.unresolved.append(Unresolved(line, col, reason, err, _context(sql, uoff)))
        elif not spans and status == "command":
            # whole segment is an opaque Command we could not improve
            line, col = _line_col(sql, s)
            rep.unresolved.append(
                Unresolved(line, col, "opaque_command",
                           "parsed only as opaque Command (unsupported syntax)",
                           _context(sql, s)))

    return rep, applied


def apply_insertions(sql: str, insertions: list[Insertion], annotate: bool) -> str:
    out = sql
    for ins in sorted(insertions, key=lambda x: x.offset, reverse=True):
        marker = ";" + ("  -- [auto-delimiter]" if annotate else "")
        out = out[: ins.offset] + marker + out[ins.offset :]
    return out


def count_parsable(sql: str, dialect: str) -> tuple[int, int]:
    """Return (parsable_segments, total_segments) under top-level splitting."""
    ok = total = 0
    seg_iter = [
        (bs + ls, bs + le)
        for (bs, be) in batch_spans(sql)
        for (ls, le) in semicolon_spans(sql[bs:be], dialect)
    ]
    for (s, e) in seg_iter:
        seg = sql[s:e].strip().rstrip(";").strip()
        if not seg:
            continue
        total += 1
        st, _ = _parse_status(seg, dialect)
        if st in ("clean", "multi"):
            ok += 1
    return ok, total


# ------------------------------- CLI ---------------------------------------

def render_report(rep: FileReport) -> str:
    L = []
    L.append(f"file        : {rep.path}")
    L.append(f"dialect     : {rep.dialect}")
    L.append(f"segments    : {rep.segments}")
    hi = sum(1 for i in rep.insertions if i.confidence == "high")
    me = sum(1 for i in rep.insertions if i.confidence == "medium")
    L.append(f"insertions  : {len(rep.insertions)} ({hi} high, {me} medium)")
    L.append(f"unresolved  : {len(rep.unresolved)}")
    if rep.parsable_before is not None:
        L.append(f"parsable    : {rep.parsable_before} -> {rep.parsable_after} (of {rep.segments} segments, before-fix basis)")
    if rep.insertions:
        L.append("\n-- proposed ';' insertions --")
        for i in rep.insertions:
            L.append(f"  L{i.line}:C{i.col} [{i.confidence}] {i.note}")
            L.append(f"      … {i.context}")
    if rep.unresolved:
        L.append("\n-- NEEDS MANUAL REVIEW (left untouched) --")
        for u in rep.unresolved:
            L.append(f"  L{u.line}:C{u.col} [{u.reason}] {u.error}")
            L.append(f"      … {u.context}")
    if not rep.insertions and not rep.unresolved:
        L.append("\nNo changes needed; all statements already terminated and parseable.")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate-and-annotate SQL statement delimiters.")
    ap.add_argument("infile")
    ap.add_argument("-o", "--outfile", help="write repaired SQL here")
    ap.add_argument("--dialect", default="tsql", help="sqlglot dialect (default tsql; '' = generic)")
    ap.add_argument("--apply-medium", action="store_true",
                    help="also apply medium-confidence (opaque-Command) splits")
    ap.add_argument("--annotate", action="store_true",
                    help="append a ' -- [auto-delimiter]' marker to each inserted ';'")
    ap.add_argument("--json", dest="json_path", help="write machine-readable report here")
    args = ap.parse_args(argv)

    with open(args.infile, encoding="utf-8") as f:
        sql = f.read()

    rep, applied = analyze(sql, args.infile, args.dialect, args.apply_medium)
    rep.parsable_before, _ = count_parsable(sql, args.dialect)

    if args.outfile:
        repaired = apply_insertions(sql, applied, args.annotate)
        rep.parsable_after, _ = count_parsable(repaired, args.dialect)
        with open(args.outfile, "w", encoding="utf-8", newline="") as f:
            f.write(repaired)
    else:
        repaired = apply_insertions(sql, applied, args.annotate)
        rep.parsable_after, _ = count_parsable(repaired, args.dialect)

    print(render_report(rep))

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(rep), f, indent=2)

    # exit code 0 if fully resolved, 2 if anything needs manual review
    return 0 if not rep.unresolved else 2


if __name__ == "__main__":
    raise SystemExit(main())
