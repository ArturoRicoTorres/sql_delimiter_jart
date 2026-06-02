#!/usr/bin/env python3
"""token_delimit.py - token-stream statement delimiter for procedural T-SQL.

Unlike sql_delimit.py (which inserts ';' only where sqlglot CONFIRMS a clean
parse), this inserts a ';' before every token that begins a new statement,
detected purely from the token stream. It therefore works inside BEGIN/END,
BEGIN TRY/CATCH, and procedure bodies that sqlglot cannot parse.

Trade-off: this is a HEURISTIC, not parse-verified. It is tuned for T-SQL
procedural code (DECLARE/SET/SELECT/EXEC/RETURN/IF/WHILE/TRY-CATCH) and flags
nothing - every boundary is a best-effort guess. Review the output.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import sqlglot
from sqlglot.tokens import TokenType

logging.getLogger("sqlglot").setLevel(logging.ERROR)

# Keywords (matched by TEXT, since e.g. RETURN tokenizes as VAR in tsql) that
# begin a new statement.
STARTERS = frozenset({
    "DECLARE", "SET", "SELECT", "INSERT", "UPDATE", "DELETE", "MERGE",
    "EXEC", "EXECUTE", "RETURN", "PRINT", "THROW", "RAISERROR", "WAITFOR",
    "GOTO", "BREAK", "CONTINUE", "FETCH", "OPEN", "CLOSE", "DEALLOCATE",
    "CREATE", "ALTER", "DROP", "TRUNCATE", "WITH", "IF", "WHILE", "BEGIN",
    "USE", "GRANT", "REVOKE", "DENY", "COMMIT", "ROLLBACK", "SAVE",
    "CHECKPOINT", "DBCC", "ELSE",
})

# If the previous significant token is one of these, the starter that follows
# is a body/continuation, not a new statement -> no delimiter.
SUPPRESS_PREV = frozenset({
    "BEGIN", "IF", "WHILE", "ELSE", "THEN", "AS",
    "UNION", "INTERSECT", "EXCEPT", "TRY",  # 'TRY' = END TRY -> BEGIN CATCH
})


@dataclass
class TokInsertion:
    offset: int
    line: int
    col: int
    before_keyword: str
    note: str


def _retract(text: str, offset: int) -> int:
    """Place the ';' right after the last real code char, skipping whitespace
    and trailing block/line comments before ``offset``."""
    i = offset
    while i > 0 and text[i - 1] in " \t\r\n":
        i -= 1
    # block comment immediately before?
    if i >= 2 and text[i - 2:i] == "*/":
        start = text.rfind("/*", 0, i)
        if start != -1:
            return _retract(text, start)
    # line comment occupying the tail of this line?
    line_start = text.rfind("\n", 0, i) + 1
    seg = text[line_start:i]
    pos = seg.find("--")
    if pos != -1:
        return _retract(text, line_start + pos)
    return i


def annotate(sql: str, dialect: str = "tsql"):
    """Return (insertions, fixed_text_builder). insertions are offsets in sql."""
    try:
        toks = [t for t in sqlglot.tokenize(sql, dialect=dialect or None)]
    except Exception:
        return [], sql

    insertions: list[TokInsertion] = []
    depth = 0
    prev_sig: str | None = None
    prev_txt: str | None = None
    current_lead: str | None = None
    seen_first = False
    expect_body = False  # we are between an IF/WHILE/ELSE and its body statement

    n = len(toks)
    for idx, t in enumerate(toks):
        txt = (t.text or "").upper()
        tt = t.token_type
        if tt == TokenType.L_PAREN:
            depth += 1
            prev_sig = prev_txt = "("
            continue
        if tt == TokenType.R_PAREN:
            depth = max(0, depth - 1)
            prev_sig = prev_txt = ")"
            continue

        # Compound block markers: BEGIN TRY / BEGIN CATCH / END TRY / END CATCH.
        # The TRY/CATCH word tokenizes as a bare VAR; normalise prev_sig so the
        # statement that follows is suppressed (block body) or allowed (after
        # END CATCH) as appropriate.
        if txt in {"TRY", "CATCH"} and prev_txt in {"BEGIN", "END"}:
            if prev_txt == "BEGIN":
                prev_sig = "BEGIN"        # body follows -> suppress
            else:  # END
                prev_sig = "TRY" if txt == "TRY" else "END_CATCH"
            prev_txt = txt
            continue

        nxt = (toks[idx + 1].text or "").upper() if idx + 1 < n else ""
        is_starter = txt in STARTERS

        if is_starter and depth == 0:
            suppress = False
            if not seen_first:
                suppress = True
            elif expect_body:
                suppress = True  # body of a preceding IF / WHILE / ELSE
            elif prev_sig in SUPPRESS_PREV:
                suppress = True
            elif current_lead == "INSERT" and txt in {"SELECT", "EXEC", "EXECUTE"}:
                suppress = True  # INSERT ... SELECT/EXEC is one statement
            elif current_lead in {"UPDATE", "MERGE"} and txt == "SET":
                suppress = True  # SET clause of UPDATE/MERGE
            elif txt == "BEGIN" and nxt == "CATCH":
                suppress = True  # END TRY  BEGIN CATCH
            if txt == "ELSE":
                suppress = True  # never terminate the IF-body before ELSE

            if not suppress:
                off = _retract(sql, t.start)
                line = sql.count("\n", 0, off) + 1
                last_nl = sql.rfind("\n", 0, off)
                col = off + 1 if last_nl == -1 else off - last_nl
                insertions.append(TokInsertion(off, line, col, txt,
                                               f"boundary before {txt}"))
            # the next top-level starter is a control-flow body when this token
            # is IF / WHILE / ELSE
            expect_body = txt in {"IF", "WHILE", "ELSE"}
            # update current statement lead unless this is a continuation clause
            if not (current_lead == "INSERT" and txt in {"SELECT", "EXEC", "EXECUTE"}) \
               and not (current_lead in {"UPDATE", "MERGE"} and txt == "SET"):
                current_lead = txt
            seen_first = True

        prev_sig = txt
        prev_txt = txt

    return insertions, sql


def apply(sql: str, insertions: list[TokInsertion], annotate_marker: bool = False) -> str:
    out = sql
    for ins in sorted(insertions, key=lambda x: x.offset, reverse=True):
        if ins.offset <= 0 or out[ins.offset - 1] == ";":
            continue
        mark = ";" + ("  -- [tok]" if annotate_marker else "")
        out = out[:ins.offset] + mark + out[ins.offset:]
    return out
