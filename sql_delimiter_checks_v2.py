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
    "UNION", "INTERSECT", "EXCEPT", "ALL",  # ALL = UNION ALL
    "TRY",  # 'TRY' = END TRY -> BEGIN CATCH
})

# Token types whose TEXT may equal a keyword but which are never a statement
# starter: string literals (sqlglot reports the unquoted content, so 'select'
# tokenizes with text SELECT) and quoted/delimited identifiers ([select]).
NON_STARTER_TYPES = frozenset({
    TokenType.STRING, TokenType.NATIONAL_STRING, TokenType.RAW_STRING,
    TokenType.UNICODE_STRING, TokenType.HEX_STRING, TokenType.BYTE_STRING,
    TokenType.BIT_STRING, TokenType.HEREDOC_STRING, TokenType.FIXEDSTRING,
    TokenType.IDENTIFIER, TokenType.NUMBER,
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


def annotate(sql: str, dialect: str = "tsql", terminate_all: bool = False):
    """Return (insertions, sql). insertions are offsets in sql where a ';' goes.

    terminate_all=False : place ';' only BETWEEN statements (separate them).
    terminate_all=True  : also terminate the LAST statement of every block
        (before a block-closing END / END CATCH) and the final statement at end
        of input. Never before END TRY (illegal) or a CASE...END (an expression).
    """
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
    block_stack: list[str] = []  # 'BEGIN' | 'TRY' | 'CATCH' | 'CASE'

    def add_terminator(off_token_start: int, label: str):
        off = _retract(sql, off_token_start)
        if off <= 0 or sql[off - 1] == ";":
            return
        line = sql.count("\n", 0, off) + 1
        last_nl = sql.rfind("\n", 0, off)
        col = off + 1 if last_nl == -1 else off - last_nl
        insertions.append(TokInsertion(off, line, col, label, f"terminate before {label}"))

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
                # The body after BEGIN TRY / BEGIN CATCH IS delimited (so the
                # block opener stands alone), per downstream parser requirement.
                prev_sig = "BEGIN_TC"     # not in SUPPRESS_PREV -> body delimited
                if block_stack:
                    block_stack[-1] = txt  # mark the just-pushed BEGIN as TRY/CATCH
            else:  # END
                prev_sig = "TRY" if txt == "TRY" else "END_CATCH"
            prev_txt = txt
            continue

        # Block stack maintenance + terminate-before-END (terminate_all only).
        if tt == TokenType.END or txt == "END":
            nxt2 = (toks[idx + 1].text or "").upper() if idx + 1 < n else ""
            kind = block_stack.pop() if block_stack else "BEGIN"
            # Terminate the block's last statement, but only for a real
            # BEGIN..END or BEGIN CATCH block - never END TRY, never CASE..END,
            # and not when the block is empty / nested END just closed.
            if terminate_all and depth == 0 and kind != "CASE" \
               and prev_txt not in {"BEGIN", "TRY", "CATCH", "END", None}:
                add_terminator(t.start, "END")
            prev_sig = prev_txt = "END"
            continue
        if tt == TokenType.BEGIN or txt == "BEGIN":
            # may become TRY/CATCH on the next token; default plain block
            block_stack.append("BEGIN")
        elif tt == TokenType.CASE or txt == "CASE":
            block_stack.append("CASE")

        nxt = (toks[idx + 1].text or "").upper() if idx + 1 < n else ""
        # A keyword-looking token is NOT a statement starter when it is part of
        # an identifier: a @variable whose name is a keyword (@return, @select),
        # or a qualified name (schema.return). The sigil/dot is the prev token.
        # Also: ELSE inside a CASE expression is not a control-flow ELSE.
        in_case = bool(block_stack) and block_stack[-1] == "CASE"
        is_starter = (txt in STARTERS and tt not in NON_STARTER_TYPES
                      and prev_txt not in {"@", "."}
                      and not (in_case and txt == "ELSE"))

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
            elif current_lead == "WITH" and txt in {"SELECT", "INSERT", "UPDATE", "DELETE", "MERGE"}:
                suppress = True  # CTE main query is part of the WITH statement
            elif txt == "BEGIN" and nxt == "CATCH":
                suppress = True  # END TRY  BEGIN CATCH
            elif txt == "IF" and current_lead in {"DROP", "CREATE", "ALTER", "TRUNCATE"}:
                suppress = True  # DROP TABLE IF EXISTS: IF is a DDL clause
            elif txt == "WITH" and nxt == "(":
                suppress = True  # table hint WITH (NOLOCK), not a CTE
            elif txt == "FETCH" and prev_sig in {"ROWS", "ROW"}:
                suppress = True  # OFFSET ... FETCH NEXT paging, not a cursor FETCH
            if txt == "ELSE":
                suppress = True  # never terminate the IF-body before ELSE

            if not suppress:
                off = _retract(sql, t.start)
                line = sql.count("\n", 0, off) + 1
                last_nl = sql.rfind("\n", 0, off)
                col = off + 1 if last_nl == -1 else off - last_nl
                insertions.append(TokInsertion(off, line, col, txt,
                                               f"boundary before {txt}"))
            # The next top-level starter is a control-flow body only for a REAL
            # IF/WHILE/ELSE - not for a DDL 'IF EXISTS' clause (DROP TABLE IF...).
            ddl_if = txt == "IF" and current_lead in {"DROP", "CREATE", "ALTER", "TRUNCATE"}
            expect_body = (txt in {"IF", "WHILE", "ELSE"}) and not ddl_if
            # Advance current_lead. A suppressed continuation keyword (INSERT's
            # SELECT/EXEC, WITH's main query) still advances the lead so the NEXT
            # statement is no longer treated as part of it - except a DDL IF
            # clause and an UPDATE/MERGE SET clause, which are not new leads.
            if not ddl_if and not (current_lead in {"UPDATE", "MERGE"} and txt == "SET"):
                current_lead = txt
            seen_first = True

        prev_sig = txt
        prev_txt = txt

    # terminate_all: terminate the final statement at end of input.
    if terminate_all and toks:
        last = toks[-1]
        ltxt = (last.text or "").upper()
        if ltxt not in {"GO", ";"} and last.token_type != TokenType.SEMICOLON:
            eof = _retract(sql, len(sql))
            if eof > 0 and sql[eof - 1] != ";":
                line = sql.count("\n", 0, eof) + 1
                last_nl = sql.rfind("\n", 0, eof)
                col = eof + 1 if last_nl == -1 else eof - last_nl
                insertions.append(TokInsertion(eof, line, col, "EOF",
                                               "terminate final statement"))

    return insertions, sql


def apply(sql: str, insertions: list[TokInsertion], annotate_marker: bool = False,
          marker: str = "[AUTOMATED DELIMITER]") -> str:
    out = sql
    for ins in sorted(insertions, key=lambda x: x.offset, reverse=True):
        if ins.offset <= 0 or out[ins.offset - 1] == ";":
            continue
        # Use a BOUNDED block comment, not '--': a line comment would comment
        # out any code that follows on the same line (e.g. two statements that
        # share a physical line). '/* ... */' ends at '*/', so nothing is lost.
        mark = ";" + (f"  /* {marker} */" if annotate_marker else "")
        out = out[:ins.offset] + mark + out[ins.offset:]
    return out


def check_commented_out(text_or_path, marker: str = "-- [AUTOMATED DELIMITER]",
                        is_path: bool = False, verbose: bool = True) -> list[dict]:
    """Scan a finished, delimited file for marker comments that have code after
    them on the same line.

    A '--' comment runs to end of line, so if anything non-whitespace follows
    the marker, that text has been commented out - almost always unintended
    (e.g. an inserted ';  -- [AUTOMATED DELIMITER]' landed mid-line and swallowed
    the rest, such as  + 'select' ...). Returns a list of problem records.

    Pass the marker EXACTLY as it appears in the file (including the leading
    '-- '). Set is_path=True to pass a file path instead of text.
    """
    text = (open(text_or_path, encoding="utf-8", errors="replace").read()
            if is_path else text_or_path)
    problems: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        start = 0
        while True:
            idx = line.find(marker, start)
            if idx == -1:
                break
            trailing = line[idx + len(marker):]
            if trailing.strip():
                problems.append({
                    "line": lineno,
                    "commented_out": trailing.strip(),
                    "full_line": line.rstrip(),
                })
            start = idx + len(marker)
    if verbose:
        if problems:
            print(f"!! {len(problems)} marker(s) with code commented out:")
            for p in problems:
                print(f"  L{p['line']}: commented-out -> {p['commented_out']!r}")
                print(f"      {p['full_line']}")
        else:
            print("OK - no code found commented out after any delimiter marker.")
    return problems


# Characters that, when they immediately precede an inserted ';', indicate the
# delimiter landed mid-expression (e.g. a string-concatenation '+') and is
# almost certainly wrong - a ';' can never validly follow these.
_SUSPICIOUS_PREV = set("+-*/%&|^=<>,(.")


def review(sql: str, insertions: list[TokInsertion], context: int = 40) -> list[dict]:
    """Return (and print) insertions that look suspicious so they can be checked.

    A delimiter is flagged when the last non-space character before it is an
    operator/comma/open-paren (a ';' cannot validly follow these), which catches
    cases like a keyword appearing inside a string ( ... + 'select' + ... ).
    """
    flagged: list[dict] = []
    for ins in sorted(insertions, key=lambda x: x.offset):
        j = ins.offset - 1
        while j >= 0 and sql[j] in " \t\r\n":
            j -= 1
        prev_char = sql[j] if j >= 0 else ""
        if prev_char in _SUSPICIOUS_PREV:
            before = sql[max(0, ins.offset - context):ins.offset].replace("\n", " ")
            after = sql[ins.offset:ins.offset + context].replace("\n", " ")
            rec = {
                "line": ins.line, "col": ins.col,
                "before_keyword": ins.before_keyword,
                "prev_char": prev_char,
                "context": f"{before} <<;>> {after}",
            }
            flagged.append(rec)
    if flagged:
        print(f"!! {len(flagged)} suspicious insertion(s) - review these:")
        for r in flagged:
            print(f"  L{r['line']}:C{r['col']}  before {r['before_keyword']}  "
                  f"(after '{r['prev_char']}')")
            print(f"      {r['context']}")
    else:
        print("No suspicious insertions detected.")
    return flagged
