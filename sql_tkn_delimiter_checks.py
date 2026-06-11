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
import re
from dataclasses import dataclass

import sqlglot
from sqlglot import expressions as exp
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

# Low-confidence starters: keywords that ALSO appear as sub-clauses (SELECT in
# INSERT/UNION/subqueries, SET in UPDATE, EXEC in INSERT, IF as DDL clause, WITH
# as CTE/hint/option, FETCH in OFFSET paging, RETURN/ELSE in flow). In
# conservative mode a boundary before one of these is inserted only when
# CORROBORATED (the keyword starts its physical line, or the statement text
# before it parses as one complete statement). Everything NOT in this set is
# high-confidence and is always delimited at a boundary.
LOW_CONFIDENCE = frozenset({
    "SELECT", "SET", "EXEC", "EXECUTE", "IF", "WHILE", "WITH",
    "RETURN", "FETCH", "ELSE",
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

# Object-type keywords that follow DROP/CREATE/ALTER. An 'IF' immediately after
# one of these is the DDL 'IF [NOT] EXISTS' clause (DROP TABLE IF EXISTS ...),
# not a control-flow IF. Required so a standalone IF that merely FOLLOWS a DROP
# statement is not mistaken for a DDL clause.
DDL_OBJECT_TYPES = frozenset({
    "TABLE", "PROCEDURE", "PROC", "VIEW", "INDEX", "FUNCTION", "TRIGGER",
    "DATABASE", "SCHEMA", "SEQUENCE", "TYPE", "SYNONYM", "USER", "ROLE",
    "COLUMN", "CONSTRAINT", "STATISTICS",
})


@dataclass
class TokInsertion:
    offset: int
    line: int
    col: int
    before_keyword: str
    note: str
    flagged: bool = False  # conservative mode: low-confidence, NOT applied; routes file to review


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


def _starts_line(text: str, tok_start: int) -> bool:
    """True if the token at tok_start is the first non-whitespace on its line."""
    ls = text.rfind("\n", 0, tok_start) + 1
    return text[ls:tok_start].strip() == ""


def _parses_clean(text: str, dialect: str) -> bool:
    """True if `text` parses as exactly one complete, non-Command statement.

    A clean parse means the text before a candidate boundary is a whole
    statement, which corroborates that the candidate starts a new one. Returns
    False on any parse error or on a Command (sqlglot's catch-all for input it
    could not structure), so procedural bodies it cannot parse never corroborate.
    """
    s = text.strip()
    if not s:
        return False
    try:
        parsed = [p for p in sqlglot.parse(s, read=dialect or None) if p is not None]
    except Exception:
        return False
    return len(parsed) == 1 and not isinstance(parsed[0], exp.Command)


def annotate(sql: str, dialect: str = "tsql", terminate_all: bool = False,
             conservative: bool = False):
    """Return (insertions, sql). insertions are offsets in sql where a ';' goes.

    terminate_all=False : place ';' only BETWEEN statements (separate them).
    terminate_all=True  : also terminate the LAST statement of every block
        (before a block-closing END / END CATCH) and the final statement at end
        of input. Never before END TRY (illegal) or a CASE...END (an expression).

    conservative=False : insert a delimiter at every detected boundary.
    conservative=True  : high-confidence starters (INSERT/UPDATE/CREATE/...) are
        always delimited; low-confidence ones (SELECT/SET/EXEC/IF/WHILE/WITH/
        RETURN/FETCH/ELSE) are delimited ONLY when corroborated - the keyword
        starts its physical line, or the statement text before it parses clean.
        An uncorroborated low-confidence boundary is NOT inserted (so a statement
        is never cut on a guess); it is recorded as a flagged TokInsertion so the
        file can be routed to review. apply() ignores flagged insertions.
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
    stmt_start = 0  # offset where the current statement began (for prefix parsing)

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
        if txt in {"TRY", "CATCH"} and tt not in NON_STARTER_TYPES and prev_txt in {"BEGIN", "END"}:
            if prev_txt == "BEGIN":
                prev_sig = "BEGIN"        # body follows -> NOT delimited
                if block_stack:
                    block_stack[-1] = txt  # mark the just-pushed BEGIN as TRY/CATCH
            else:  # END
                prev_sig = "TRY" if txt == "TRY" else "END_CATCH"
            prev_txt = txt
            continue

        # Block stack maintenance + terminate-before-END (terminate_all only).
        # Only a real END keyword token counts - never a string literal 'end',
        # and never an identifier part like @end / schema.end.
        if tt == TokenType.END and prev_txt not in {"@", "."}:
            nxt2 = (toks[idx + 1].text or "").upper() if idx + 1 < n else ""
            kind = block_stack.pop() if block_stack else "BEGIN"
            # Terminate the block's last statement, but only for a real
            # BEGIN..END or BEGIN CATCH block - never END TRY, never CASE..END,
            # and not when the block is empty / nested END just closed.
            if terminate_all and depth == 0 and kind != "CASE" \
               and prev_txt not in {"BEGIN", "TRY", "CATCH", "END", None}:
                add_terminator(t.start, "END")
            prev_sig = prev_txt = "END"
            stmt_start = t.end + 1  # prefix parsing restarts after a block close
            continue
        if tt == TokenType.BEGIN:
            # may become TRY/CATCH on the next token; default plain block
            block_stack.append("BEGIN")
        elif tt == TokenType.CASE:
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
            # 'IF' is a DDL clause (DROP TABLE IF EXISTS) only when it directly
            # follows a DDL object-type keyword inside a DROP/CREATE/ALTER - NOT
            # when a standalone IF merely follows a finished DROP statement.
            ddl_if = (txt == "IF"
                      and current_lead in {"DROP", "CREATE", "ALTER", "TRUNCATE"}
                      and prev_sig in DDL_OBJECT_TYPES)
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
            elif ddl_if:
                suppress = True  # DROP TABLE IF EXISTS: IF is a DDL clause
            elif txt == "WITH" and nxt == "(":
                suppress = True  # table hint WITH (NOLOCK), not a CTE
            elif txt == "WITH" and current_lead in {"CREATE", "ALTER"}:
                suppress = True  # procedure/function option: WITH RECOMPILE / ENCRYPTION / ...
            elif txt == "FETCH" and prev_sig in {"ROWS", "ROW"}:
                suppress = True  # OFFSET ... FETCH NEXT paging, not a cursor FETCH
            if txt == "ELSE":
                suppress = True  # never terminate the IF-body before ELSE

            if not suppress:
                off = _retract(sql, t.start)
                line = sql.count("\n", 0, off) + 1
                last_nl = sql.rfind("\n", 0, off)
                col = off + 1 if last_nl == -1 else off - last_nl
                flagged = False
                if conservative and txt in LOW_CONFIDENCE:
                    # Corroborate a low-confidence boundary: the keyword must
                    # start its physical line, OR the statement text before it
                    # must parse as one complete statement. Otherwise do NOT cut
                    # (record it flagged so the file routes to review).
                    corroborated = (_starts_line(sql, t.start)
                                    or _parses_clean(sql[stmt_start:off], dialect))
                    flagged = not corroborated
                insertions.append(TokInsertion(off, line, col, txt,
                                               f"boundary before {txt}",
                                               flagged=flagged))
                if not flagged:
                    stmt_start = t.start  # a confirmed new statement begins here
            # The next top-level starter is a control-flow body for a REAL
            # IF/WHILE/ELSE - not for a DDL 'IF EXISTS' clause.
            expect_body = (txt in {"IF", "WHILE", "ELSE"}) and not ddl_if
            # Advance current_lead. A suppressed continuation keyword (INSERT's
            # SELECT/EXEC, WITH's main query) still advances the lead so the NEXT
            # statement is no longer treated as part of it - except a DDL IF
            # clause and an UPDATE/MERGE SET clause, which are not new leads.
            if not ddl_if and not (current_lead in {"UPDATE", "MERGE"} and txt == "SET"):
                current_lead = txt
            seen_first = True

        # Don't let a literal's CONTENT pollute the previous-token tracking:
        # a string such as 'union' / 'as' / 'then' / 'rows' must not be read as
        # the keyword in SUPPRESS_PREV or the block/identifier guards.
        if tt in NON_STARTER_TYPES:
            prev_sig = prev_txt = "LIT"
        else:
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


def flagged_cuts(insertions: list[TokInsertion], sql: str | None = None,
                 verbose: bool = True) -> list[dict]:
    """Return the low-confidence boundaries that conservative mode suppressed.

    These are spots where a SELECT/SET/IF/... looked like a statement start but
    was NOT corroborated (didn't start its line and the preceding text didn't
    parse). They are NOT delimited - the statement is left intact - but their
    presence means the file should be reviewed by a human. Pass the same `sql`
    used in annotate to include a short context snippet per flag.
    """
    flags = [i for i in insertions if i.flagged]
    out = []
    for f in flags:
        rec = {"line": f.line, "col": f.col, "before_keyword": f.before_keyword}
        if sql is not None:
            ls = sql.rfind("\n", 0, f.offset) + 1
            le = sql.find("\n", f.offset)
            rec["context"] = sql[ls:le if le != -1 else len(sql)].strip()[:80]
        out.append(rec)
    if verbose:
        if out:
            print(f"!! {len(out)} uncorroborated low-confidence boundary(ies) - review:")
            for r in out:
                ctx = f"  {r['context']}" if "context" in r else ""
                print(f"  L{r['line']}:C{r['col']}  before {r['before_keyword']}{ctx}")
        else:
            print("OK - no uncorroborated low-confidence boundaries.")
    return out


def apply(sql: str, insertions: list[TokInsertion], annotate_marker: bool = False,
          marker: str = "[AUTOMATED DELIMITER]") -> str:
    out = sql
    for ins in sorted(insertions, key=lambda x: x.offset, reverse=True):
        if ins.flagged:
            continue  # conservative mode: low-confidence, recorded but not cut
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


def review(sql: str, insertions: list[TokInsertion], context: int = 40,
           verbose: bool = True) -> list[dict]:
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
    if verbose:
        if flagged:
            print(f"!! {len(flagged)} suspicious insertion(s) - review these:")
            for r in flagged:
                print(f"  L{r['line']}:C{r['col']}  before {r['before_keyword']}  "
                      f"(after '{r['prev_char']}')")
                print(f"      {r['context']}")
        else:
            print("No suspicious insertions detected.")
    return flagged


_BLOCK_MARK_RE = re.compile(r"/\*[^*]*\*/")  # strip /* ... */ markers/comments


def check_midline_cut(text_or_path, marker: str = "[AUTOMATED DELIMITER]",
                      is_path: bool = False, verbose: bool = True) -> list[dict]:
    """Flag inserted delimiters that cut a statement MID-LINE.

    A delimiter is suspect when, on the same physical line, there is real code
    BEFORE the ';' and real code AFTER it (ignoring the marker comment and
    whitespace). A correct delimiter normally sits at end-of-line or just before
    a comment, so code on both sides usually means it sliced into one statement
    (e.g. an IF body, a string-built command).

    Works on the finished file. ``marker`` is the INNER text (no '/*'); pass the
    same value used in apply(). Returns the list of suspect lines.
    """
    text = (open(text_or_path, encoding="utf-8", errors="replace").read()
            if is_path else text_or_path)
    needle = "/* " + marker + " */"
    problems: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        idx = line.find(needle)
        if idx == -1:
            continue
        # code before the ';' that precedes this marker
        semi = line.rfind(";", 0, idx)
        before = line[:semi if semi != -1 else idx]
        before_code = _BLOCK_MARK_RE.sub("", before)
        before_code = re.sub(r"--.*$", "", before_code).strip()
        # code after the marker, with any further markers/comments stripped
        after = line[idx + len(needle):]
        after_code = _BLOCK_MARK_RE.sub("", after)
        after_code = re.sub(r"--.*$", "", after_code).strip()
        if before_code and after_code:
            problems.append({
                "line": lineno,
                "after": after_code[:60],
                "full_line": line.rstrip(),
            })
    if verbose:
        if problems:
            print(f"!! {len(problems)} mid-line cut(s) - delimiter splits a statement:")
            for p in problems:
                print(f"  L{p['line']}: code after ';' -> {p['after']!r}")
                print(f"      {p['full_line']}")
        else:
            print("OK - no mid-line statement cuts detected.")
    return problems
