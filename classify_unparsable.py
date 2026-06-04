import json
import re
import zipfile
from pathlib import Path

import pandas as pd

# Matches queries that start with EXEC or EXECUTE (case-insensitive),
# ignoring leading whitespace. \b prevents matching words like "executive".
EXEC_RE = re.compile(r"^\s*exec(ute)?\b", re.IGNORECASE)


def is_exec(query: str) -> bool:
    return bool(EXEC_RE.match(query or ""))


def classify_queries(queries: list[dict]) -> dict:
    """Classify one file's list of unparsable_queries."""
    texts = [q.get("query", "") for q in queries]
    non_exec = [t for t in texts if not is_exec(t)]

    if not texts:
        verdict = "No unparsable queries"
    elif non_exec:
        verdict = "Needs manual review"
    else:
        verdict = "No manual review"

    return {
        "verdict": verdict,
        "num_queries": len(texts),
        "num_exec": len(texts) - len(non_exec),
        "num_non_exec": len(non_exec),
        # first non-exec query, trimmed, so you can see why it was flagged
        "first_non_exec": (non_exec[0][:120] if non_exec else ""),
    }


def classify_zip(zip_path: str) -> pd.DataFrame:
    """Walk every .json in the zip and classify it."""
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        json_names = [
            n for n in zf.namelist()
            if n.lower().endswith(".json") and not n.endswith("/")
        ]
        for name in sorted(json_names):
            row = {"file": Path(name).name}
            try:
                data = json.loads(zf.read(name).decode("utf-8"))
                queries = data.get("unparsable_queries", [])
                row.update(classify_queries(queries))
            except Exception as e:
                row.update({
                    "verdict": "Parse error",
                    "num_queries": None,
                    "num_exec": None,
                    "num_non_exec": None,
                    "first_non_exec": f"{type(e).__name__}: {e}",
                })
            rows.append(row)

    cols = ["file", "verdict", "num_queries", "num_exec", "num_non_exec", "first_non_exec"]
    return pd.DataFrame(rows, columns=cols)


if __name__ == "__main__":
    import sys
    df = classify_zip(sys.argv[1])
    print(df.to_string(index=False))
    print("\nSummary:")
    print(df["verdict"].value_counts().to_string())
