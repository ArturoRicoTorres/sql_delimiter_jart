def extract_non_exec(zip_path):
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist()
                       if n.lower().endswith(".json") and not n.endswith("/"))
        for name in names:
            try:
                data = json.loads(zf.read(name).decode("utf-8"))
            except Exception:
                continue
            for i, qd in enumerate(data.get("unparsable_queries", [])):
                text = qd.get("query", "")
                if not EXEC_RE.match(text or ""):
                    rows.append({
                        "file": Path(name).name,
                        "query_index": i,
                        "query": text,
                        "error": qd.get("error", ""),
                    })
    return pd.DataFrame(rows, columns=["file", "query_index", "query", "error"])

queries_df = extract_non_exec(ZIP_PATH)
display(queries_df)
print("Total non-exec queries:", len(queries_df))

queries_df.to_excel("non_exec_queries.xlsx", index=False)
