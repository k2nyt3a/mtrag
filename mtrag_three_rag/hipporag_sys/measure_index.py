import glob, os, pickle, sqlite3, sys
D = sys.argv[1]
sub = glob.glob(os.path.join(D, "*text-embedding-3-small*"))[0]

def parquet_rows(p):
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(p).metadata.num_rows
    except Exception as e:
        return f"?({e})"

for kind in ["chunk", "entity", "fact"]:
    fp = glob.glob(os.path.join(sub, f"{kind}_embeddings", f"vdb_{kind}.parquet"))
    if fp:
        p = fp[0]
        print(f"{kind}: rows={parquet_rows(p)}  bytes={os.path.getsize(p)}")

# LLM cache rows == number of cached LLM calls (NER + triple extraction)
cf = glob.glob(os.path.join(D, "llm_cache", "*.sqlite"))
if cf:
    c = sqlite3.connect(cf[0])
    tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'").fetchall()]
    for t in tabs:
        try:
            n = c.execute("select count(*) from " + t).fetchone()[0]
            print(f"llm_cache table {t}: {n} rows")
        except Exception as e:
            print(f"llm_cache {t}: {e}")

# graph size
gp = glob.glob(os.path.join(sub, "graph.pickle"))
if gp:
    g = pickle.load(open(gp[0], "rb"))
    try:
        print(f"graph: nodes={g.vcount()} edges={g.ecount()}  (igraph)")
    except Exception:
        try:
            print(f"graph: nodes={g.number_of_nodes()} edges={g.number_of_edges()} (networkx)")
        except Exception as e:
            print(f"graph: type={type(g)} {e}")

total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(D) for f in fs)
print(f"TOTAL index bytes: {total} ({total/1e6:.1f} MB)")
