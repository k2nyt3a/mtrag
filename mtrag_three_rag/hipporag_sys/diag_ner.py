import glob, os, sys
import pyarrow.parquet as pq

D = sys.argv[1]
ner_cap = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
sub = glob.glob(os.path.join(D, "*text-embedding-3-small*"))[0]
cp = glob.glob(os.path.join(sub, "chunk_embeddings", "vdb_chunk.parquet"))[0]
schema = pq.ParquetFile(cp).schema_arrow.names
id_col = "hash_id" if "hash_id" in schema else "id"
tbl = pq.read_table(cp, columns=[id_col, "content"]).to_pydict()
m = dict(zip(tbl[id_col], tbl["content"]))

fails = ['chunk-a22a8c071a7ed6c84e35ca0acf3528ca', 'chunk-271ddc28d8d51cb5faec54f67a3d319e',
         'chunk-c92fa0d2866b64393ef9adcf654db9bf', 'chunk-a9eb280528637ef2c53c8add8b070ba8',
         'chunk-44577c663dba586d25cbfefcfd57945e', 'chunk-a30b97bfbd76f9575f003b1e469eb0b3',
         'chunk-c8640a8c58d203f52a1ee2042d2e29fb', 'chunk-3828071fa45d9fb9250b8016aa73a68d',
         'chunk-49919e03452dcc765754e8d2ad41549c', 'chunk-176c84a44acbd56493ee0dc2e38d1009',
         'chunk-b86e99ebe8eaeeb166235ac1a90c25a5', 'chunk-bf0d5619bb469fb43fab26593f5f4c0a',
         'chunk-222455d1c39cc9eb173c2e730eec8d7d', 'chunk-771eb560ff76acdbaa6552f3042fd4b1',
         'chunk-e82ce19542fb4b721385afa0636ea489', 'chunk-c66847fdf3560cf4837462fbb7474a12',
         'chunk-e48ca6fd53f85f8c74c2ea7bc479762d', 'chunk-cf6846fbaabc29a6701aa63213e058b1']

from hipporag.utils.config_utils import BaseConfig
from hipporag.llm.openai_gpt import CacheOpenAI
from hipporag.information_extraction.openie_openai import OpenIE
cfg = BaseConfig()
cfg.save_dir = "/tmp/diag_ner_probe"; cfg.llm_name = "gpt-4o-mini"
cfg.embedding_model_name = "text-embedding-3-small"; cfg.llm_base_url = "https://api.openai.com/v1"
cfg.openie_mode = "online"; cfg.max_retry_attempts = 10
llm = CacheOpenAI.from_experiment_config(cfg)
oie = OpenIE(llm_model=llm, ner_max_tokens=ner_cap)      # <-- cap via constructor (as HippoRAG does)

print(f"testing {len(fails)} chunks at ner_max_tokens={ner_cap}", flush=True)
ok = still = 0
for f in fails:
    c = m.get(f)
    res = oie.ner(chunk_key=f, passage=c if isinstance(c, str) else "")
    md = getattr(res, "metadata", {}) or {}
    ents = getattr(res, "unique_entities", getattr(res, "entities", None))
    err = md.get("error")
    if err:
        still += 1
        print(f"STILL FAIL {f}: finish={md.get('finish_reason')} toks={md.get('completion_tokens')} err={err[:60]}", flush=True)
    else:
        ok += 1
        print(f"OK {f}: n_entities={len(ents) if ents else 0} toks={md.get('completion_tokens')}", flush=True)
print(f"\nRESULT at cap={ner_cap}: {ok} fixed, {still} still failing", flush=True)
