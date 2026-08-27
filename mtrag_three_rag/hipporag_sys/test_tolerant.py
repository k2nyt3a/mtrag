import os, sys, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.mtrag_io import load_env_openai_key
load_env_openai_key()
from hipporag_sys.hipporag_adapter import build_index

# 2 normal passages + 2 known-pathological ones (deterministic non-JSON NER)
texts = [
    "Arizona Cardinals\nThe Arizona Cardinals are a professional American football team based in Phoenix, Arizona.",
    "Chicago Cardinals\nThe Chicago Cardinals were founded in 1898 and later became the Arizona Cardinals.",
    "Student's t-test\nStudent's t-test\ns Δ  ̄ = s 1 2 n 1 + s 2 2 n 2 . ( \\ displaystyle s_ ( \\ bar ( \\ Delta ) ) = ( \\ sqrt ( ( \\ frac ( s_ ( 1 ) ^ ( 2 ) ) ( n_ ( 1 ) ) ) + ( \\ frac ( s_ ( 2 ) ^ ( 2 ) ) ( n_ ( 2 ) ) ) ) ) . )",
    "Primary source\nPrimary source\nWhat is the tone ? Who is the intended audience ? What is the purpose of the publication ? What assumptions does the author make ?",
]
ids = ["docA_0", "docB_0", "docLATEX_0", "docQ_0"]

save_dir = "/tmp/hippo_tolerant_test"
shutil.rmtree(save_dir, ignore_errors=True)
rag = build_index(texts, ids, save_dir, openie_max_workers=4, tolerant_openie=True)
print("BUILD COMPLETED (did not abort on pathological chunks)")
docs = rag.retrieve("Where are the Arizona Cardinals based?")
print("retrieve OK:", [(d["doc_id"], round(d["score"], 3)) for d in docs])
print("TOLERANT_TEST_OK")
