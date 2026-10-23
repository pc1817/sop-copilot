from src.embed import load_env
from src.index import get_index
from src.retrieve import search_dense, search_bm25

load_env()
index = get_index()

groups = {
    "IN-SCOPE": [
        "How do I submit an expense report?",
        "How are team member reimbursements processed in Navan?",
        "What is the process for paying a vendor invoice?",
        "What is the procurement process for buying software?",
    ],
    "OUT-OF-SCOPE": [
        "What is the parental leave policy?",
        "How do I reset my laptop password?",
        "What is the capital of France?",
        "How do I configure an n8n webhook?",
    ],
    "NEAR-MISS (handbook only points to internal pages)": [
        "How is payroll processed?",
        "What does the SOX program require?",
        "What is the sales commission plan?",
    ],
}

for name, questions in groups.items():
    print(f"\n== {name}")
    for q in questions:
        d = search_dense(index, q, 1)[0]
        b = search_bm25(index, q, 1)[0]
        print(f"dense {d['score']:.3f} | bm25 {b['score']:5.1f} | {q}")
        print(f"      top dense: {d['heading_path'][:80]}")
