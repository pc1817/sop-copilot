# SOP Knowledge Copilot

A retrieval-augmented assistant for GitLab's public Finance handbook. It answers procedural questions (for example "How do I submit an expense report?"), cites the handbook pages it used, and declines questions the handbook does not cover.

It is built to be measured, not just demoed: the repo includes a 40-question evaluation set, the scripts that produce every number below, and an honest list of where it still fails.

**Live demo:** https://sop-copilot.streamlit.app/ (password-protected to protect a small free-tier budget; ask me for access)

Not affiliated with GitLab. See [Corpus and license](#corpus-and-license).



![App screenshot](assets/demo2.png)

## What it does

1. **Hybrid retrieval.** One Pinecone document index holds a dense vector field and a BM25 keyword field. Each question runs both searches and fuses the two rankings with reciprocal rank fusion (RRF).
2. **Refusal gate.** If the best dense similarity score is below 0.665, the app declines without calling the LLM.
3. **Grounded answers.** Otherwise an LLM answers from the top 5 chunks only, cites claims as [n], and replies NOT_IN_DOCS when the excerpts do not contain the answer.
4. **Guard rails.** Password gate, 10 questions per session, 30 model calls per day for the whole app.

## Architecture
GitLab handbook, Finance section (66 pages, pinned commit)
| parse -> clean -> chunk -> embed (Gemini) -> index (Pinecone)
v
Pinecone document index: dense vector field + BM25 text field (753 chunks)

question
|-- embed query (Gemini) --> dense search (top 20) --+
|-- BM25 search (top 20) ----------------------------+--> RRF fusion --> top 5 chunks
|
+-- gate: dense top score < 0.665 ? --> decline (no LLM call)
otherwise --> Groq (gpt-oss-120b) answers with [n] citations,
or replies NOT_IN_DOCS --> decline

## Stack

| Part | Choice |
|---|---|
| Language | Python |
| Embeddings | Gemini embedding-001, 768 dimensions (query and document task types) |
| Retrieval | Pinecone document index: BM25 field (English analyzer) + dense vector field |
| LLM | Groq, openai/gpt-oss-120b, temperature 0, reasoning effort low |
| App | Streamlit 1.65.0 on Streamlit Community Cloud, Python 3.12 |
| Cost | $0 so far (all free tiers) |

No orchestration framework: each stage is a small module you can read in a few minutes.

## Repository layout
src/
parse.py read handbook markdown, build source URLs
clean.py strip HTML, shortcodes, link targets; drop near-empty pages
chunk.py heading-aware chunks (about 1800 chars max, 200 overlap)
embed.py Gemini embeddings with a resumable disk cache
index.py create the Pinecone document index, upsert chunks
retrieve.py dense, BM25, hybrid (RRF)
answer.py refusal gate, prompts v1 and v2, Groq call, citation parsing
ingest.py parse -> clean -> chunk -> embed -> index; writes data/ingest_report.json
app.py Streamlit app (password gate, usage caps)
qa.py, ask.py command-line question tools
eval/ questions.json, own_questions.json, results/
check_gold.py checks that every gold section exists in the corpus
run_eval.py retrieval eval (uses no LLM tokens)
run_answers.py answer-level eval (uses LLM tokens)
calibrate.py scores used to choose the refusal threshold
rescore.py compares citation gaps across saved runs

Ingest result: 66 pages parsed, 49 kept (17 dropped: empty pages or pages that only point to the internal handbook), 753 chunks, average 684 characters.

## Evaluation method

| Group | Count | What it tests |
|---|---|---|
| In-scope | 25 | 20 drafted with Claude from the handbook's heading outline, 5 written by me in my own words |
| Out-of-scope | 10 | 5 unrelated topics, 5 finance-adjacent topics the corpus does not cover |
| Near-miss | 5 | topics where the public page only points to the internal handbook |

- **Gold:** a retrieval counts as a hit when one of the top 5 chunks comes from the gold page and its heading contains the gold text (for two questions, a required phrase as well).
- **Metrics:** hit@1 and hit@5 (the correct section is first / in the top 5), and MRR (mean of 1 / position of the first correct chunk).
- **Calibration:** the refusal threshold was chosen from a separate set of 11 questions that is not part of the evaluation.
- **Provenance:** Claude drafted 20 questions and their gold sections from the heading outline; `check_gold.py` confirms that each gold heading exists, but I did not read every gold section in full. The 5 own-words questions are a small control for vocabulary overlap. Answer faithfulness was graded by Claude, claim by claim against the excerpts the model saw. I looked over two of the flagged answers (q02 and q04). The grading is not blind.

## Results

### 1. Retrieval (25 in-scope questions, top 5)

| Method | hit@1 | hit@5 | MRR |
|---|---|---|---|
| BM25 only | 13/25 | 18/25 | 0.590 |
| Dense only | 21/25 | 24/25 | 0.900 |
| Hybrid (RRF) | 20/25 | 24/25 | 0.863 |

Own-words subset (n=5), hit@5: BM25 4/5, dense 4/5, hybrid 5/5.

- Dense and hybrid tie at hit@5. A one- or two-question difference at n=25 is noise.
- BM25 alone is clearly weaker. My untested hypothesis: questions that use different words than the heading ("I lost a receipt" vs "Missing Receipt Affidavit") give it nothing to match.
- Hybrid helped on 2 questions and hurt on 3 compared with dense: when BM25 finds nothing correct, its results push a correct dense hit down.
- **Gold was widened for one question (q22) after I saw the results**, because a second chunk stating the same rule exists. Before the change: BM25 17, dense 24, hybrid 23 of 25 at hit@5. After: 18, 24, 24.

### 2. Refusal gate (threshold 0.665)

| Group | Passed the gate | Declined |
|---|---|---|
| In-scope (25) | 25 | 0 |
| Out-of-scope (10) | 1 | 9 |
| Near-miss (5) | 0 | 5 |

- In the 11-question calibration set, in-scope questions scored 0.700 to 0.770, and everything else scored 0.633 or lower.
- In the evaluation, the lowest in-scope score was 0.680 and the one leak (o09, "total revenue last fiscal year") scored 0.677. The margin is thin.
- At answer level, all 15 non-answerable questions were declined: 14 by the gate and 1 (o09) by the model replying NOT_IN_DOCS. So the model-level refusal was exercised on a single question.
- BM25 scores cannot gate: "How do I reset my laptop password?" scored 14.9 and "What is the sales commission plan?" 15.8, higher than some in-scope questions (10.8 to 19.1). The gate uses the dense score for that reason.

### 3. Answers: prompt v1 vs v2

v2 adds rules for per-step citations, saying who must act, flagging different cases and conflicting sources, saying what is missing, and not assuming an unmentioned item is covered. It changes several rules at once, so any gain belongs to the bundle.

**Citations** (list steps with no citation, 25 answered questions; two runs per v2 configuration):

| Config | Uncited list steps | Avg words per answer |
|---|---|---|
| v1, dense | 14 | 67 |
| v2, dense | 2 and 0 | 97 and 86 |
| v2, hybrid (10 questions) | 0 and 2 | 78 and 87 |

Repeating a configuration changes the counts slightly (the reasoning model is not fully deterministic at temperature 0). The gap between 14 and 0 to 2 is much larger than that variation. The 2 in one hybrid run were nested bullets with the citation on the last sub-bullet.

**Faithfulness** (10 shared questions, claim by claim against the excerpts shown to the model):

| Config | Supported | Partly unsupported | Incomplete |
|---|---|---|---|
| v1, dense | 6 | 3 | 1 |
| v2, dense | 7 | 2 | 1 |
| v2, hybrid | 8 | 2 | 0 |

"Supported" includes answers with a minor flaw (an omitted alternative, a slightly misattributed citation, a small inference). On 5 extra questions graded for v2 + dense only: 3 supported, 2 incomplete. No graded answer contradicted its excerpts. Differences here are single questions.

### 4. Why the demo ships v2 + hybrid

The first example question in the app ("How do I submit an expense report?") exposed the difference. With dense retrieval it was answered the same muddled way in three runs: every step cited, but the contractor/PEO payroll rules were presented as the general procedure. With hybrid retrieval it was answered correctly and with citations in four runs. Hybrid also fixed q22. Retrieval metrics tie, so this is a small-sample judgment, and hybrid was graded on 10 of the 25 in-scope questions only.

## Where it still fails

1. **Cited is not correct.** The muddled answer above had a citation on every step.
2. **Blends cases without saying so.** q04 (payout timing): dense labels GitLab Inc a non-US entity and does not flag that two sources disagree; hybrid imports a "Thursday payment run" from the vendor-payments page.
3. **Overclaims.** q05: "Yes, you can expense a standing desk", while no excerpt mentions standing desks.
4. **Does not say what is missing.** q22 with dense retrieval omits the dollar threshold the question asks about; q20 omits that the detailed process is not in the handbook page; q25 drops a $5K threshold.
5. **Stale content presented as current.** The handbook still contains 2021 Coupa pages next to newer Zip pages, and some pages name specific people to contact.

v2 reliably fixed formatting problems (citations, the wrong actor on q02, an omitted alternative on q13) and mostly did not fix the judgment problems in items 2 to 4.

## Design decisions

- **Gate on the dense score, not BM25** (see Results 2).
- **Hybrid by client-side RRF.** Pinecone scores one ranking method per request, so the fusion is my code, and each retriever can be evaluated alone.
- **Heading-aware chunks, with the heading path in the embedded text.** The handbook nests headings oddly, so gold uses the leaf heading text. I did not run an ablation without the heading prefix.
- **Prompts are versioned in code** (v1 and v2) and each eval run records its prompt version and retrieval mode, so results stay reproducible.
- **No excerpt text in the app**, only headings and links (see Corpus and license).

## Known limitations

- 40 questions, 25 in scope: small. Differences of one or two questions are noise.
- The refusal threshold comes from 11 calibration questions and the margin against the highest leak is 0.003.
- Questions drafted from headings share vocabulary with the corpus, which favors keyword matching. The 5 own-words questions are too few to settle it.
- Parent headings can match several chunks (for example "3. General Guidelines"), which makes some gold entries loose.
- Single runs; one replicate shows small differences; the answer grading is by one non-blind grader.
- The corpus is a snapshot (2026-10-22) of a living document.
- Usage caps are kept in memory and reset when the app restarts. Free tiers can change or pause idle resources; if the demo is down, rebuild the index with `python ingest.py`.
- Not tested against adversarial or prompt-injection input.

## Corpus and license

The corpus is the Finance section of GitLab's public handbook (`content/handbook/finance` in gitlab.com/gitlab-com/content-sites/handbook), commit `1d017e2417c7510ed270a22e03f038e968fdb679`, retrieved 2026-10-22. The handbook site footer reads "All Rights Reserved" and I found no content license, so:

- the handbook text is **not** in this repo (`docs/` is git-ignored);
- the app shows answers that paraphrase the handbook, plus headings and links to the source pages, but no excerpt text;
- a private Pinecone index holds the chunk text for retrieval;
- all rights in the handbook remain with GitLab.

## Run it yourself
pip install -r requirements.txt

.env (never commit it): GEMINI_API_KEY, PINECONE_API_KEY, GROQ_API_KEY, APP_PASSWORD
get the corpus (not included in this repo)

git clone --depth 1 --filter=blob:none --sparse https://gitlab.com/gitlab-com/content-sites/handbook.git
cd handbook
git sparse-checkout set content/handbook/finance

copy the .md files into docs/ in this repo, skipping the readmes/ folder

python ingest.py # builds the index (creates it if missing)
python qa.py “How do I submit an expense report?” --show
python run_eval.py # retrieval eval, no LLM tokens
python run_answers.py # answer eval, uses Groq tokens
python -m streamlit run app.py

The sparse clone fetches the latest handbook, not the pinned commit, so a newer snapshot may not match the evaluation gold.

## Security and cost notes

- API keys come from environment variables or Streamlit secrets. `.env` and `.streamlit/secrets.toml` are git-ignored, and the deployed app uses separate keys from my laptop.
- Dependencies are pinned to exact versions. After the October 2026 PyPI incidents I chose a Streamlit release more than two weeks old.
- Everything runs on free tiers. The Groq budget is shared per organization, which is why the app has a daily cap.

## What I would do next

- A verification pass: a second model call that checks each claim against the excerpts, or structured output with an explicit "what is missing" field.
- Ablate the heading prefix in the embedded text.
- More questions, graded blind, and grade hybrid on all 25.
- A second corpus with a clear open license.
- An n8n front door (chat or webhook) that calls the same pipeline.
