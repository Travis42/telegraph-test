# Task bank v2 (V2.1)

Schema per SPEC-V21 §2 (unchanged from v1 plus register/len_class/facts):

    {id, register, len_class, passage, question, expected_answer, facts[]}

Classes:

- **prose** — the 24 existing v1 passages (`ab/tasks/`), reused verbatim;
  `facts` is an empty list by design (the v1 bank never had a fact gate;
  the runner records this explicitly as the `no_facts_declared` state — it
  is NOT a parse failure and NOT a checked zero).
- **fact** — fact-shaped passages, medium (~800 chars) and long
  (~1500–2000 chars), ≥6 per length class. Shipped files were
  hand-authored offline with every fact literally stated in the passage;
  `author_fact_tasks.py` is the one-shot writer. `generate.py` regenerates
  this class with the model-driven generation-verification loop
  (generate → verify every fact → accept only when facts parse AND ≥60%
  verify; all raw model answers logged to `results/tasks2_gen_*.jsonl`).
  Run `python3 ab/tasks2/generate.py` with `ZAI_API_KEY` set; it refuses to
  run without the key and refuses to write a class with fewer than 6
  accepted candidates.
- **formulaic** — 14 workspace/status-register passages with recurring
  schema (STATUS / STANDUP / CHECKLIST / CONFIG lines); written by
  `author_formulaic_tasks.py`.

Length-class conventions used by the matrix: prose=short, fact=medium|long,
formulaic=short.
