# Checklist: web search quality and guardrails

Temporary working checklist (started 2026-11-07). Goal: accurate, relevant, easy-to-understand answers
that keep us out of legal trouble in Ontario. Delete or fold into `docs/agent_architecture.md` when done.

---

## A. Fix the web search

Design inspiration: a **search tool** returns titles and URLs only, then a **page-fetcher tool**
(`httpx` + BeautifulSoup) downloads each page and strips the HTML noise. We keep our own rule that the
graph has fixed edges, so this is one bounded pass, not a model-driven loop.

### A0. Decisions
- [x] A model writes an easy-to-understand web summary, **without losing the context or meaning** of the
      source (see A5). Reverses Deviation 7's "snippets verbatim, never model-written".
- [x] Keep the current setup and fallback: ddgs first, SearXNG fallback. Revisit the order if ddgs keeps
      failing. In a live check both worked; SearXNG ranked `ontario.ca` and `ibc.ca` first, ddgs ranked
      broker blogs.
- [x] Add a trusted-domain priority list, plus auto-insurance information sites (A2).
- [x] Local LLM is not an option; cloud models only (Groq or Gemini).

### A1. Quick fixes (done 2026-11-07)
- [x] `.env`: trailing backslash removed from `IRAG_SEARXNG_URL`.
- [x] `_searxng` strips a stray `/` or `\` from the URL.
- [x] Ad links (`bing.com/aclick`, `duckduckgo.com/y.js`, `googleadservices.com`, `/aclk?`) dropped from every
      backend; a backend that returns only ads falls through to the next one.
- [x] ddgs "No results found." is an empty result, not a logged failure.
- [x] English only: backends asked for English (`ddgs` region `ca-en`, SearXNG `language=en`); French
      titles, text and `/fr/` URLs dropped; a bilingual snippet keeps only its English half.
- [x] 8 new tests in `tests/test_web.py` (19 pass).
- [ ] Known gap: the language check is a word-list heuristic, not a language detector; revisit in A3.

### A2. Better queries and candidates (done 2026-11-07)
- [x] Query gains "Ontario" and "auto insurance" when the question lacks them.
- [x] 10 candidates per backend, de-duplicated by URL.
- [x] Domain ranking (a nudge added to the relevance score: tier 1 +0.25, tier 2 +0.05, tier 3 -0.15).
      Tier 1 official: `ontario.ca`, `fsrao.ca`, `ibc.ca`, `insurancehotline.com`, `tribunalsontario.ca`,
      `canlii.org`, `ribo.com`, `giocanada.org`, `lso.ca`. Tier 2 information: `ratehub.ca`, `rates.ca`,
      `caa.ca`, `insurance-canada.ca`. Tier 3 sales pages: a listed set of insurers and brokers, plus any
      domain containing "insur" or "broker". **Lists still need your confirmation.**

### A3. Relevance filter (done 2026-11-07, method changed)
- [x] Each candidate scored by term overlap: the share of the question's distinctive words (plus
      "Ontario") found in the title and snippet. Floor `web_min_relevance = 0.3`.
- [x] Why not the encoder: embedding 10 results took 93 s on this CPU even warm, and the time varied a lot.
      Term overlap is instant. Cost: it misses synonyms ("raise" vs "affect"), so it is a weaker proxy.
- [x] Both backends now feed one ranked list (ddgs gave broker blogs, SearXNG gave `ibc.ca` and `fsrao.ca`);
      either still works alone if the other fails.
- [ ] Known gap: floor calibrated on about 55 live results plus 4 junk texts; recheck on the unanswerable
      golden questions.

### A4. Fetch pages, not snippets (done 2026-11-07)
- [x] Top 3 results fetched with httpx: `robots.txt` respected, generic User-Agent (no email), 10 s
      timeout, 1 MB cap, HTML only; BeautifulSoup strips script, nav, header, footer, aside and forms.
- [x] Readable English paragraphs of 60 to 600 characters; the best 3 per page are kept verbatim, in page
      order. A passage must contain "Ontario" and about 60% of the question's terms (this dropped the
      Newfoundland and Nova Scotia paragraphs on the IBC page).
- [x] A page that cannot be read (403, robots, not HTML, timeout) keeps its search snippet.
- [x] Live timing: 7 to 13 s per question (was 209 s with encoder scoring of paragraphs).
- [ ] Known gaps: many sites answer 403 to automated fetches (`fsrao.ca`, `lowestrates.ca`, `rates.ca`, and
      others), so those fall back to snippets; passages are sometimes intro text rather than the answer
      (A5 summary will help); no JavaScript rendering.

### A5. Plain-language summary (flag `IRAG_WEB_SUMMARY`, off by default)
- [ ] One cloud LLM call over the extracted passages only.
- [ ] **Preserve meaning:** same facts, numbers, dates and conditions ("unless", "subject to", exceptions);
      no new facts; say so when the sources disagree or are silent.
- [ ] A deterministic faithfulness check: every number, date and quoted term in the summary must appear in
      the passages, or the summary is dropped and the verbatim passages are shown instead.
- [ ] Every sentence tied to a source URL; labelled "Web summary: unverified, not legal advice".
- [ ] Never enters the corpus answer, the citations or the eval path.

### A6. Graph shape
- [ ] Web node = search, fetch, rank, (summarise); one bounded pass; still only after a refusal and only
      when the web flag is on. Raise `max_agent_steps` only if the node count changes.

### A7. Tests and checks
- [ ] Unit tests: relevance threshold, domain ranking, passage extraction, summary labelling and the
      faithfulness check.
- [ ] Guards stay: web off in evals, web text never in the corpus prompt.
- [ ] Live check on the five test questions; judge relevance, usefulness and plain language.

### A8. Docs
- [ ] Update `docs/agent_architecture.md` (web section, Deviation 7 note) and the layer diagrams.

---

## B. Guardrails and policies around the agent

### B0. Business rules (from the risk table below). Needs a lawyer's review before deployment.

| # | Risk | When it becomes serious | How to reduce it | Guardrail in the agent |
|---|---|---|---|---|
| 1 | Acting like an unlicensed insurance agent or broker | Recommending a specific insurer or policy, selling coverage, taking applications, binding coverage, negotiating premiums, or advising what coverage to buy | Do not sell, quote-bind, recommend a particular insurer, or hold yourself out as a broker or agent. FSRA licenses people and businesses that sell or provide insurance products or services to the public | Block insurer and product recommendations and rankings; no quotes; never say or imply we are a broker; no insurer sales pages as web sources (B2, B4, A2) |
| 2 | Personalised insurance advice | Telling a user "Buy Policy X", "You are definitely covered" or "You should choose this deductible" | General education, document summaries, checklists, questions to ask, and citations, not individual recommendations | Banned advice and certainty phrases; rewrite to general information (B4) |
| 3 | Giving legal advice | Telling someone how to respond to a denial, what their legal rights are, or how to win a dispute | State the tool gives no legal advice; direct users to a licensed Ontario lawyer or paralegal, the insurer's complaint process, the ombudservice, or FSRA where appropriate. Ontario restricts legal services to licensed legal professionals | Detect "what should I do / will I win" requests and answer with the safe template plus referral (B2, B6) |
| 4 | Wrong or hallucinated answers | The model says a policy covers something it does not, cites the wrong clause, or misstates a deadline | Retrieve and display the exact source passage, require citations, flag low-confidence answers, tell users to verify with their insurer or broker | Keep citations required and the verifier; show the source text; low-confidence flag; "verify" line (B4) |
| 5 | Privacy breach | Users upload declarations pages, claim files, IDs, accident photos, health details, addresses or emails | Clear consent, minimise data, encrypt, restrict retention, and avoid sending raw personal data to third-party model providers where possible. PIPEDA applies to commercial handling of personal information in Ontario; organisations stay responsible for information sent to third-party processors | Redact personal data **before any call to Groq or Gemini** and before logging; retention rule (B2, B7) |
| 6 | Misrepresentation or unfair practices | Marketing claims like "guaranteed lowest rate", "we will get your claim approved" or "legal advice" | Precise wording: "educational", "research assistant", "document organiser", "not insurance advice", "not legal advice", "verify with your insurer or licensed professional" | Fixed disclaimer wording on every answer and in the UI copy (B4) |
| 7 | Complaint-handling liability | The tool escalates incorrectly or misses an important deadline | Present escalation routes as general information, not instructions; users confirm deadlines and procedures with FSRA, their insurer or a qualified professional. FSRA accepts complaints and has an official complaint process | Referral template with "confirm deadlines" wording; no invented contact details (B6) |

Source tags in the original text were not kept; the legal claims above are the author's and unverified.

- [ ] Your further business rules and the exact disclaimer wording (send when ready).

### B1. Policy layer
- [ ] One config-driven policy module (`insurance_rag/agent/policy.py` plus a rules file), deterministic
      first; an LLM classifier only if the rules prove too weak.
- [ ] Two graph nodes: `guard_input` before `plan`, `guard_output` after the answer or refusal.

### B2. Input guardrails
- [ ] Scope: Ontario auto insurance only; out-of-scope gets a polite redirect.
- [ ] Detect requests for advice, recommendations, legal strategy or outcome predictions (rules 1 to 3).
- [ ] Redact personal data (names, policy and claim numbers, health details, addresses, emails) before
      any model call and before logging (rule 5).
- [ ] Prompt-injection check on the question.

### B3. Untrusted content
- [ ] Treat retrieved and web text as data: delimit it, never follow instructions found inside it.

### B4. Output guardrails
- [ ] Standard disclaimer on every answer: general information, not insurance or legal advice, not an
      official version, verify with your insurer or a licensed professional (rules 2, 3, 6).
- [ ] Banned phrases ("guaranteed", "you will", "you are definitely covered", "you should buy") and advice
      or prediction wording (rules 1, 2).
- [ ] Keep the citation requirement, the revoked-law flag and the web-block labelling (rule 4).
- [ ] Show the exact source passage with the answer; flag low confidence (rule 4).
- [ ] State the jurisdiction and the law's consolidation date.
- [ ] Handle a cut-off answer: "What changed with Ontario auto insurance?" ended mid-sentence because the
      reply hit `max_tokens=1024`. Detect `finish_reason == "length"`, and either trim to the last
      complete sentence with a note or ask for a shorter answer.
- [ ] Stop answering with tangential provisions (same question): refuse unless an excerpt directly
      answers it (see the memory note's fix 2).

### B5. Easy-to-understand language
- [ ] Answer format: one-line answer first, then what the law says, then what it does not cover.
- [ ] Short sentences and everyday words; explain defined terms (we already tag `defined_terms`).
- [ ] A readability check on the golden answers and the five test questions.

### B6. Refusal and referral templates
- [ ] Fixed wording for: no answer in the corpus, out of scope, needs a professional, web results shown.
- [ ] Refer to a licensed broker, a licensed Ontario lawyer or paralegal, the insurer's complaint process,
      the ombudservice or FSRA, as general information; never invent contact details (rules 3, 7).

### B7. Audit trail and privacy
- [ ] Log every policy decision with the question, answer and sources, redacted; decide retention
      (rule 5).

### B8. Tests
- [ ] An adversarial question set: jailbreaks, "recommend an insurer", "will I win", personal data,
      injection text.
- [ ] Unit tests per rule; confirm guardrails do not lower golden-set correctness.

### B9. Review
- [ ] Legal review of the rules, the disclaimer and the UI copy before deployment. This checklist is not
      legal advice.

---

## Order of work
1. A1 (done), then A2 and A3 (relevance), then A4 (fetch passages).
2. You send further B0 rules while A is in progress; then B1 to B4 together.
3. A5 and B5 last, once relevance and policy are stable.
4. After each step re-run the free retrieval row and the five test questions; the generation row once
   both parts are done.

Test questions: "Is auto insurance mandatory in Ontario?", "What changed with Ontario auto insurance?",
"Does my insurance cover someone else driving my car?", "Do parking tickets or demerit points raise my
insurance rates?", "How can I lower my Ontario auto insurance rates?"
