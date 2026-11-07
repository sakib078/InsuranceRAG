# Checklist: web search quality and guardrails

Temporary working checklist (started 2026-11-07). Goal: accurate, relevant, easy-to-understand answers
that keep us out of legal trouble in Ontario. Delete or fold into `docs/agent_architecture.md` when done.

---

## A. Fix the web search: done (2026-11-07)

All eight items are finished and documented in `docs/agent_architecture.md` ("Web supplement"), with the
diagram `docs/Images/web_search_flow.svg`. In short: English-only results with ads and social sites
filtered, relevance ranking with trusted-domain tiers (the lists are in `insurance_rag/agent/web.py`),
page passages read word for word with `robots.txt` respected, and an optional Gemini plain-language
summary behind `IRAG_WEB_SUMMARY` with a faithfulness check.

Still open from A:
- [ ] Confirm the trusted-domain tiers (official, information sites, insurer and broker sales pages).
- [ ] Recheck the relevance floor (0.3) on the unanswerable golden questions.
- [ ] Judge the live output for relevance, usefulness and plain language.

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
1. You send further B0 rules; I build B1 to B4 together (the policy module, input and output rules,
   untrusted-content handling).
2. Then B5 (plain language), B6 (refusal and referral wording) and B7 (audit trail).
3. Then B8 (adversarial tests) and B9 (legal review).
4. After each step re-run the free retrieval row and the five test questions; the generation row last.

Test questions: "Is auto insurance mandatory in Ontario?", "What changed with Ontario auto insurance?",
"Does my insurance cover someone else driving my car?", "Do parking tickets or demerit points raise my
insurance rates?", "How can I lower my Ontario auto insurance rates?"
