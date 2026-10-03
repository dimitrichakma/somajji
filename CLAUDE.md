# Somajji

AI helper for community mental health volunteers in the Chittagong Hill Tracts. The AI never talks to
the person in distress. A volunteer talks to them in the local language, types a summary in Bangla or
English, and Somajji suggests next steps from WHO guides, checks for danger signs, writes an anonymous
case note, and alerts a supervisor when needed. Portfolio demo only: never used with real patients.

**The full plan is in `docs/PLAN.md`.** Read the current phase there before any work. It has the
phases, "done when" checks, code to copy from the chatbot, repo layout and evaluation targets.
If this file and PLAN.md disagree, ask Dimitri. Do not guess.
Each phase has a spec in `specs/phase-<N>-<name>.md`. Build only what the spec says; its acceptance
criteria are the definition of done. Record decisions in the spec's decision log.
How each phase is run (plan mode, review, commit) is in `docs/WORKFLOW.md`.

## How we work

- Dimitri directs the design; you write the code; he reads it and asks why.
- One phase at a time. Show a short plan first and wait for approval.
- At the end of a phase, STOP and report: files changed, key decisions and why (simple English),
  how to run and test, and anything you were unsure about. Wait for "next phase".
- If a request would break a rule below, say so and do not do it.
- A phase is done only when its tests pass. Commit at the end of each phase.

## Safety rules (never break)

- If the keyword layer OR the LLM classifier says high risk, run the fixed protocol.
- The fixed protocol never calls the LLM: static safety steps + an urgent case.
- If the risk check errors or times out, treat the input as high risk.
- The guide agent answers ONLY from retrieved WHO text and cites source and page. No citation, no answer.
- Never diagnose or suggest medicine. Refuse and suggest a referral.
- Remove names, phone numbers and place names before storing or sending text to the LLM.
- Referrals come only from the hand checked list in `data/referrals/`.
- No web search, no voice input, no drug or disease graph.
- Only made up demo data. Never real personal data.

## Access rules (never break)

- New Google users are `pending` and see nothing until an admin approves them with a role and area.
- Read the role from Postgres on EVERY request. Never trust a role from the token, URL or frontend.
- Every protected FastAPI route checks the role itself. Hiding a page in Next.js is not protection.
- Only admins approve, reject or change roles. An admin cannot change their own role.
- Supervisors see only their area's cases. Volunteers see only their own cases.
- First admin comes from `ADMIN_EMAILS` in `.env` or a seed script, never a public route.
- Demo accounts exist only when `DEMO_MODE=true`, with fake data reset nightly.
- Log every AI suggestion, escalation, login and role change to the audit log.
- `/docs` is off in production.

## Stack

- Backend (`app/`): Python 3.13, uv, FastAPI, LangGraph, langchain-anthropic (Claude), Pinecone +
  Voyage embeddings, PostgreSQL, LangSmith, pytest.
- Frontend (`web/`): Next.js App Router, Tailwind CSS, shadcn/ui, Auth.js with Google. The Next.js
  server mints a 10 minute HS256 token for FastAPI (same pattern as the mental health chatbot).
- UI style: calm colours, risk colours green / amber / red, card style cases, Bangla font, mobile first.
- Deploy: Railway, two services (FastAPI and Next.js).

## Code conventions

- Type hints everywhere. Small functions with clear names.
- Structured LLM outputs use Pydantic models.
- Secrets in `.env` only. Keep `.env.example` updated.
- Cost: keep the daily and eval budgets on. Before creating any online resource, remind Dimitri to
  add it to `docs/TEARDOWN.md`. Never create paid cloud resources without asking.
- Every new module gets tests in the same phase.
- Explain things to Dimitri in simple, plain English.
