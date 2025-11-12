# Face2Phrase Phase 2 Product Requirements Document (PRD)

## 1. Purpose & Vision

Phase 2 elevates Face2Phrase from a metrics dashboard into a comprehensive communication coaching suite. The goals are:

- Deliver industry-grade analytics with actionable insights.
- Provide contextual, personalized coaching through the chatbot.
- Produce professional reports that users can share with stakeholders.
- Preserve session history to monitor growth over time.

## 2. Scope Overview

1. Data foundation & session persistence.
2. Advanced analytics pipeline (audio, visual, text).
3. Scoring engine with sub-scores and benchmarks.
4. Chatbot v2 with contextual coaching.
5. Professional PDF export.
6. Dashboard history & compare experience.

## 3. Detailed Requirements

### 3.1 Data Foundation & Session Persistence

**Functional**
- Extend persistence layer to store:
  - Composite & sub-scores (clarity, delivery, content, presence).
  - Analytics summaries (filler trends, pause cadence, emotion timeline stats).
  - Coaching highlights (top strengths, growth areas, recommended drills).
  - PDF export URL + versioning.
- New endpoints:
  - `POST /sessions`: save analysis payload.
  - `GET /sessions`: list summaries (supports filters: date range, score range, badge tier).
  - `GET /sessions/:id`: retrieve full analytics package.

**Non-functional**
- Write ops within 1s of analysis completion.
- Data retention defaults to 12 months (configurable).
- Role-based access for future multi-user tenants.

### 3.2 Advanced Analytics Pipeline

**Audio Enhancements**
- Filler trend clustering (density per minute, top clusters).
- Pause cadence map (short vs long pauses, context segments).
- Confidence projection (predict final score from opening segment).

**Visual Enhancements**
- Tension percentage (frames flagged as tense vs relaxed).
- Emotion timeline smoothing (moving average, stability index).
- Eye-contact stability (distribution vs threshold).

**Text Enhancements**
- Topic coherence score (NLP similarity vs planned topics).
- Keyword coverage (compare transcript against expected vocabulary).
- Sentence pattern variety & repetition alerts.

**Frontend Updates**
- New charts: filler trend line, pause cadence heat, emotion ribbon.
- Badges & tooltips describing analytics thresholds.

### 3.3 Scoring Engine & Benchmarks

**Requirements**
- Configurable weight matrix for sub-scores (JSON-based).
- Composite score computed as weighted blend with minimum/maximum bounds.
- Benchmarking: compare to user’s rolling median and anonymized cohort percentile.
- Badge tiers: Practice Ready, Polished, Interview Ready (thresholds configurable).

**Deliverables**
- `models/scoring_engine.py` (or service) with unit tests.
- Score breakdown UI: `AnalyticsTab`, `CoachingTab`, dashboard cards.
- API responses include breakdown + benchmarks.

### 3.4 Chatbot V2 With Contextual Coaching

**Back-end**
- Session context store (key moments, metrics, flagged transcript snippets).
- Prompt templates referencing context and analytics (e.g., filler hotspots, pace tips).
- Optionally support streaming responses (server-sent events or websockets).

**Front-end**
- Chatbot panel displays context chips (e.g., “Top fillers”, “Eye contact 42%”).
- Quick action buttons triggering targeted prompts (“How do I reduce pauses?”).
- Loading states, error handling, transcript reference inline.

### 3.5 Professional PDF Export

**Layout**
- Cover page: logo, session info, summary scores, date.
- Sections:
  1. Executive Summary (strengths & growth bullets).
  2. Score Breakdown (charts for composite & sub-scores).
  3. Analytics Deep Dive (filler trend chart, pause heatmap, emotion timeline).
  4. Transcript Highlights (flagged moments & suggestions).
  5. Recommended Drills & Next Steps.
- Footer with branding and contact info.

**Implementation**
- Generate via HTML-to-PDF (Playwright or headless Chrome) to preserve styling.
- Charts exported as SVG/PNG and embedded.
- Endpoint returns PDF download link; handle retries & error states.

### 3.6 Dashboard History & Compare

**UI**
- Dashboard list/grid of sessions showing badge, composite score, date.
- Filters: time range, badge, improvement indicator.
- Comparison view between two sessions (score delta, key metric changes, chart overlays).
- Quick actions: reopen analytics, download PDF, delete session.

**Data**
- Each session references stored analytics + cached PDF.
- Provide API and UI safety around deletions and retention policy.

## 4. Success Metrics

- ≥20% increase in repeat session usage (leveraging history & comparison).
- ≥50% of users download PDF report after analysis.
- Chatbot usage per session increases by ≥30% with contextual prompts.
- Composite scoring aligns within ±5 points of manual expert grading on sample set.

## 5. Dependencies & Risks

- Additional storage costs for session history & PDFs.
- Complex analytics requires performant pipelines; monitor latency (<10s total post-processing).
- PDF generation infrastructure must be hardened (timeouts, retries).
- Chatbot improvements depend on LLM availability & cost management.

## 6. Timeline (High-level)

| Phase | Duration | Milestones |
|-------|----------|------------|
| Foundations | 2 weeks | DB migration, session endpoints, basic dashboard list |
| Analytics + Scoring | 3 weeks | New metrics, scoring engine, UI updates |
| Chatbot V2 | 2 weeks | Context store, prompt redesign, front-end UI |
| PDF Export | 2 weeks | Template, generator service, UI integration |
| History & Compare polish | 1 week | Comparison UI, retention controls, QA |

Total estimate: ~10 weeks (overlapping work streams possible with larger team).

## 7. Rollout & QA

- Staging environment with anonymized sessions for regression.
- Automated tests for scoring engine, API endpoints, analytics functions.
- Manual QA checklist for PDF layouts across browsers.
- Soft launch with select users; gather feedback before GA.

## 8. Open Questions

- Weighting priorities for scoring — require stakeholder approval.
- Expected benchmark dataset for cohort comparisons (internal vs external).
- Branding assets & tone-of-voice guidelines for PDF and chatbot.

