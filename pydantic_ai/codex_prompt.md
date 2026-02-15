# Codex Prompt: Worker–Critic Executive Summary Pipeline (WHAT → WHY → HOW)

## Objective

Refactor the existing worker–critic loop into a **two-phase pipeline** that produces
executive summaries of quantitative market-research findings. The exploration must
follow a deliberate **WHAT → WHY → HOW** narrative arc, and the final output must be
a polished, rubric-scored executive summary — not just the last iteration's response.

---

## Step 0 — Discover the Codebase (do this FIRST, change nothing)

Before writing any code, scan the repository to locate and understand:

1. **The existing worker–critic loop**
   - Find the notebook or module that currently runs the iterative worker–critic loop.
   - Identify: how the worker agent is created, what tools it has, what its system prompt is.
   - Identify: how the critic agent is created, what context it receives, how its feedback becomes the next query.
   - Note the `max_iterations` value and the current stopping condition (empty feedback).

2. **The worker agent's tool stack**
   - Find all `@agent.tool` or tool-registration calls for the worker.
   - These likely include: multi-study question retrieval, segment metadata fetch, perform analysis, etc.
   - List every tool name, its parameters, and what data source it hits (Mongo, DocDB, cache, etc.).

3. **Data caches and dependencies**
   - Find `survey_wise_questions_metadata_cache` — how is it populated? What structure does it hold?
   - Find `survey_wise_segment_metadata_cache` — same questions.
   - Find how entity → active-insight mappings work.
   - Find `ctx.deps` — what fields does the dependency object carry? (study list, segment metadata, tool_list, etc.)

4. **Model and agent configuration**
   - What LLM model name is used? (Do NOT hardcode — read it from wherever the existing code defines it.)
   - What model settings are used (temperature, etc.)?
   - How are agents instantiated in this project? (pydantic_ai.Agent, or a custom wrapper?)

5. **Output artifacts**
   - Find where outputs are written (likely `v5/notebooks/outputs/` or similar).
   - What format? (JSON histories, text summaries, etc.)

6. **Segment and study metadata schemas**
   - Find `SurveyWiseSegmentMetadata` — what fields does it have?
   - Find `SurveyWiseQuestionsData` — what fields does it have?
   - These inform what the critic can reference when steering exploration.

---

## Step 1 — Define Structured Models

Create Pydantic models for all intermediate artifacts. Do NOT use raw strings.

### `ExploreStage` (enum)
```python
class ExploreStage(str, Enum):
    WHAT = "what"   # Top-line findings, key metrics
    WHY  = "why"    # Drivers, segment differences, causation
    HOW  = "how"    # Implications, recommendations, actions
```

### `Finding`
Each finding the worker produces must be tagged:
- `claim` (str) — the insight
- `depth` (ExploreStage) — which narrative layer it serves
- `metric` (Optional[str]) — supporting quant data point
- `segment` (Optional[str]) — which segment/cohort this applies to
- `study_id` (Optional[str]) — source study entity ID
- `confidence` (str) — high / medium / low

### `ExplorationOutput`
What the worker returns each iteration:
- `query_answered` (str)
- `findings` (List[Finding])
- `segments_covered` (List[str])
- `studies_used` (List[str])
- `data_gaps` (List[str])

### `CriticDirective`
What the explore critic returns:
- `has_new_angle` (bool)
- `next_stage` (ExploreStage) — which layer the follow-up targets
- `follow_up_query` (str) — scoped to that stage
- `rationale` (str)
- `what_coverage` (str) — brief assessment
- `why_coverage` (str)
- `how_coverage` (str)
- `gaps_identified` (List[str])

### `ExecutiveSummary`
The final deliverable:
- `title`, `overview`, `key_findings`, `strategic_implications`, `risks_and_caveats`, `recommended_actions`

### `RefineVerdict`
The refine critic's structured score:
- `approved` (bool) — True only if score >= 8
- `score` (int, 1-10)
- `factual_grounding` (int, 1-10)
- `clarity` (int, 1-10)
- `completeness` (int, 1-10)
- `revision_notes` (List[str])

---

## Step 2 — Create 4 Role-Specialised Agents

Read the existing model name and settings from the codebase. Do NOT hardcode them.

### Agent 1: `explore_worker`
- **output_type**: `ExplorationOutput`
- **tools**: Reuse ALL existing worker tools from the codebase (question retrieval, segment metadata fetch, perform analysis, etc.). Discover them from Step 0 and register them on this agent.
- **system_prompt**: You are a quantitative MR analyst. You will be told which narrative depth to focus on (WHAT / WHY / HOW). Tag each finding accordingly. Cite metrics and segments. Flag data gaps.
- **temperature**: Low (0.2-0.3) — factual extraction should be deterministic.

### Agent 2: `explore_critic`
- **output_type**: `CriticDirective`
- **tools**: None (read-only evaluator).
- **system_prompt**: You are a senior research director steering exploration for an executive summary. Assess coverage at each layer (WHAT, WHY, HOW). If WHAT is thin → ask a WHAT question. If WHAT is solid but WHY is thin → ask WHY. If both solid but HOW thin → ask HOW. If all covered → set has_new_angle=False. Do NOT skip layers.
- **context**: Feed it the available segments (from `SurveyWiseSegmentMetadata`), available studies, and the current finding bank.
- **temperature**: Very low (0.2).

### Agent 3: `synthesizer`
- **output_type**: `ExecutiveSummary`
- **tools**: None.
- **system_prompt**: You receive findings grouped by WHAT/WHY/HOW. Map them to the summary structure: overview ← WHAT, key_findings ← WHAT+WHY, strategic_implications ← WHY+HOW, risks ← WHY+data_gaps, recommended_actions ← HOW. Merge, deduplicate, keep under 300 words. Address revision notes if provided.
- **temperature**: Slightly higher (0.4) for narrative quality.

### Agent 4: `refine_critic`
- **output_type**: `RefineVerdict`
- **tools**: None.
- **system_prompt**: Evaluate the draft against: (1) factual grounding — every claim traceable to finding bank? (2) clarity — executive grasps the 'so what' in 60s? (3) completeness — risks, implications, actions present? (4) tone — professional, concise? Set approved=True only if score >= 8.
- **temperature**: Low (0.3).

---

## Step 3 — Build the Graph (or Loop) with Shared State

### `PipelineState`
Mutable state carried across all steps. Read available studies and segments from the existing caches/deps at runtime — do NOT hardcode them.

Fields:
- `original_query` — the user's business question (passed in)
- `available_studies` — read from the existing entity/study mapping in the codebase
- `available_segments` — read from `SurveyWiseSegmentMetadata` cache
- `current_query` — the active question (starts as original_query)
- `current_stage` — ExploreStage (starts as WHAT)
- `finding_bank` — List[Finding] (accumulated, never replaced)
- `segments_covered` — set (updated each iteration)
- `studies_used` — set
- `data_gaps` — List[str]
- `explore_count` — int
- `draft` — Optional[ExecutiveSummary]
- `verdict` — Optional[RefineVerdict]
- `refine_count` — int
- `history` — list of dicts (audit trail)

Helper methods:
- `uncovered_segments()` → segments in available but not yet in segments_covered
- `uncovered_studies()` → same for studies
- `findings_by_stage()` → dict grouping findings by ExploreStage
- `stage_counts()` → {"what": N, "why": N, "how": N}

---

## Step 4 — Implement the 4 Pipeline Steps

### Step A: ExploreWorker
1. If `current_query` is empty, set it to `original_query`.
2. Build prompt with: current stage instruction (WHAT/WHY/HOW), the business question, available studies/segments, stage counts, and existing findings (to avoid repetition).
3. Call `explore_worker` agent (this agent has tools — it will call retrieval, analysis, etc.).
4. **APPEND** output findings to `finding_bank` (never replace).
5. Update `segments_covered`, `studies_used`, `data_gaps`.
6. Increment `explore_count`.
7. Log to `history`.
8. Transition → ExploreCritic.

### Step B: ExploreCritic
1. If `explore_count >= MAX_EXPLORE_ITERS` (read from config, default 4) → transition to SynthesizeDraft.
2. Build prompt with: original query, current stage, stage counts, all findings, uncovered segments/studies, data gaps.
3. Call `explore_critic` agent.
4. Log coverage assessment and decision to `history`.
5. If `has_new_angle=False` → transition to SynthesizeDraft.
6. Else: set `current_stage = directive.next_stage`, `current_query = directive.follow_up_query` → transition back to ExploreWorker.

### Step C: SynthesizeDraft
1. Group findings by stage using `findings_by_stage()`.
2. Build prompt with: original query, WHAT findings, WHY findings, HOW findings, data gaps.
3. If this is a revision loop, include `verdict.revision_notes` in the prompt.
4. Call `synthesizer` agent.
5. Store result in `state.draft`.
6. Log to `history`.
7. Transition → RefineCritic.

### Step D: RefineCritic
1. Build prompt with: full finding bank (ground truth) + draft summary.
2. Call `refine_critic` agent.
3. Store result in `state.verdict`, increment `refine_count`.
4. Log scores to `history`.
5. If `approved=True` OR `refine_count >= MAX_REFINE_ITERS` (default 3) → **return final draft**.
6. Else → transition back to SynthesizeDraft with revision notes.

---

## Step 5 — Integration with Existing Codebase

1. **Reuse existing tool registrations** — do not rewrite retrieval/analysis tools. Import them from wherever they are currently defined and register them on `explore_worker`.

2. **Reuse existing dependency injection** — the `ctx.deps` pattern used by the current worker should carry over. Populate `available_studies` and `available_segments` from the same caches the current code uses.

3. **Reuse existing output paths** — write the final `ExecutiveSummary` (as JSON) and the `history` audit trail to the same output directory the current code uses (discover the path, don't hardcode it).

4. **Preserve existing logging** — if the current code logs tool calls to `ctx.deps.tool_list`, keep that. The explore_worker's tools should still log there.

5. **Config values** — `MAX_EXPLORE_ITERS`, `MAX_REFINE_ITERS`, model name, temperatures should all be read from the project's existing config system (env vars, config file, or constants module). Discover where config lives and use it.

---

## Step 6 — Output Format

After the pipeline completes, produce:

1. **Executive Summary** (structured JSON from the `ExecutiveSummary` model)
2. **Pipeline Stats**:
   - Exploration iterations run
   - Finding counts by stage (WHAT / WHY / HOW)
   - Segments covered vs available
   - Refine iterations run
   - Final rubric scores (overall, grounding, clarity, completeness)
3. **Audit Trail** (full `history` list as JSON — every step, every decision, every score)

Write all three to the project's standard output directory.

---

## Constraints

- **No hardcoded values**: model names, temperatures, study IDs, segment names, file paths, max iterations — all must be read from the existing codebase's config, caches, or deps.
- **No new dependencies**: use only libraries already in the project's requirements.
- **Preserve existing interfaces**: if other code calls the worker-critic loop, maintain the same entry-point signature (or provide a wrapper).
- **Do not delete the old loop**: keep it as a fallback. Name the new pipeline clearly (e.g., `run_executive_summary_pipeline`).

---

## Reference Implementation

See `pydantic_ai/worker_v2.py` in this repository for a standalone reference implementation
using `pydantic_graph`. It demonstrates the exact architecture (4 nodes, 2 phases,
WHAT/WHY/HOW staging, structured models, audit trail). Adapt the patterns to your
project's agent framework, tool registration, and data layer — but keep the same
logical flow and separation of concerns.
