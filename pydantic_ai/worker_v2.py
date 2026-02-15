"""
Worker–Critic v2: Two-Phase Executive-Summary Pipeline
======================================================
Designed for quantitative market-research with multi-study retrieval,
segment metadata, and iterative exploration.

Phase 1 — EXPLORE with WHAT → WHY → HOW narrative arc
    Stage 1 (WHAT):  Answer the core business question — top-line findings
    Stage 2 (WHY):   Dig into drivers, segment differences, causal factors
    Stage 3 (HOW):   Surface implications, recommendations, actionable angles
    Critic steers the worker through these stages deliberately.
    Findings ACCUMULATE with a depth tag; nothing is lost.

Phase 2 — SYNTHESIZE & REFINE (polish one document)
    Synthesizer merges ALL findings (organised by WHAT/WHY/HOW) into one draft
    Critic scores against rubric → Writer refines → repeat until approved

This separates "discover more insights" from "make the document better",
and ensures the exploration follows the natural executive-summary narrative.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import List, Optional, Union

from enum import Enum

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

# ─────────────────────────────────────────────────────────
# 1. Structured data models
# ─────────────────────────────────────────────────────────


class ExploreStage(str, Enum):
    """The three narrative stages of executive-summary exploration."""

    WHAT = "what"   # What is happening? (top-line findings, key metrics)
    WHY = "why"     # Why is it happening? (drivers, segment splits, causation)
    HOW = "how"     # How should business respond? (implications, actions)


class Finding(BaseModel):
    """A single insight extracted from one exploration iteration."""

    claim: str = Field(description="The insight or finding statement")
    depth: ExploreStage = Field(
        description="Which narrative layer this finding serves: what / why / how"
    )
    metric: Optional[str] = Field(
        default=None,
        description="Supporting quantitative data point, if available",
    )
    segment: Optional[str] = Field(
        default=None,
        description="Segment or cohort this applies to (e.g., 'Cardiologists, US')",
    )
    study_id: Optional[str] = Field(
        default=None, description="Source study entity ID"
    )
    confidence: str = Field(
        default="medium",
        description="high / medium / low — based on sample size and data quality",
    )


class ExplorationOutput(BaseModel):
    """Structured output from one worker exploration iteration."""

    query_answered: str = Field(description="The question this iteration addressed")
    findings: List[Finding] = Field(
        description="Key findings extracted from this pass"
    )
    segments_covered: List[str] = Field(
        default_factory=list,
        description="Segment names that were analysed in this pass",
    )
    studies_used: List[str] = Field(
        default_factory=list,
        description="Study entity IDs whose data was used",
    )
    data_gaps: List[str] = Field(
        default_factory=list,
        description="Questions that could not be answered due to missing data",
    )


class CriticDirective(BaseModel):
    """Critic's guidance after an exploration iteration."""

    has_new_angle: bool = Field(
        description="True if there is a meaningful new angle to explore"
    )
    next_stage: ExploreStage = Field(
        description=(
            "Which narrative stage the follow-up targets: "
            "'what' (more top-line facts), 'why' (drivers/causation), "
            "'how' (implications/actions)"
        )
    )
    follow_up_query: str = Field(
        default="",
        description="The next question for the worker, scoped to next_stage",
    )
    rationale: str = Field(
        description="Why this angle matters for the executive summary"
    )
    what_coverage: str = Field(
        description="Brief assessment: do we have enough WHAT findings?"
    )
    why_coverage: str = Field(
        description="Brief assessment: do we have enough WHY findings?"
    )
    how_coverage: str = Field(
        description="Brief assessment: do we have enough HOW findings?"
    )
    gaps_identified: List[str] = Field(
        default_factory=list,
        description="Segments/studies/themes not yet explored",
    )


class ExecutiveSummary(BaseModel):
    """The final deliverable: a board-ready executive summary."""

    title: str = Field(description="Short headline for the summary")
    overview: str = Field(
        description="1-2 paragraph narrative overview of findings"
    )
    key_findings: List[str] = Field(
        description="3-6 bullet-point findings, each grounded in a metric"
    )
    strategic_implications: List[str] = Field(
        description="2-4 'so what' implications for the business"
    )
    risks_and_caveats: List[str] = Field(
        description="2-3 risks or data-quality caveats"
    )
    recommended_actions: List[str] = Field(
        description="2-4 concrete next-step recommendations"
    )


class RefineVerdict(BaseModel):
    """Structured output from the critic during the refine phase."""

    approved: bool = Field(
        description="True if the summary meets the quality bar (score >= 8)"
    )
    score: int = Field(
        description="1-10 overall quality score", ge=1, le=10
    )
    factual_grounding: int = Field(
        description="1-10: every claim traceable to a finding?", ge=1, le=10
    )
    clarity: int = Field(
        description="1-10: can an executive grasp the 'so what' in 60 seconds?",
        ge=1,
        le=10,
    )
    completeness: int = Field(
        description="1-10: are risks, implications, actions all present?",
        ge=1,
        le=10,
    )
    revision_notes: List[str] = Field(
        description="Specific, actionable improvements to make"
    )


# ─────────────────────────────────────────────────────────
# 2. Agents
# ─────────────────────────────────────────────────────────

MODEL = "gemini-2.0-flash"

# --- Phase 1 agents ---

explore_worker = Agent(
    model=MODEL,
    output_type=ExplorationOutput,
    system_prompt=(
        "You are a quantitative market-research analyst with access to "
        "multi-study survey data, segment metadata, and cross-tabulation tools.\n\n"
        "You will be told which narrative depth to focus on:\n"
        "  - WHAT: Surface top-line findings, key metrics, headline numbers.\n"
        "  - WHY:  Investigate drivers behind the WHAT — segment splits, "
        "correlations, causal factors, regional differences.\n"
        "  - HOW:  Surface strategic implications, competitive positioning, "
        "actionable recommendations grounded in the data.\n\n"
        "Tag each finding with the correct depth (what/why/how). "
        "Cite specific metrics and segments. Flag data gaps honestly."
    ),
    model_settings={"temperature": 0.3},
)

explore_critic = Agent(
    model=MODEL,
    output_type=CriticDirective,
    system_prompt=(
        "You are a senior research director steering the exploration for an "
        "executive summary. An executive summary must answer three layers:\n\n"
        "  1. WHAT is happening? (top-line metrics, headline findings)\n"
        "  2. WHY is it happening? (drivers, segment differences, causation)\n"
        "  3. HOW should the business respond? (implications, recommendations)\n\n"
        "You see all accumulated findings tagged by depth (what/why/how), "
        "plus the segments and studies not yet explored.\n\n"
        "Your job:\n"
        "  - Assess coverage at each layer (what_coverage, why_coverage, how_coverage).\n"
        "  - If WHAT is thin → ask a WHAT question first.\n"
        "  - If WHAT is solid but WHY is thin → ask a WHY question.\n"
        "  - If WHAT and WHY are solid but HOW is thin → ask a HOW question.\n"
        "  - If all three layers have sufficient depth → set has_new_angle=False.\n\n"
        "Do NOT skip layers. Do NOT suggest marginal drill-downs. "
        "The goal is a well-rounded executive summary, not exhaustive analysis."
    ),
    model_settings={"temperature": 0.2},
)

# --- Phase 2 agents ---

synthesizer = Agent(
    model=MODEL,
    output_type=ExecutiveSummary,
    system_prompt=(
        "You are an executive-communications specialist. You receive a bank of "
        "accumulated findings tagged by depth: WHAT, WHY, and HOW.\n\n"
        "Structure the executive summary to follow this natural narrative:\n"
        "  - overview: Lead with the WHAT (what is happening — headline numbers)\n"
        "  - key_findings: Mix of WHAT and WHY findings, each grounded in a metric\n"
        "  - strategic_implications: Drawn from WHY and HOW findings\n"
        "  - risks_and_caveats: From WHY findings + data gaps\n"
        "  - recommended_actions: Drawn directly from HOW findings\n\n"
        "Rules:\n"
        "  1. Merge and deduplicate findings across iterations.\n"
        "  2. Keep the total summary under 300 words.\n"
        "  3. Ground every bullet in a specific metric.\n"
        "  4. If revision notes from a prior critique are provided, address each one.\n"
        "Use active voice, avoid jargon, lead with the 'so what'."
    ),
    model_settings={"temperature": 0.4},
)

refine_critic = Agent(
    model=MODEL,
    output_type=RefineVerdict,
    system_prompt=(
        "You are a rigorous editorial reviewer for executive MR documents. "
        "Evaluate the draft against this rubric:\n"
        "  1. Factual grounding – every claim traceable to the finding bank?\n"
        "  2. Clarity – executive can grasp the 'so what' in 60 seconds?\n"
        "  3. Completeness – risks, implications, and actions all present?\n"
        "  4. Tone – professional, concise, free of filler?\n"
        "Set approved=True only if score >= 8. "
        "Provide specific, actionable revision notes."
    ),
    model_settings={"temperature": 0.3},
)


# ─────────────────────────────────────────────────────────
# 3. Graph state
# ─────────────────────────────────────────────────────────

MAX_EXPLORE_ITERS = 4
MAX_REFINE_ITERS = 3


@dataclass
class PipelineState:
    """Mutable state threaded through both phases."""

    # --- Inputs ---
    original_query: str
    available_studies: List[str] = field(default_factory=list)
    available_segments: List[str] = field(default_factory=list)

    # --- Phase 1: Explore ---
    current_query: str = ""            # query for the current explore iteration
    current_stage: ExploreStage = ExploreStage.WHAT  # start with WHAT
    finding_bank: List[Finding] = field(default_factory=list)
    segments_covered: set = field(default_factory=set)
    studies_used: set = field(default_factory=set)
    data_gaps: List[str] = field(default_factory=list)
    explore_count: int = 0

    # --- Phase 2: Refine ---
    draft: Optional[ExecutiveSummary] = None
    verdict: Optional[RefineVerdict] = None
    refine_count: int = 0

    # --- Audit ---
    history: list = field(default_factory=list)

    def uncovered_segments(self) -> List[str]:
        return [s for s in self.available_segments if s not in self.segments_covered]

    def uncovered_studies(self) -> List[str]:
        return [s for s in self.available_studies if s not in self.studies_used]

    def findings_by_stage(self) -> dict:
        """Group findings by their narrative depth."""
        grouped = {stage: [] for stage in ExploreStage}
        for f in self.finding_bank:
            grouped[f.depth].append(f)
        return grouped

    def stage_counts(self) -> dict:
        """Count findings per narrative stage."""
        counts = {stage.value: 0 for stage in ExploreStage}
        for f in self.finding_bank:
            counts[f.depth.value] += 1
        return counts


# ─────────────────────────────────────────────────────────
# 4. Phase 1 nodes — EXPLORE
# ─────────────────────────────────────────────────────────


@dataclass
class ExploreWorker(BaseNode[PipelineState, None, ExecutiveSummary]):
    """Worker answers the current query and appends findings to the bank."""

    async def run(
        self, ctx: GraphRunContext[PipelineState]
    ) -> "ExploreCritic":
        state = ctx.state

        # First iteration uses original query
        if not state.current_query:
            state.current_query = state.original_query

        stage_instructions = {
            ExploreStage.WHAT: (
                "Focus on WHAT is happening. Surface top-line metrics, "
                "headline numbers, key quantitative findings. "
                "Tag findings with depth='what'."
            ),
            ExploreStage.WHY: (
                "Focus on WHY things are happening. Investigate drivers, "
                "segment differences, regional splits, correlations, and "
                "causal factors behind the WHAT findings. "
                "Tag findings with depth='why'."
            ),
            ExploreStage.HOW: (
                "Focus on HOW the business should respond. Surface strategic "
                "implications, competitive positioning insights, and "
                "actionable recommendations grounded in the data. "
                "Tag findings with depth='how'."
            ),
        }

        prompt = (
            f"## Current Narrative Stage: {state.current_stage.value.upper()}\n"
            f"{stage_instructions[state.current_stage]}\n\n"
            f"## Business Question\n{state.current_query}\n\n"
            f"## Available Studies\n{json.dumps(state.available_studies)}\n\n"
            f"## Available Segments\n{json.dumps(state.available_segments)}\n\n"
            f"## Findings So Far (by stage)\n"
            f"{json.dumps(state.stage_counts())}\n\n"
            f"## Already-Known Findings (avoid repeating these)\n"
            + (
                json.dumps([f.model_dump() for f in state.finding_bank], indent=2)
                if state.finding_bank
                else "None yet — this is the first pass."
            )
        )

        result = await explore_worker.run(prompt)
        output: ExplorationOutput = result.output

        # ACCUMULATE, don't replace
        state.finding_bank.extend(output.findings)
        state.segments_covered.update(output.segments_covered)
        state.studies_used.update(output.studies_used)
        state.data_gaps.extend(output.data_gaps)
        state.explore_count += 1

        state.history.append({
            "phase": "explore",
            "stage": state.current_stage.value,
            "iteration": state.explore_count,
            "query": state.current_query,
            "new_findings": len(output.findings),
            "total_findings": len(state.finding_bank),
            "stage_counts": state.stage_counts(),
        })

        return ExploreCritic()


@dataclass
class ExploreCritic(BaseNode[PipelineState, None, ExecutiveSummary]):
    """Critic steers exploration through WHAT → WHY → HOW, then exits."""

    async def run(
        self, ctx: GraphRunContext[PipelineState]
    ) -> Union[ExploreWorker, "SynthesizeDraft"]:
        state = ctx.state

        # Hard cap on exploration
        if state.explore_count >= MAX_EXPLORE_ITERS:
            state.history.append({
                "phase": "explore_critic",
                "decision": "max_iters_reached",
                "stage_counts": state.stage_counts(),
                "total_findings": len(state.finding_bank),
            })
            return SynthesizeDraft()

        prompt = (
            f"## Original Business Question\n{state.original_query}\n\n"
            f"## Current Exploration Stage: {state.current_stage.value.upper()}\n\n"
            f"## Finding Counts by Narrative Layer\n"
            f"{json.dumps(state.stage_counts(), indent=2)}\n\n"
            f"## All Findings ({len(state.finding_bank)} total)\n"
            + json.dumps([f.model_dump() for f in state.finding_bank], indent=2)
            + f"\n\n## Segments NOT yet explored\n"
            + json.dumps(state.uncovered_segments())
            + f"\n\n## Studies NOT yet used\n"
            + json.dumps(state.uncovered_studies())
            + f"\n\n## Known Data Gaps\n"
            + json.dumps(state.data_gaps)
        )

        result = await explore_critic.run(prompt)
        directive: CriticDirective = result.output

        state.history.append({
            "phase": "explore_critic",
            "iteration": state.explore_count,
            "previous_stage": state.current_stage.value,
            "next_stage": directive.next_stage.value,
            "has_new_angle": directive.has_new_angle,
            "follow_up": directive.follow_up_query,
            "rationale": directive.rationale,
            "coverage": {
                "what": directive.what_coverage,
                "why": directive.why_coverage,
                "how": directive.how_coverage,
            },
        })

        if not directive.has_new_angle:
            # All three layers sufficiently covered → move to Phase 2
            return SynthesizeDraft()

        # Advance the stage and feed the follow-up query
        state.current_stage = directive.next_stage
        state.current_query = directive.follow_up_query
        return ExploreWorker()


# ─────────────────────────────────────────────────────────
# 5. Phase 2 nodes — SYNTHESIZE & REFINE
# ─────────────────────────────────────────────────────────


@dataclass
class SynthesizeDraft(BaseNode[PipelineState, None, ExecutiveSummary]):
    """Merge all accumulated findings into one executive summary."""

    async def run(
        self, ctx: GraphRunContext[PipelineState]
    ) -> "RefineCritic":
        state = ctx.state

        # Organize findings by narrative depth for the synthesizer
        by_stage = state.findings_by_stage()
        prompt_parts = [
            f"## Original Business Question\n{state.original_query}\n",
            f"## WHAT Findings (top-line, {len(by_stage[ExploreStage.WHAT])} items)\n"
            + json.dumps([f.model_dump() for f in by_stage[ExploreStage.WHAT]], indent=2),
            f"\n## WHY Findings (drivers/causation, {len(by_stage[ExploreStage.WHY])} items)\n"
            + json.dumps([f.model_dump() for f in by_stage[ExploreStage.WHY]], indent=2),
            f"\n## HOW Findings (implications/actions, {len(by_stage[ExploreStage.HOW])} items)\n"
            + json.dumps([f.model_dump() for f in by_stage[ExploreStage.HOW]], indent=2),
            f"\n## Data Gaps to Acknowledge\n"
            + json.dumps(list(set(state.data_gaps))),
        ]

        # On revision loops, include prior critique
        if state.verdict and state.verdict.revision_notes:
            prompt_parts.append(
                "\n## Revision Notes from Previous Review\n"
                + "\n".join(f"- {n}" for n in state.verdict.revision_notes)
            )

        result = await synthesizer.run("\n".join(prompt_parts))
        state.draft = result.output
        state.history.append({
            "phase": "synthesize",
            "revision": state.refine_count,
            "title": result.output.title,
        })
        return RefineCritic()


@dataclass
class RefineCritic(BaseNode[PipelineState, None, ExecutiveSummary]):
    """Score the draft; approve or loop back for revision."""

    async def run(
        self, ctx: GraphRunContext[PipelineState]
    ) -> Union[SynthesizeDraft, End[ExecutiveSummary]]:
        state = ctx.state

        prompt = (
            f"## Finding Bank (ground truth, {len(state.finding_bank)} findings)\n"
            + json.dumps([f.model_dump() for f in state.finding_bank], indent=2)
            + f"\n\n## Draft Executive Summary\n"
            + state.draft.model_dump_json(indent=2)
        )

        result = await refine_critic.run(prompt)
        state.verdict = result.output
        state.refine_count += 1

        state.history.append({
            "phase": "refine_critic",
            "revision": state.refine_count,
            "score": result.output.score,
            "factual_grounding": result.output.factual_grounding,
            "clarity": result.output.clarity,
            "completeness": result.output.completeness,
            "approved": result.output.approved,
        })

        if result.output.approved or state.refine_count >= MAX_REFINE_ITERS:
            return End(state.draft)

        return SynthesizeDraft()


# ─────────────────────────────────────────────────────────
# 6. Assemble the graph
# ─────────────────────────────────────────────────────────

pipeline = Graph(
    nodes=[
        ExploreWorker,
        ExploreCritic,
        SynthesizeDraft,
        RefineCritic,
    ]
)


# ─────────────────────────────────────────────────────────
# 7. Demo runner
# ─────────────────────────────────────────────────────────

async def main() -> None:
    state = PipelineState(
        original_query=(
            "What are the key prescribing behaviour trends for biologic "
            "therapies in rheumatology across the US and EU5, and how do "
            "they differ by physician specialty?"
        ),
        available_studies=[
            "STUDY-RA-2024-US",
            "STUDY-RA-2024-EU5",
            "STUDY-BIOSIM-2024-GLOBAL",
        ],
        available_segments=[
            "Rheumatologists - US",
            "Rheumatologists - EU5",
            "Primary Care - US",
            "Primary Care - EU5",
            "High Prescribers (>50 pts/mo)",
            "Low Prescribers (<20 pts/mo)",
        ],
    )

    result = await pipeline.run(ExploreWorker(), state=state)
    summary: ExecutiveSummary = result.output

    # --- Pretty print ---
    print("=" * 70)
    print(f"  {summary.title}")
    print("=" * 70)
    print(f"\n{summary.overview}\n")

    print("KEY FINDINGS")
    for i, f in enumerate(summary.key_findings, 1):
        print(f"  {i}. {f}")

    print("\nSTRATEGIC IMPLICATIONS")
    for imp in summary.strategic_implications:
        print(f"  • {imp}")

    print("\nRISKS & CAVEATS")
    for r in summary.risks_and_caveats:
        print(f"  ⚠ {r}")

    print("\nRECOMMENDED ACTIONS")
    for a in summary.recommended_actions:
        print(f"  → {a}")

    # --- Pipeline stats ---
    counts = state.stage_counts()
    print(f"\n{'─' * 70}")
    print(f"Exploration iters : {state.explore_count}")
    print(f"Findings collected: {len(state.finding_bank)} total")
    print(f"  WHAT findings   : {counts['what']}")
    print(f"  WHY  findings   : {counts['why']}")
    print(f"  HOW  findings   : {counts['how']}")
    print(f"Segments covered  : {state.segments_covered}")
    print(f"Refine iters      : {state.refine_count}")
    print(f"Final score       : {state.verdict.score}/10  "
          f"(grounding={state.verdict.factual_grounding}, "
          f"clarity={state.verdict.clarity}, "
          f"completeness={state.verdict.completeness})")
    print(f"Audit trail       : {len(state.history)} steps")

    # Optionally dump full history
    # with open("pipeline_audit.json", "w") as f:
    #     json.dump(state.history, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
