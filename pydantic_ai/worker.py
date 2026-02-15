"""
Worker–Critic Executive-Summary Pipeline
=========================================
Uses PydanticAI agents for role-specialised LLM calls and pydantic_graph
for stateful orchestration with bounded iteration.

Workflow:
    ExtractFacts  ──▶  DraftSummary  ──▶  CritiqueCheck
                            ▲                    │
                            │  (revision notes)  │
                            └────────────────────┘
                                  or End(final)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

# ──────────────────────────────────────────────
# 1. Structured data models
# ──────────────────────────────────────────────


class KeyMetric(BaseModel):
    """A single quantitative finding from the research."""

    label: str = Field(description="Metric name, e.g. 'YoY Revenue Growth'")
    value: str = Field(description="Metric value with unit, e.g. '+12 % YoY'")
    source: Optional[str] = Field(
        default=None, description="Source citation for the metric"
    )


class FactSheet(BaseModel):
    """Structured extraction of market-research facts."""

    market_name: str = Field(description="Name of the market or segment analysed")
    key_metrics: List[KeyMetric] = Field(
        description="Quantitative highlights (at least 3)"
    )
    trends: List[str] = Field(description="Qualitative trends (2-5 bullet points)")
    risks: List[str] = Field(description="Key risks or headwinds (2-4 bullet points)")
    data_gaps: List[str] = Field(
        default_factory=list,
        description="Areas where data was insufficient or missing",
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


class CritiqueResult(BaseModel):
    """Structured output from the critic agent."""

    approved: bool = Field(
        description="True if the summary meets quality bar"
    )
    overall_score: int = Field(
        description="Score from 1 (poor) to 10 (excellent)", ge=1, le=10
    )
    strengths: List[str] = Field(description="What the draft does well")
    revision_notes: List[str] = Field(
        description="Specific, actionable improvements required"
    )


# ──────────────────────────────────────────────
# 2. PydanticAI agents (role-specialised)
# ──────────────────────────────────────────────

MODEL = "gemini-2.0-flash"

extractor_agent = Agent(
    model=MODEL,
    output_type=FactSheet,
    system_prompt=(
        "You are a senior market-research analyst. "
        "Extract ONLY verifiable facts, metrics, trends, and risks from the "
        "raw research text provided. Never invent numbers. If data is missing, "
        "note it in data_gaps."
    ),
    model_settings={"temperature": 0.2},
)

writer_agent = Agent(
    model=MODEL,
    output_type=ExecutiveSummary,
    system_prompt=(
        "You are an executive-communications specialist who writes concise, "
        "insight-driven summaries for C-suite audiences. "
        "Ground every bullet point in a specific metric from the fact sheet. "
        "Use active voice, avoid jargon, and keep the total summary under 300 words. "
        "If revision notes from a prior critique are provided, address each one."
    ),
    model_settings={"temperature": 0.4},
)

critic_agent = Agent(
    model=MODEL,
    output_type=CritiqueResult,
    system_prompt=(
        "You are a rigorous editorial reviewer for executive documents. "
        "Evaluate the draft against this rubric:\n"
        "  1. Factual grounding – every claim traceable to the fact sheet?\n"
        "  2. Clarity – can a busy executive grasp the 'so what' in 60 seconds?\n"
        "  3. Completeness – are key risks and recommended actions present?\n"
        "  4. Tone – professional, concise, free of filler?\n"
        "Set approved=True only if score >= 8. "
        "Always provide specific, actionable revision notes."
    ),
    model_settings={"temperature": 0.3},
)


# ──────────────────────────────────────────────
# 3. Graph state
# ──────────────────────────────────────────────

MAX_REVISIONS = 3


@dataclass
class PipelineState:
    """Mutable state threaded through the graph."""

    raw_research: str
    fact_sheet: Optional[FactSheet] = None
    draft: Optional[ExecutiveSummary] = None
    critique: Optional[CritiqueResult] = None
    revision_count: int = 0
    history: list = field(default_factory=list)  # audit trail


# ──────────────────────────────────────────────
# 4. Graph nodes
# ──────────────────────────────────────────────


@dataclass
class ExtractFacts(BaseNode[PipelineState, None, ExecutiveSummary]):
    """Step 1 – parse raw research into a structured FactSheet."""

    async def run(
        self, ctx: GraphRunContext[PipelineState]
    ) -> DraftSummary:
        result = await extractor_agent.run(
            f"Extract facts from the following market research:\n\n"
            f"{ctx.state.raw_research}"
        )
        ctx.state.fact_sheet = result.output
        ctx.state.history.append({"step": "extract", "facts": result.output.model_dump()})
        return DraftSummary()


@dataclass
class DraftSummary(BaseNode[PipelineState, None, ExecutiveSummary]):
    """Step 2 – compose an executive summary from the fact sheet."""

    async def run(
        self, ctx: GraphRunContext[PipelineState]
    ) -> CritiqueCheck:
        prompt_parts = [
            "Write an executive summary from this fact sheet:\n",
            ctx.state.fact_sheet.model_dump_json(indent=2),
        ]
        if ctx.state.critique and ctx.state.critique.revision_notes:
            prompt_parts.append(
                "\n\nAddress these revision notes from the previous review:\n"
                + "\n".join(f"- {n}" for n in ctx.state.critique.revision_notes)
            )

        result = await writer_agent.run("\n".join(prompt_parts))
        ctx.state.draft = result.output
        ctx.state.history.append({
            "step": "draft",
            "revision": ctx.state.revision_count,
            "title": result.output.title,
        })
        return CritiqueCheck()


@dataclass
class CritiqueCheck(BaseNode[PipelineState, None, ExecutiveSummary]):
    """Step 3 – evaluate the draft; loop back or finish."""

    async def run(
        self, ctx: GraphRunContext[PipelineState]
    ) -> Union[DraftSummary, End[ExecutiveSummary]]:
        prompt = (
            "Critique the following executive summary.\n\n"
            f"## Fact Sheet (ground truth)\n"
            f"{ctx.state.fact_sheet.model_dump_json(indent=2)}\n\n"
            f"## Draft Summary\n"
            f"{ctx.state.draft.model_dump_json(indent=2)}"
        )
        result = await critic_agent.run(prompt)
        ctx.state.critique = result.output
        ctx.state.revision_count += 1
        ctx.state.history.append({
            "step": "critique",
            "revision": ctx.state.revision_count,
            "score": result.output.overall_score,
            "approved": result.output.approved,
        })

        if result.output.approved or ctx.state.revision_count >= MAX_REVISIONS:
            return End(ctx.state.draft)

        # Loop back for another revision
        return DraftSummary()


# ──────────────────────────────────────────────
# 5. Assemble & run the graph
# ──────────────────────────────────────────────

summary_graph = Graph(nodes=[ExtractFacts, DraftSummary, CritiqueCheck])

SAMPLE_RESEARCH = """\
Global EV Battery Market – Q4 2025 Analyst Brief

The global EV battery market reached $112 B in 2024, up 22 % year-over-year.
CATL maintained the largest market share at 37 %, followed by BYD (16 %) and
LG Energy Solution (13 %). Lithium iron phosphate (LFP) chemistry overtook
NMC for the first time, accounting for 52 % of GWh shipped globally.

Average cell-level costs fell to $92/kWh (BloombergNEF), a 14 % decline from
the prior year, driven by scale efficiencies in Chinese manufacturing.  Sodium-
ion batteries entered pilot production, with CATL and BYD both announcing
2025 vehicle launches.

Key risks: lithium carbonate prices remain volatile ($18-24 k/tonne range);
EU Carbon Border Adjustment Mechanism (CBAM) may raise landed costs for
Asian-manufactured cells by 6-9 %; and US IRA subsidy eligibility rules create
supply-chain bifurcation risk for OEMs.

Regional demand is shifting: China accounted for 63 % of global battery
installations, Europe 20 %, and North America 12 %.  European OEMs are
accelerating local gigafactory plans (Northvolt, ACC, PowerCo) to comply
with CBAM and secure IRA-equivalent subsidies.
"""


async def main() -> None:
    """Run the full pipeline on sample research data."""
    state = PipelineState(raw_research=SAMPLE_RESEARCH)
    result = await summary_graph.run(ExtractFacts(), state=state)

    summary: ExecutiveSummary = result.output
    print("=" * 70)
    print(f"  {summary.title}")
    print("=" * 70)
    print(f"\n{summary.overview}\n")
    print("KEY FINDINGS")
    for i, finding in enumerate(summary.key_findings, 1):
        print(f"  {i}. {finding}")
    print("\nSTRATEGIC IMPLICATIONS")
    for imp in summary.strategic_implications:
        print(f"  • {imp}")
    print("\nRISKS & CAVEATS")
    for risk in summary.risks_and_caveats:
        print(f"  ⚠ {risk}")
    print("\nRECOMMENDED ACTIONS")
    for action in summary.recommended_actions:
        print(f"  → {action}")
    print(f"\n{'─' * 70}")
    print(f"Revisions: {state.revision_count} | "
          f"Final critic score: {state.critique.overall_score}/10")
    print(f"Audit trail: {len(state.history)} steps logged")


if __name__ == "__main__":
    asyncio.run(main())
