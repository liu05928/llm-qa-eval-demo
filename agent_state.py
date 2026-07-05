from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentTraceStep:
    """A user-visible execution step for the RAG Agent."""

    node: str
    action: str
    tool_name: str
    input_summary: str
    output_summary: str
    status: str = "success"
    elapsed_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    """Runtime state carried through the local Single-Agent workflow."""

    question: str
    top_k: int = 3
    candidate_k: int = 10
    max_rewrites: int = 1
    use_rerank: bool = True
    context_mode: str = "small_to_big"
    guard_mode: str = "v2"
    current_query: str = ""
    resolved_question: str = ""
    query_type: str = "general"
    query_type_label: str = "普通问题"
    query_reason: str = ""
    memory_used: bool = False
    memory_reason: str = ""
    memory_snapshot: Dict[str, Any] = field(default_factory=dict)
    retriever_mode: str = "dense_rerank"
    context_sufficient: bool = False
    context_reason: str = ""
    context_coverage: float = 0.0
    support_level: str = "unsupported"
    evidence_score: float = 0.0
    guard_details: Dict[str, Any] = field(default_factory=dict)
    claim_verification: Dict[str, Any] = field(default_factory=dict)
    rewrite_count: int = 0
    rewritten_queries: List[str] = field(default_factory=list)
    rewrite_steps: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    small_retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    long_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_logs: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    agent_trace: List[AgentTraceStep] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.current_query:
            self.current_query = self.question

        if not self.resolved_question:
            self.resolved_question = self.current_query

    def add_trace(self, step: AgentTraceStep):
        self.agent_trace.append(step)

    def trace_dicts(self) -> List[Dict[str, Any]]:
        return [step.to_dict() for step in self.agent_trace]

    def to_result(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "resolved_question": self.resolved_question,
            "answer": self.answer,
            "sources": self.sources,
            "retrieved_chunks": self.retrieved_chunks,
            "small_retrieved_chunks": self.small_retrieved_chunks,
            "long_context": self.long_context,
            "agent_trace": self.trace_dicts(),
            "query_type": self.query_type,
            "query_type_label": self.query_type_label,
            "query_reason": self.query_reason,
            "memory_used": self.memory_used,
            "memory_reason": self.memory_reason,
            "memory_snapshot": self.memory_snapshot,
            "context_sufficient": self.context_sufficient,
            "context_reason": self.context_reason,
            "context_coverage": self.context_coverage,
            "support_level": self.support_level,
            "evidence_score": self.evidence_score,
            "guard_mode": self.guard_mode,
            "guard_details": self.guard_details,
            "claim_verification": self.claim_verification,
            "rewritten_queries": self.rewritten_queries,
            "rewrite_steps": self.rewrite_steps,
            "retriever_mode": self.retriever_mode,
            "context_mode": self.context_mode,
            "candidate_k": self.candidate_k,
            "top_k": self.top_k,
            "use_rerank": self.use_rerank,
            "errors": self.errors,
        }
