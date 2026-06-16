import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


TOPIC_KEYWORDS = [
    {
        "topic": "RAG",
        "keywords": ["rag", "检索增强生成", "知识库问答"],
    },
    {
        "topic": "Agent",
        "keywords": ["agent", "智能体", "工具调用", "任务边界", "多智能体"],
    },
    {
        "topic": "Prompt Engineering",
        "keywords": ["prompt engineering", "prompt", "提示词"],
    },
    {
        "topic": "Embedding",
        "keywords": ["embedding", "向量模型", "向量表示"],
    },
    {
        "topic": "向量数据库",
        "keywords": ["向量数据库", "chromadb", "向量知识库"],
    },
    {
        "topic": "Hybrid Search",
        "keywords": ["hybrid search", "hybrid", "bm25", "rrf", "混合检索"],
    },
    {
        "topic": "Rerank",
        "keywords": ["rerank", "重排序", "bge-reranker"],
    },
    {
        "topic": "教育 AI",
        "keywords": ["教育 ai", "教育场景", "辅助教学", "智能助教", "备课"],
    },
    {
        "topic": "模型微调",
        "keywords": ["微调", "sft", "dpo", "lora"],
    },
]

SOURCE_TOPIC_HINTS = {
    "rag_intro": "RAG",
    "agent_intro": "Agent",
    "prompt_engineering": "Prompt Engineering",
    "embedding_intro": "Embedding",
    "vector_database": "向量数据库",
    "vector_knowledge_base": "向量数据库",
    "hybrid_search": "Hybrid Search",
    "rerank_intro": "Rerank",
    "education_ai": "教育 AI",
    "retrieval_optimization": "RAG",
}

FOLLOW_UP_PATTERNS = [
    "它",
    "它们",
    "这个",
    "这个技术",
    "这项技术",
    "该技术",
    "前者",
    "后者",
    "那",
    "那么",
    "上面",
    "前面",
    "呢",
]

REFERENCE_PHRASES = [
    "这个技术",
    "这项技术",
    "该技术",
    "这个方法",
    "这种方法",
    "这个流程",
    "这个系统",
    "它们",
    "它",
    "上面这个",
    "前面这个",
]


def _normalize_text(text: str) -> str:
    return (text or "").lower().strip()


def extract_topics_from_text(text: str) -> List[str]:
    lower_text = _normalize_text(text)
    matched_topics = []

    for item in TOPIC_KEYWORDS:
        positions = [
            lower_text.find(keyword.lower())
            for keyword in item["keywords"]
            if lower_text.find(keyword.lower()) >= 0
        ]

        if positions:
            matched_topics.append((min(positions), item["topic"]))

    matched_topics = sorted(matched_topics, key=lambda item: item[0])
    topics = [topic for _, topic in matched_topics]

    return list(dict.fromkeys(topics))


def extract_topics_from_sources(sources: List[Dict[str, Any]]) -> List[str]:
    topics = []

    for source in sources:
        source_name = source.get("source", "")

        for source_hint, topic in SOURCE_TOPIC_HINTS.items():
            if source_hint in source_name:
                topics.append(topic)

    return list(dict.fromkeys(topics))


def is_follow_up_question(question: str) -> bool:
    stripped_question = question.strip()

    if not stripped_question:
        return False

    if any(pattern in stripped_question for pattern in FOLLOW_UP_PATTERNS):
        return True

    return len(stripped_question) <= 12


def _replace_reference_phrases(question: str, topic: str) -> str:
    resolved = question

    for phrase in sorted(REFERENCE_PHRASES, key=len, reverse=True):
        resolved = resolved.replace(phrase, topic)

    return resolved


def _resolve_comparison_reference(question: str, topics: List[str]) -> str:
    resolved = question

    if topics:
        resolved = resolved.replace("前者", topics[0])

    if len(topics) >= 2:
        resolved = resolved.replace("后者", topics[1])

    return resolved


@dataclass
class MemoryTurn:
    question: str
    resolved_question: str
    answer_summary: str
    query_type: str
    retriever_mode: str
    sources: List[Dict[str, Any]]
    topic: str
    topics: List[str] = field(default_factory=list)
    rewritten_queries: List[str] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationMemory:
    current_topic: str = ""
    current_topics: List[str] = field(default_factory=list)
    recent_turns: List[MemoryTurn] = field(default_factory=list)
    max_turns: int = 6

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ConversationMemory":
        if not data:
            return cls()

        turns = [
            MemoryTurn(**turn)
            for turn in data.get("recent_turns", [])
        ]

        return cls(
            current_topic=data.get("current_topic", ""),
            current_topics=data.get("current_topics", []),
            recent_turns=turns,
            max_turns=data.get("max_turns", 6),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_topic": self.current_topic,
            "current_topics": self.current_topics,
            "recent_turns": [turn.to_dict() for turn in self.recent_turns],
            "max_turns": self.max_turns,
        }

    def reset(self):
        self.current_topic = ""
        self.current_topics = []
        self.recent_turns = []

    def get_last_turn(self) -> Optional[MemoryTurn]:
        if not self.recent_turns:
            return None

        return self.recent_turns[-1]

    def resolve_question(self, question: str) -> Dict[str, Any]:
        if not self.current_topic or not is_follow_up_question(question):
            return {
                "memory_used": False,
                "resolved_question": question,
                "memory_reason": "当前问题不需要使用会话记忆。",
            }

        resolved_question = _resolve_comparison_reference(
            question=question,
            topics=self.current_topics,
        )
        resolved_question = _replace_reference_phrases(
            question=resolved_question,
            topic=self.current_topic,
        )

        if resolved_question == question:
            resolved_question = f"围绕{self.current_topic}，{question}"

        return {
            "memory_used": resolved_question != question,
            "resolved_question": resolved_question,
            "memory_reason": f"根据会话记忆，将当前主题补全为：{self.current_topic}。",
        }

    def update_from_result(
        self,
        question: str,
        resolved_question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        query_type: str,
        retriever_mode: str,
        rewritten_queries: List[str],
    ):
        answer_summary = re.sub(r"\s+", " ", answer or "").strip()[:160]
        text_topics = extract_topics_from_text(
            " ".join([question, resolved_question, answer_summary])
        )
        source_topics = extract_topics_from_sources(sources)
        topics = list(dict.fromkeys(text_topics + source_topics))

        if query_type != "missing" and topics:
            self.current_topic = topics[0]
            self.current_topics = topics[:2]

        turn = MemoryTurn(
            question=question,
            resolved_question=resolved_question,
            answer_summary=answer_summary,
            query_type=query_type,
            retriever_mode=retriever_mode,
            sources=sources,
            topic=self.current_topic,
            topics=self.current_topics,
            rewritten_queries=rewritten_queries,
        )

        self.recent_turns.append(turn)

        if len(self.recent_turns) > self.max_turns:
            self.recent_turns = self.recent_turns[-self.max_turns:]


def ensure_memory(memory: Any) -> Optional[ConversationMemory]:
    if memory is None:
        return None

    if isinstance(memory, ConversationMemory):
        return memory

    if isinstance(memory, dict):
        return ConversationMemory.from_dict(memory)

    raise TypeError("memory 必须是 ConversationMemory、dict 或 None")
