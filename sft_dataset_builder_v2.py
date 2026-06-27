import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CHUNKS_FILE = Path("data/chunks/chunks.json")
BIG_CHUNKS_FILE = Path("data/chunks/big_chunks.json")
SCIENCE_DOCS_DIR = Path("data/raw_docs/science_textbooks")
DEFAULT_OUTPUT_DIR = Path("data/sft_v2")

DEFAULT_TARGET_TOTAL = 2400
DEFAULT_SCIENCE_SHARE = 0.80
RANDOM_SEED = 42

SCIENCE_PREFIX = "science_textbooks/"

TYPE_RATIOS: List[Tuple[str, float]] = [
    ("grounded_qa", 0.45),
    ("citation_qa", 0.20),
    ("refusal", 0.15),
    ("query_rewrite", 0.10),
    ("compare_or_reasoning", 0.10),
]

GROUNDING_INSTRUCTION = (
    "你是一名初中科学智能助教。请严格根据参考资料回答学生问题，"
    "输出必须包含“关键词、教材依据、回答、学习建议、参考来源”。"
    "如果资料不足，请明确说明“资料中未提及”，不要编造。"
)

CITATION_INSTRUCTION = (
    "请根据给定资料回答学生问题，答案末尾必须写出参考来源。"
    "只能使用参考资料中的信息，不能补充资料外事实。"
)

REFUSAL_INSTRUCTION = (
    "你是一名可信教育问答助手。请判断参考资料是否足以回答问题；"
    "资料不足时必须拒答，说明缺少哪类资料，并保留参考来源。"
)

REWRITE_INSTRUCTION = (
    "请把学生的口语化追问改写成适合知识库检索的一句话。"
    "只输出改写后的检索问题，不要回答。"
)

COMPARE_INSTRUCTION = (
    "你是一名初中科学智能助教。请严格依据参考资料完成比较、关系梳理或现象解释，"
    "输出必须包含“关键词、教材依据、回答、学习建议、参考来源”。"
)

SCIENCE_TERMS = [
    "压力",
    "压强",
    "受力面积",
    "帕斯卡",
    "氧气",
    "二氧化碳",
    "呼吸",
    "青蒿素",
    "疟疾",
    "传染病",
    "甲肝",
    "流感",
    "肝炎",
    "预防针",
    "骨折",
    "脊柱损伤",
    "紧急处理",
    "意外事故",
    "烫伤",
    "中暑",
    "触电",
    "溺水",
    "急救",
    "水",
    "固态",
    "液态",
    "气态",
    "升华",
    "凝华",
    "空气",
    "空气质量",
    "大气",
    "大气压强",
    "土壤",
    "酸碱性",
    "含盐量",
    "蒸腾作用",
    "燃烧",
    "灭火",
    "磁体",
    "磁场",
    "电生磁",
    "电阻",
    "滑动变阻器",
    "欧姆定律",
    "电流",
    "电压",
    "电路",
    "能量",
    "能量转化",
    "能量守恒",
    "浮力",
    "牛顿第一定律",
    "杠杆",
    "力",
    "地球自转",
    "四季星空",
    "透镜",
    "声音",
    "气温",
    "大陆漂移",
    "板块构造",
    "生态系统",
    "生态系统的结构",
    "生物圈",
    "生物多样性",
    "遗传",
    "进化",
    "青春期",
    "反射",
    "神经调节",
    "循环系统",
    "呼吸系统",
    "消化",
    "营养",
    "金属",
    "冶炼",
    "酸",
    "碱",
    "化学反应",
    "海水制盐",
    "自然资源",
    "可持续发展",
    "健康",
    "卫生保健",
    "计算机",
    "文化休闲",
    "地质灾害",
    "行星",
    "飞机",
    "升力",
    "开花",
    "结果",
    "种子",
    "根",
    "茎",
    "叶",
]

PROJECT_TERMS = [
    "RAG",
    "检索增强生成",
    "向量数据库",
    "向量检索",
    "embedding",
    "rerank",
    "混合检索",
    "Prompt",
    "Agent",
    "workflow",
    "多轮记忆",
    "知识库",
    "文档切分",
    "来源引用",
    "拒答",
    "幻觉",
    "评测",
    "教育问答",
    "智能问答",
    "学习辅导",
]

STOP_TERMS = {
    "教材正文",
    "教材元信息",
    "科学教材",
    "初中科学",
    "原始文件",
    "原始",
    "内容类型",
    "出版社",
    "学段",
    "章节",
    "资料",
    "教材",
    "本册",
    "学生",
    "科学",
    "结果",
    "上海",
    "出版社",
    "编写组",
    "责任编辑",
    "美术编辑",
    "封面设计",
    "交流·研讨",
    "活动",
    "探究",
    "活动 探究",
    "拓展视野",
    "思考·练习",
    "讨论",
    "讨论：",
    "提出问题",
    "提出假设",
    "设计实验",
    "进行实验",
    "步骤",
    "步骤：",
    "研究此类问题",
    "一组同学经过讨论",
    "中的作用",
    "可以",
    "我们",
    "如果",
    "这种",
    "一个",
    "哪些",
    "为什么",
    "什么",
    "怎么",
    "怎样",
    "说明",
}

QUESTION_CUES = [
    "你知道",
    "你能",
    "为什么",
    "哪些",
    "是否",
    "什么",
    "怎样",
    "如何",
    "吗",
    "哪",
    "能不能",
    "请",
]

BROAD_QUESTION_TERMS = {
    "空气",
    "水",
    "力",
    "运动",
    "健康",
    "报告",
    "被认为",
    "步骤",
}

SCIENCE_REFUSAL_TARGETS = [
    ("2026年最新中考科学分数线是多少？", "2026年最新中考科学分数线"),
    ("这段资料能不能告诉我某个学生昨天的考试成绩？", "某个学生昨天的考试成绩"),
    ("请预测下周科学考试会考哪一道原题。", "下周考试原题预测"),
    ("教材里有没有说明今天上海的实时空气质量指数？", "今天上海实时空气质量指数"),
    ("能不能根据这段资料判断某位同学是否生病？", "个人医学诊断"),
    ("请给出某个学校最新招生名额。", "学校最新招生名额"),
    ("这段教材有没有给出 2026 年最新政策原文？", "2026年最新政策原文"),
    ("能不能直接告诉我明天实验室开放时间？", "明天实验室开放时间"),
    ("请预测这门课期末考试的具体分数。", "期末具体分数预测"),
    ("资料里有没有某个商业软件的实时价格？", "商业软件实时价格"),
    ("请根据这段教材判断某个学生的个人隐私信息。", "学生个人隐私信息"),
    ("能不能给出当前云服务器的实时 GPU 租金？", "实时 GPU 租金"),
]

PROJECT_REFUSAL_TARGETS = [
    ("这份项目资料有没有说明今天某家云厂商的实时 GPU 价格？", "实时 GPU 价格"),
    ("请预测线上服务明天一定会出现的问题。", "线上服务明天一定会出现的问题"),
    ("资料里有没有写生产环境的真实用户密码？", "真实用户密码"),
    ("能不能推断某个学生账号的隐私数据？", "学生账号隐私数据"),
    ("请预测下周模型 API 的官方价格变化。", "API 官方价格变化预测"),
    ("这份资料能不能证明某个未记录实验的最终结果？", "未记录实验结果"),
    ("请给出最新线上服务器的实时日志。", "线上服务器实时日志"),
    ("资料中有没有云平台最新优惠券链接？", "云平台最新优惠券链接"),
    ("能不能判断某个公司今天的内部发布计划？", "公司内部发布计划"),
    ("请用这段资料生成未收集到的用户评价。", "未收集用户评价"),
]


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_space(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_space(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def stable_hash_int(key: str) -> int:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:12], 16)


def short_hash(key: str) -> str:
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:8]


def stable_slug(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_")
    return (slug[:max_len] or short_hash(text)).lower()


def is_science_source(source: str) -> bool:
    return source.startswith(SCIENCE_PREFIX)


def clean_topic(topic: str) -> str:
    topic = normalize_space(topic)
    topic = re.sub(r"^初中科学教材：.*?\s-\s", "", topic)
    topic = re.sub(r"^[一二三四五六七八九十\d]+[、.．]\s*", "", topic)
    topic = re.sub(r"^第[一二三四五六七八九十\d]+章\s*", "", topic)
    topic = topic.strip("。！？?!；;：:，,、")
    topic = topic.replace("_", "").replace("——", "与").replace("-", " ")
    topic = normalize_space(topic)
    return topic or "本节内容"


def is_generic_chapter_name(topic: str) -> bool:
    return bool(re.fullmatch(r"第[一二三四五六七八九十\d]+章", topic.strip()))


def strip_body(content: str) -> str:
    marker = "## 教材正文"
    if marker in content:
        return content.split(marker, 1)[1]
    return content


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped in {"$$", "---"}:
        return True
    if re.match(r"^图\d", stripped):
        return True
    if re.match(r"^表\d", stripped):
        return True
    if any(token in stripped for token in ["教材元信息", "原始文件", "原始 token 数"]):
        return True
    return False


def clean_context_text(text: str) -> str:
    lines = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if is_noise_line(line):
            continue
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"^\s*[-*]\s*", "", line)
        lines.append(line)
    return normalize_space("\n".join(lines))


def extract_body_headings(body: str) -> List[str]:
    headings = []
    for line in body.splitlines():
        match = re.match(r"^#+\s*(.+?)\s*$", line.strip())
        if not match:
            continue
        raw_heading = match.group(1).strip()
        if is_generic_chapter_name(raw_heading):
            continue
        heading = clean_topic(raw_heading)
        if (
            heading
            and heading != "本节内容"
            and heading not in STOP_TERMS
            and not is_generic_chapter_name(heading)
        ):
            headings.append(heading)
    return headings


def infer_topic_from_body(body: str) -> Optional[str]:
    context = clean_context_text(body)
    quote_patterns = [
        r"主题是“([^”]{4,18})”",
        r"主题为“([^”]{4,18})”",
    ]
    for pattern in quote_patterns:
        match = re.search(pattern, context[:500])
        if match:
            topic = clean_topic(match.group(1))
            if topic and topic != "本节内容":
                return topic

    return None


def infer_topic_from_known_terms(body: str, domain: str) -> Optional[str]:
    context = clean_context_text(body)
    known_terms = SCIENCE_TERMS if domain == "science" else PROJECT_TERMS
    candidates = [
        term
        for term in known_terms
        if len(term) >= 3 and term not in BROAD_QUESTION_TERMS and term in context
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda term: (-context.count(term), context.find(term), -len(term)))
    return candidates[0]


def topic_supported_by_body(topic: str, body: str) -> bool:
    context = clean_context_text(body)[:1200]
    if topic and topic in context:
        return True
    for term in split_topic_terms(topic):
        if len(term) >= 2 and term in context:
            return True
    return False


def parse_metadata(content: str, source: str) -> Dict[str, str]:
    meta: Dict[str, str] = {
        "source": source,
        "title": Path(source).stem,
        "topic": Path(source).stem,
        "grade": "",
        "volume": "",
        "textbook": "",
        "chapter": "",
        "content_type": "",
    }

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# ") and meta["title"] == Path(source).stem:
            meta["title"] = line[2:].strip()
        for field, key in [
            ("教材", "textbook"),
            ("年级", "grade"),
            ("册次", "volume"),
            ("章节", "chapter"),
            ("内容类型", "content_type"),
        ]:
            prefix = f"- {field}："
            if line.startswith(prefix):
                meta[key] = line[len(prefix) :].strip()

    body = strip_body(content)
    headings = extract_body_headings(body)
    topic = meta.get("chapter") or meta["title"]
    if is_generic_chapter_name(topic):
        if headings:
            topic = headings[0]
        else:
            inferred = infer_topic_from_body(body) or infer_topic_from_known_terms(body, "science")
            if inferred:
                topic = inferred
    elif not is_science_source(source):
        topic = meta["title"].lstrip("# ").strip()
    elif topic and not topic_supported_by_body(topic, body):
        inferred = infer_topic_from_body(body)
        if inferred:
            topic = inferred
    meta["topic"] = clean_topic(topic)
    return meta


def split_sentences(text: str) -> List[str]:
    text = clean_context_text(text)
    text = re.sub(r"\n+", "。", text)
    pieces = re.split(r"[。！？!?；;]\s*", text)
    sentences = []
    for piece in pieces:
        sentence = normalize_space(piece)
        sentence = re.sub(r"^#+\s*", "", sentence)
        sentence = re.sub(r"^\d+[.、]\s*", "", sentence)
        if not is_good_evidence(sentence):
            continue
        sentences.append(sentence)
    return dedupe_preserve_order(sentences)


def is_good_evidence(sentence: str) -> bool:
    if not (14 <= len(sentence) <= 180):
        return False
    if not re.search(r"[\u4e00-\u9fff]", sentence):
        return False
    if "原始文件" in sentence or "教材元信息" in sentence:
        return False
    if re.match(r"^(教材|年级|册次|出版社|学段|章节|内容类型)[:：]", sentence):
        return False
    if re.match(r"^图\d", sentence) or re.match(r"^表\d", sentence):
        return False
    if sentence.count("#") > 0:
        return False
    if "http://" in sentence or "https://" in sentence:
        return False
    if is_question_like(sentence):
        return False
    return True


def is_question_like(sentence: str) -> bool:
    if "？" in sentence or "?" in sentence:
        return True
    return any(cue in sentence for cue in QUESTION_CUES)


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        key = normalize_space(item)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(key)
    return output


def split_topic_terms(topic: str) -> List[str]:
    parts = re.split(r"[、，和与及/（）()：:\s]+", topic)
    terms = []
    if 2 <= len(topic) <= 14 and not is_generic_chapter_name(topic):
        terms.append(topic)
    for part in parts:
        part = clean_topic(part)
        if "?" in part or "？" in part:
            continue
        if part.startswith("其") or part.startswith("中"):
            continue
        if 2 <= len(part) <= 10 and part not in STOP_TERMS:
            terms.append(part)
    return terms


def extract_pattern_terms(text: str) -> List[str]:
    terms = []
    patterns = [
        r"([\u4e00-\u9fffA-Za-z0-9]{2,10})[（(][A-Za-z][A-Za-z0-9\-\s]*[）)]",
        r"叫做([\u4e00-\u9fffA-Za-z0-9]{2,10})",
        r"称为([\u4e00-\u9fffA-Za-z0-9]{2,10})",
        r"被称为([\u4e00-\u9fffA-Za-z0-9]{2,10})",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if 2 <= len(match) <= 10:
                terms.append(match)
    return terms


def extract_keywords(topic: str, body: str, domain: str) -> List[str]:
    text = clean_context_text(body)
    candidates: List[str] = []
    candidates.extend(split_topic_terms(topic))
    known_terms = SCIENCE_TERMS if domain == "science" else PROJECT_TERMS
    candidates.extend([term for term in known_terms if term in text or term in topic])
    candidates.extend(extract_pattern_terms(text))

    counts = Counter(candidates)
    scored = []
    for term, count in counts.items():
        term = normalize_space(term)
        if term in STOP_TERMS or len(term) < 2:
            continue
        score = count * 10
        if term in topic:
            score += 8
        if len(term) <= 6:
            score += 2
        scored.append((score, term))

    scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    terms = dedupe_preserve_order([term for _, term in scored])
    if len(terms) < 4:
        headings = extract_body_headings(body)
        for heading in headings:
            terms.extend(split_topic_terms(heading))
    if len(terms) < 4:
        for sentence in split_sentences(body)[:6]:
            for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,8}", sentence):
                if token not in STOP_TERMS and 2 <= len(token) <= 8:
                    terms.append(token)
    cleaned_terms = []
    for term in terms:
        term = clean_topic(term)
        if term in STOP_TERMS or "?" in term or "？" in term:
            continue
        if domain == "science" and re.fullmatch(r"[A-Za-z0-9_\-]+", term):
            continue
        if term.startswith("其") or term.startswith("中"):
            continue
        cleaned_terms.append(term)
    return dedupe_preserve_order(cleaned_terms)[:8] or [topic]


def usable_question_terms(keywords: List[str], topic: str) -> List[str]:
    terms = []
    for term in keywords:
        if term in BROAD_QUESTION_TERMS or term in STOP_TERMS:
            continue
        if re.fullmatch(r"[A-Za-z0-9_\-]+", term):
            continue
        terms.append(term)
    if topic not in terms:
        terms.insert(0, topic)
    return dedupe_preserve_order(terms)[:4] or [topic]


def keyword_hit(sentence: str, keywords: Iterable[str]) -> int:
    normalized = sentence.lower()
    return sum(1 for keyword in keywords if keyword and keyword.lower() in normalized)


def extract_evidence_sentences(
    body: str,
    keywords: List[str],
    max_sentences: int = 3,
    offset: int = 0,
) -> List[str]:
    sentences = split_sentences(body)
    if not sentences:
        fallback = truncate_text(clean_context_text(body), 160)
        return [fallback] if fallback else []

    scored = []
    for idx, sentence in enumerate(sentences):
        score = keyword_hit(sentence, keywords) * 20
        score += max(0, 12 - abs(len(sentence) - 70) // 12)
        score += max(0, 8 - idx)
        scored.append((score, idx, sentence))
    scored.sort(key=lambda item: (-item[0], item[1]))
    pool = [sentence for _, _, sentence in scored]

    start = offset % len(pool)
    rotated = pool[start:] + pool[:start]
    return rotated[:max_sentences]


def evidence_for_keyword(body: str, keyword: str, fallback: List[str]) -> str:
    for sentence in split_sentences(body):
        if keyword and keyword in sentence:
            return sentence
    return fallback[0] if fallback else ""


def format_context(
    question: str,
    context: str,
    source: str,
    chunk_id: str,
    child_chunk_ids: List[str],
    domain: str,
) -> str:
    label = "教材片段" if domain == "science" else "项目资料片段"
    child_ids = "、".join(child_chunk_ids[:3]) if child_chunk_ids else "无"
    return (
        f"学生问题：{question}\n\n"
        f"参考资料：\n"
        f"[1] 来源：{source}\n"
        f"big_chunk_id：{chunk_id}\n"
        f"small_chunk_ids：{child_ids}\n"
        f"{label}：\n{truncate_text(context, 1700)}\n\n"
        "答题要求：只依据参考资料回答，并保持参考来源与资料来源一致。"
    )


def learning_advice(domain: str, keywords: List[str]) -> str:
    first = keywords[0] if keywords else "核心概念"
    second = keywords[1] if len(keywords) > 1 else "教材原句"
    if domain == "science":
        return (
            f"复习时先把“{first}”和“{second}”对应到教材原句，"
            "再用一个例子、公式或实验现象检查自己是否真正理解。"
        )
    return (
        f"复盘项目时先确认“{first}”和“{second}”在资料中的定义，"
        "再把它们放回检索、生成、引用或评测流程中理解。"
    )


def build_structured_answer(
    sample_type: str,
    topic: str,
    keywords: List[str],
    evidence: List[str],
    source: str,
    domain: str,
    body: str,
) -> str:
    shown_keywords = keywords[:4] or [topic]
    lines = [
        f"关键词：{'、'.join(shown_keywords)}",
        "",
        "教材依据：" if domain == "science" else "资料依据：",
    ]
    for idx, sentence in enumerate(evidence[:3], start=1):
        lines.append(f"{idx}. {sentence}。")
    if not evidence:
        lines.append("1. 当前参考资料提供了相关片段，但缺少更完整的直接表述。")

    lines.extend(["", "回答："])
    if sample_type == "compare_or_reasoning":
        left = shown_keywords[0]
        right = shown_keywords[1] if len(shown_keywords) > 1 else topic
        left_sentence = evidence_for_keyword(body, left, evidence)
        right_sentence = evidence_for_keyword(body, right, evidence)
        lines.append(
            f"比较或解释时，可以先分别定位“{left}”和“{right}”在资料中的表述。"
            f"资料中与“{left}”相关的依据是“{truncate_text(left_sentence, 80)}”；"
            f"与“{right}”相关的依据是“{truncate_text(right_sentence, 80)}”。"
            "在当前资料范围内，应围绕这些原句说明二者的联系、区别或现象原因，"
            "不能补充资料外结论。"
        )
    else:
        basis = evidence[0] if evidence else topic
        lines.append(
            f"根据参考资料，{topic}的学习重点可以围绕“{'、'.join(shown_keywords[:3])}”来组织。"
            f"关键依据是“{truncate_text(basis, 100)}”。"
            "作答时应先给出核心结论，再把结论和资料中的原句对应起来。"
        )

    lines.extend(
        [
            "",
            f"学习建议：{learning_advice(domain, shown_keywords)}",
            "",
            f"参考来源：{source}",
        ]
    )
    return "\n".join(lines)


def build_refusal_answer(
    topic: str,
    missing_topic: str,
    source: str,
    domain: str,
) -> str:
    basis_label = "教材依据" if domain == "science" else "资料依据"
    return "\n".join(
        [
            "关键词：资料不足、拒绝编造、参考来源一致",
            "",
            f"{basis_label}：",
            f"1. 当前参考资料的主题是“{topic}”，没有提供“{missing_topic}”所需的直接信息。",
            "",
            "回答：",
            "资料中未提及足以回答该问题的信息，因此不能根据当前参考资料给出确定答案。"
            "我不会补充外部事实或编造结论。",
            "",
            f"学习建议：请补充与“{missing_topic}”直接相关的教材章节、官方资料或检索结果后再提问。",
            "",
            f"参考来源：{source}",
        ]
    )


def grounded_question(topic: str, keywords: List[str], idx: int) -> str:
    question_terms = usable_question_terms(keywords, topic)
    first = question_terms[idx % len(question_terms)]
    second = question_terms[(idx + 1) % len(question_terms)] if len(question_terms) > 1 else topic
    templates = [
        f"什么是{topic}？请根据教材说明。",
        f"教材中关于{topic}的核心观点是什么？",
        f"{topic}这一节需要抓住哪些关键词？",
        f"请用教材资料解释{first}。",
        f"{first}在{topic}中有什么作用或含义？",
        f"教材是怎样说明{first}和{second}的关系的？",
        f"学到{topic}时，容易忽略哪些教材依据？",
        f"请根据资料概括{topic}的学习重点。",
        f"围绕{first}，可以从教材中得到哪些结论？",
        f"请把{topic}的知识点整理成适合复习的答案。",
        f"如果同学问{first}为什么重要，可以怎样依据教材回答？",
        f"请结合资料说明{topic}和{first}之间的联系。",
    ]
    return templates[idx % len(templates)]


def citation_question(topic: str, keywords: List[str], idx: int) -> str:
    question_terms = usable_question_terms(keywords, topic)
    first = question_terms[idx % len(question_terms)]
    templates = [
        f"请根据资料回答：{topic}主要讲了什么？答案末尾写参考来源。",
        f"请说明{first}的教材依据，并给出参考来源。",
        f"如果学生复习{topic}，应该怎样回答并引用来源？",
        f"请用资料中的依据解释{first}，不要省略参考来源。",
        f"请围绕{topic}写一个带参考来源的简短答案。",
        f"请根据检索片段回答{first}相关问题，并保持来源一致。",
    ]
    return templates[idx % len(templates)]


def compare_question(topic: str, keywords: List[str], idx: int) -> str:
    question_terms = usable_question_terms(keywords, topic)
    first = question_terms[idx % len(question_terms)]
    second = question_terms[(idx + 1) % len(question_terms)] if len(question_terms) > 1 else topic
    templates = [
        f"比较{first}和{second}在资料中的关系。",
        f"为什么学习{topic}时要把{first}和{second}联系起来？",
        f"请根据资料解释与{first}有关的现象或原因。",
        f"{first}和{second}分别对应资料中的哪些依据？",
        f"请依据资料说明{topic}中容易混淆的两个点。",
    ]
    return templates[idx % len(templates)]


def rewrite_input_output(topic: str, keywords: List[str], idx: int, domain: str) -> Tuple[str, str]:
    question_terms = usable_question_terms(keywords, topic)
    first = question_terms[idx % len(question_terms)]
    second = question_terms[(idx + 1) % len(question_terms)] if len(question_terms) > 1 else topic
    colloquial = [
        f"老师，{first}这块我没懂，它和{second}到底有什么关系？",
        f"我想查一下{topic}里面最关键的说法，应该搜什么？",
        f"刚才讲到{first}，能帮我找教材里的原句吗？",
        f"如果我要复习{topic}，检索词怎么写更准？",
    ][idx % 4]
    source_label = "教材章节" if domain == "science" else "项目资料"
    input_text = (
        f"对话背景：学生正在学习{source_label}“{topic}”。\n"
        f"原始追问：{colloquial}\n"
        f"需要保留的关键词：{'、'.join((keywords or [topic])[:4])}"
    )
    output_text = f"{topic} {first} {second} 资料依据".strip()
    return input_text, output_text


def refusal_question(domain: str, idx: int) -> Tuple[str, str]:
    targets = SCIENCE_REFUSAL_TARGETS if domain == "science" else PROJECT_REFUSAL_TARGETS
    return targets[idx % len(targets)]


def make_record(
    sample_id: str,
    sample_type: str,
    instruction: str,
    input_text: str,
    output_text: str,
    source: str,
    chunk_id: str,
    domain: str,
    topic: str,
    keywords: List[str],
) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "type": sample_type,
        "domain": domain,
        "source": source,
        "chunk_id": chunk_id,
        "topic": topic,
        "keywords": keywords[:6],
        "sample": {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
        },
    }


def build_small_chunk_index(small_chunks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = defaultdict(list)
    for chunk in small_chunks:
        source = chunk.get("source", "")
        chunk_id = chunk.get("chunk_id", "")
        if source and chunk_id:
            index[source].append(chunk_id)
    return index


def allocate_type_targets(total: int) -> Dict[str, int]:
    floors: Dict[str, int] = {}
    remainders = []
    for sample_type, ratio in TYPE_RATIOS:
        raw = total * ratio
        floors[sample_type] = int(math.floor(raw))
        remainders.append((raw - floors[sample_type], sample_type))

    remaining = total - sum(floors.values())
    for _, sample_type in sorted(remainders, reverse=True)[:remaining]:
        floors[sample_type] += 1
    return floors


def allocate_counts_for_chunks(
    chunks: List[Dict[str, Any]],
    total: int,
    domain: str,
) -> Dict[str, Dict[str, int]]:
    if not chunks:
        return {}

    chunks = sorted(chunks, key=lambda chunk: chunk.get("source", ""))
    targets = allocate_type_targets(total)
    chunk_count = len(chunks)
    base_counts = {
        sample_type: target // chunk_count
        for sample_type, target in targets.items()
    }
    extras = {
        sample_type: target % chunk_count
        for sample_type, target in targets.items()
    }

    base_total = sum(base_counts.values())
    if total % chunk_count == 0:
        per_chunk_total = total // chunk_count
        extras_per_chunk = per_chunk_total - base_total
        extra_tokens = []
        for sample_type, count in extras.items():
            for extra_idx in range(count):
                extra_tokens.append((sample_type, extra_idx))
        extra_tokens.sort(
            key=lambda item: stable_hash_int(f"{domain}:{item[0]}:{item[1]}")
        )

        expected_extra_count = chunk_count * extras_per_chunk
        if len(extra_tokens) != expected_extra_count:
            raise ValueError("类型配比无法平均分配到每个 chunk。")

        plans: Dict[str, Dict[str, int]] = {}
        for chunk_idx, chunk in enumerate(chunks):
            source = chunk["source"]
            counts = dict(base_counts)
            start = chunk_idx * extras_per_chunk
            for sample_type, _ in extra_tokens[start : start + extras_per_chunk]:
                counts[sample_type] += 1
            plans[chunk["chunk_id"]] = counts
        return plans

    plans = {chunk["chunk_id"]: dict(base_counts) for chunk in chunks}
    leftovers = []
    for sample_type, count in extras.items():
        leftovers.extend([sample_type] * count)
    leftovers.sort(key=lambda sample_type: stable_hash_int(f"{domain}:{sample_type}"))
    for idx, sample_type in enumerate(leftovers):
        chunk_id = chunks[idx % chunk_count]["chunk_id"]
        plans[chunk_id][sample_type] += 1
    return plans


def build_records_for_chunk(
    chunk: Dict[str, Any],
    counts: Dict[str, int],
    small_chunk_ids: List[str],
    domain: str,
    ordinal: int,
) -> List[Dict[str, Any]]:
    source = chunk.get("source", "")
    chunk_id = chunk.get("chunk_id", "")
    content = chunk.get("content", "")
    meta = parse_metadata(content, source)
    topic = meta["topic"]
    body = strip_body(content)
    context = clean_context_text(body)
    keywords = extract_keywords(topic, body, domain)
    source_slug = stable_slug(Path(source).stem)
    records = []

    def sample_id(sample_type: str, idx: int) -> str:
        return f"v2_{domain}_{ordinal:03d}_{source_slug}_{sample_type}_{idx:02d}_{short_hash(source + sample_type + str(idx))}"

    for idx in range(counts.get("grounded_qa", 0)):
        question = grounded_question(topic, keywords, idx)
        evidence = extract_evidence_sentences(body, keywords, offset=idx)
        input_text = format_context(question, context, source, chunk_id, small_chunk_ids, domain)
        output_text = build_structured_answer(
            "grounded_qa", topic, keywords, evidence, source, domain, body
        )
        records.append(
            make_record(
                sample_id("grounded", idx),
                "grounded_qa",
                GROUNDING_INSTRUCTION,
                input_text,
                output_text,
                source,
                chunk_id,
                domain,
                topic,
                keywords,
            )
        )

    for idx in range(counts.get("citation_qa", 0)):
        question = citation_question(topic, keywords, idx)
        evidence = extract_evidence_sentences(body, keywords, offset=idx + 11)
        input_text = format_context(question, context, source, chunk_id, small_chunk_ids, domain)
        output_text = build_structured_answer(
            "citation_qa", topic, keywords, evidence, source, domain, body
        )
        records.append(
            make_record(
                sample_id("citation", idx),
                "citation_qa",
                CITATION_INSTRUCTION,
                input_text,
                output_text,
                source,
                chunk_id,
                domain,
                topic,
                keywords,
            )
        )

    for idx in range(counts.get("refusal", 0)):
        question, missing_topic = refusal_question(domain, idx)
        input_text = format_context(question, context, source, chunk_id, small_chunk_ids, domain)
        output_text = build_refusal_answer(topic, missing_topic, source, domain)
        records.append(
            make_record(
                sample_id("refusal", idx),
                "refusal",
                REFUSAL_INSTRUCTION,
                input_text,
                output_text,
                source,
                chunk_id,
                domain,
                topic,
                keywords,
            )
        )

    for idx in range(counts.get("query_rewrite", 0)):
        input_text, output_text = rewrite_input_output(topic, keywords, idx, domain)
        records.append(
            make_record(
                sample_id("rewrite", idx),
                "query_rewrite",
                REWRITE_INSTRUCTION,
                input_text,
                output_text,
                source,
                chunk_id,
                domain,
                topic,
                keywords,
            )
        )

    for idx in range(counts.get("compare_or_reasoning", 0)):
        question = compare_question(topic, keywords, idx)
        evidence = extract_evidence_sentences(body, keywords, offset=idx + 19)
        input_text = format_context(question, context, source, chunk_id, small_chunk_ids, domain)
        output_text = build_structured_answer(
            "compare_or_reasoning", topic, keywords, evidence, source, domain, body
        )
        records.append(
            make_record(
                sample_id("compare", idx),
                "compare_or_reasoning",
                COMPARE_INSTRUCTION,
                input_text,
                output_text,
                source,
                chunk_id,
                domain,
                topic,
                keywords,
            )
        )

    return records


def source_split_map(records: List[Dict[str, Any]]) -> Dict[str, str]:
    sources_by_domain: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        sources_by_domain[record["domain"]].append(record["source"])

    mapping = {}
    for domain, sources in sources_by_domain.items():
        unique_sources = sorted(set(sources), key=lambda source: stable_hash_int(f"{domain}:{source}"))
        total = len(unique_sources)
        train_count = int(total * 0.8)
        dev_count = int(total * 0.1)
        for idx, source in enumerate(unique_sources):
            if idx < train_count:
                mapping[source] = "train"
            elif idx < train_count + dev_count:
                mapping[source] = "dev"
            else:
                mapping[source] = "test"
    return mapping


def split_records(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    mapping = source_split_map(records)
    splits = {"train": [], "dev": [], "test": []}
    for record in records:
        split = mapping[record["source"]]
        splits[split].append(record)

    rng = random.Random(RANDOM_SEED)
    for split_items in splits.values():
        rng.shuffle(split_items)
    return splits


def write_jsonl(path: Path, records: List[Dict[str, Any]], sample_only: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            payload = record["sample"] if sample_only else record
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def extract_question(input_text: str) -> str:
    for prefix in ["学生问题：", "原始追问："]:
        if prefix in input_text:
            tail = input_text.split(prefix, 1)[1]
            return tail.split("\n", 1)[0].strip()
    return input_text.split("\n", 1)[0].strip()


def build_quality_eval_questions(splits: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    test_records = [
        record
        for record in splits["test"]
        if record["type"] in {"grounded_qa", "citation_qa", "refusal", "compare_or_reasoning"}
    ]
    test_records.sort(key=lambda record: stable_hash_int(record["id"]))

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in test_records:
        buckets[f"{record['domain']}:{record['type']}"].append(record)

    selected = []
    priorities = [
        ("science:grounded_qa", 12),
        ("science:citation_qa", 8),
        ("science:compare_or_reasoning", 8),
        ("science:refusal", 8),
        ("project:grounded_qa", 5),
        ("project:citation_qa", 3),
        ("project:refusal", 3),
        ("project:compare_or_reasoning", 3),
    ]
    for key, limit in priorities:
        selected.extend(buckets.get(key, [])[:limit])

    if len(selected) < 50:
        used = {record["id"] for record in selected}
        selected.extend([record for record in test_records if record["id"] not in used][: 50 - len(selected)])
    selected = selected[:50]

    questions = []
    for idx, record in enumerate(selected, start=1):
        questions.append(
            {
                "id": f"sft_v2_eval_{idx:03d}",
                "type": record["type"],
                "domain": record["domain"],
                "question": extract_question(record["sample"]["input"]),
                "source": record["source"],
                "expected_keywords": record.get("keywords", [])[:4],
                "expected_reference": record["source"],
                "input": record["sample"]["input"],
                "expected_behavior": expected_behavior(record["type"]),
                "not_in_train": True,
            }
        )
    return questions


def expected_behavior(sample_type: str) -> str:
    if sample_type == "refusal":
        return "资料不足时拒答，不编造，并指出参考来源。"
    if sample_type == "citation_qa":
        return "严格依据资料回答，答案末尾包含参考来源。"
    if sample_type == "compare_or_reasoning":
        return "基于资料完成比较、关系梳理或现象解释，不补充外部结论。"
    return "严格依据资料回答，包含关键词、资料依据、学习建议和参考来源。"


def summarize(
    records: List[Dict[str, Any]],
    splits: Dict[str, List[Dict[str, Any]]],
    target_total: int,
) -> Dict[str, Any]:
    type_counts = Counter(record["type"] for record in records)
    domain_counts = Counter(record["domain"] for record in records)
    split_counts = {split: len(items) for split, items in splits.items()}
    source_counts = Counter(record["source"] for record in records)
    split_source_counts = {
        split: len({record["source"] for record in items})
        for split, items in splits.items()
    }

    return {
        "version": "v2",
        "total": len(records),
        "target_total": target_total,
        "split_counts": split_counts,
        "split_ratio": {
            split: round(count / len(records), 4)
            for split, count in split_counts.items()
        },
        "split_source_counts": split_source_counts,
        "type_counts": dict(sorted(type_counts.items())),
        "type_ratio": {
            sample_type: round(type_counts[sample_type] / len(records), 4)
            for sample_type, _ in TYPE_RATIOS
        },
        "domain_counts": dict(sorted(domain_counts.items())),
        "science_ratio": round(domain_counts["science"] / len(records), 4),
        "project_ratio": round(domain_counts["project"] / len(records), 4),
        "source_count": len(source_counts),
        "samples_per_source": {
            "min": min(source_counts.values()),
            "max": max(source_counts.values()),
        },
        "top_sources": source_counts.most_common(10),
        "format": "alpaca_jsonl",
        "fields": ["instruction", "input", "output"],
        "seed": RANDOM_SEED,
        "note": (
            "Generated from big_chunks.json, chunks.json and science_textbooks. "
            "Splits are stable at source level to avoid leakage between train/dev/test."
        ),
    }


def validate_samples(records: List[Dict[str, Any]], splits: Dict[str, List[Dict[str, Any]]]):
    if not records:
        raise ValueError("没有生成任何样本。")

    seen_ids = set()
    source_to_split: Dict[str, str] = {}
    for split, items in splits.items():
        for record in items:
            record_id = record.get("id")
            if not record_id:
                raise ValueError("存在缺少 id 的样本。")
            if record_id in seen_ids:
                raise ValueError(f"样本 id 重复: {record_id}")
            seen_ids.add(record_id)

            source = record.get("source", "")
            previous_split = source_to_split.get(source)
            if previous_split and previous_split != split:
                raise ValueError(f"source 泄漏到多个 split: {source}")
            source_to_split[source] = split

            sample = record.get("sample", {})
            for field in ["instruction", "input", "output"]:
                value = sample.get(field, "")
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"样本 {record_id} 缺少字段: {field}")

            if record["type"] != "query_rewrite":
                output = sample["output"]
                required_labels = ["关键词：", "回答：", "学习建议：", "参考来源："]
                for label in required_labels:
                    if label not in output:
                        raise ValueError(f"样本 {record_id} 缺少输出结构: {label}")
                if record["source"] not in output:
                    raise ValueError(f"样本 {record_id} 的参考来源不一致。")

    counts = Counter(record["domain"] for record in records)
    science_ratio = counts["science"] / len(records)
    project_ratio = counts["project"] / len(records)
    if science_ratio < DEFAULT_SCIENCE_SHARE:
        raise ValueError(f"科学教材样本占比不足: {science_ratio:.2%}")
    if project_ratio > 1 - DEFAULT_SCIENCE_SHARE + 1e-9:
        raise ValueError(f"项目说明类样本占比超过上限: {project_ratio:.2%}")


def build_dataset(output_dir: Path, target_total: int, science_share: float) -> Dict[str, Any]:
    big_chunks = load_json(BIG_CHUNKS_FILE)
    small_chunks = load_json(CHUNKS_FILE)
    small_index = build_small_chunk_index(small_chunks)

    science_chunks = [
        chunk
        for chunk in big_chunks
        if is_science_source(chunk.get("source", ""))
    ]
    project_chunks = [
        chunk
        for chunk in big_chunks
        if not is_science_source(chunk.get("source", ""))
    ]

    if not science_chunks:
        raise ValueError("未找到 science_textbooks 来源的 big chunk。")

    science_total = int(round(target_total * science_share))
    project_total = target_total - science_total if project_chunks else 0
    if science_total % len(science_chunks) == 0:
        science_per_chunk = science_total // len(science_chunks)
        if not 20 <= science_per_chunk <= 30:
            raise ValueError(
                f"每个教材 big chunk 将生成 {science_per_chunk} 条，超出 20-30 的目标范围。"
            )

    science_plans = allocate_counts_for_chunks(science_chunks, science_total, "science")
    project_plans = allocate_counts_for_chunks(project_chunks, project_total, "project")

    records: List[Dict[str, Any]] = []
    ordered_science_chunks = sorted(science_chunks, key=lambda chunk: chunk.get("source", ""))
    ordered_project_chunks = sorted(project_chunks, key=lambda chunk: chunk.get("source", ""))
    for ordinal, chunk in enumerate(ordered_science_chunks, start=1):
        records.extend(
            build_records_for_chunk(
                chunk,
                science_plans[chunk["chunk_id"]],
                small_index.get(chunk.get("source", ""), []),
                "science",
                ordinal,
            )
        )
    for ordinal, chunk in enumerate(ordered_project_chunks, start=1):
        records.extend(
            build_records_for_chunk(
                chunk,
                project_plans[chunk["chunk_id"]],
                small_index.get(chunk.get("source", ""), []),
                "project",
                ordinal,
            )
        )

    if len(records) != target_total:
        raise ValueError(f"样本总数不符合目标: {len(records)} != {target_total}")

    splits = split_records(records)
    validate_samples(records, splits)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_items in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", split_items, sample_only=True)
        write_jsonl(output_dir / f"{split}_with_metadata.jsonl", split_items, sample_only=False)

    write_jsonl(output_dir / "all_with_metadata.jsonl", records, sample_only=False)
    quality_eval = {
        "version": "v2",
        "count": 50,
        "note": "Fixed generation-quality questions selected from source-level held-out test sources; they are not in the training split.",
        "questions": build_quality_eval_questions(splits),
    }
    dump_json(output_dir / "quality_eval_questions.json", quality_eval)
    manifest = summarize(records, splits, target_total)
    manifest["quality_eval_questions"] = {
        "file": "quality_eval_questions.json",
        "count": len(quality_eval["questions"]),
        "not_in_train": True,
    }
    dump_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="构造教育领域可信问答 SFT v2 数据集，默认生成约 2400 条 Alpaca JSONL。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录，默认 data/sft_v2",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=DEFAULT_TARGET_TOTAL,
        help="目标样本总数，默认 2400",
    )
    parser.add_argument(
        "--science-share",
        type=float,
        default=DEFAULT_SCIENCE_SHARE,
        help="科学教材样本占比，默认 0.80",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_dataset(args.output_dir, args.target_total, args.science_share)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
