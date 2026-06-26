# Engineering Experiment Report

- Generated at: 2026-06-17 10:16:25
- USE_MOCK: `False`
- LangGraph available: `True`

## 1. Agent Engine

- Questions: 4
- Pass rate: 100.00%
- LangGraph graph trace pass rate: 100.00%
- LangGraph avg skill trace count: 9.00

## 2. Session Memory API

- Pass rate: 100.00%
- Memory used accuracy: 100.00%
- Resolved question hit rate: 100.00%
- Session snapshot OK: `True`

## 3. Cache And Rate Limit

- Redis available: `True`
- Mode: `redis_available`
- Cache test: `passed`
- Rate limit test: `passed`

## 4. RAG API Regression

- Questions: 6
- Route OK rate: 100.00%
- Source hit rate: 100.00%
- Keyword hit rate: 100.00%
- Citation rate: 100.00%
- No-context reject rate: 100.00%
- Strict source miss count: 0

## Conclusion

本轮实验重点验证 LangGraph Agent 编排、session 级记忆、缓存限流运行状态和原 RAG API 回归稳定性。Redis 不可用时系统会 fail-open，不影响 Agent/RAG 主链路运行。
