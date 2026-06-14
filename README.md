# langgraph-shop-agent

基于 LangGraph 的智能导购 Agent。用户用自然语言描述需求，Agent 自动挖掘偏好、检索产品、对比推荐，并跨会话持久化用户画像。

```
 React 前端 ──SSE──→ FastAPI ──→ Agent (单例)
                                  ├─ 阶段分析 → 规则 + LLM 七阶段分类
                                  ├─ 画像检索 → ChromaDB 三级向量召回 + Redis 缓存
                                  ├─ 推理决策 → per-stage Prompt 动态注入 + 6 工具
                                  └─ 工具调用 → 工厂模式依赖注入
                                  └─ 画像存储 → SQLite KV + 时间衰减 + 语义向量
```

## 快速开始

```bash
# Docker
docker compose up

# 手动
pip install -e .
python -m uvicorn backend.main:app --port 8000
cd frontend && npm install && npm run dev
```

浏览器打开 `http://localhost:5173`。

## 项目结构

```
src/agent/       LangGraph 编排、per-stage Prompt、6 工具
src/retrieval/   ChromaDB 三级检索 + Redis 缓存
src/profile/     双层画像存储（SQLite + ChromaDB）
src/cache/       Redis RAG 缓存
backend/         FastAPI REST + SSE 流式
frontend/        React + TypeScript
tests/           38 tests
```

## License

MIT
