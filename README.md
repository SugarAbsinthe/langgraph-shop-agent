# 🛒 ShopAgent — 基于 LangGraph 的智能导购对话机器人

一个 LangGraph Agent 工程的参考项目，展示从原型到前后端分离的完整演进过程。

## 做了什么

用户用自然语言描述需求（"预算 8000 打游戏，偶尔出差"），Agent 自动挖掘偏好、检索产品、对比推荐，并将用户画像持久化到下次对话复用。

核心链路：**阶段分析 → 画像检索 → 推理决策 ⇄ 工具调用**，四节点 StateGraph 管线。

## 快速开始

```bash
# 后端
pip install -e .
python -m uvicorn backend.main:app --port 8000

# 前端
cd frontend && npm install && npm run dev
```

浏览器打开 `http://localhost:5173`。

环境要求：Python 3.9+、Node 18+、本地需有 ChromaDB 数据和 Redis（可选）。

## 技术栈

`Python` · `LangGraph` · `LangChain` · `FastAPI` · `React` · `TypeScript` · `ChromaDB` · `Redis` · `SQLite`

## 架构

```
React 前端 → FastAPI 后端 → Agent 单例
                              ├─ 阶段分析节点（规则 + LLM 分类）
                              ├─ 画像检索节点（ChromaDB 三级召回 + Redis 缓存）
                              ├─ 推理决策节点（per-stage Prompt 注入 + 6 工具）
                              └─ 工具调用节点（工厂模式依赖注入）
```

## 核心设计

- **per-stage Prompt 注入**：按导购阶段动态加载聚焦指令，Token 消耗降低约 60%
- **Agentic RAG**：描述 → 规格 → 评价三级向量索引，逐层召回去重，Redis 缓存加速
- **双层画像存储**：结构化 KV（时间衰减自动淘汰）+ 语义向量（跨会话偏好召回）
- **单 Agent + 动态 Prompt**：评估过多 Agent 架构后回退，同样效果，更少复杂度

## 项目结构

```
src/agent/     LangGraph 编排、工具、Prompt
src/retrieval/ ChromaDB 三级检索 + 格式化
src/profile/   双层画像存储（SQLite + ChromaDB）
src/cache/     Redis RAG 缓存
src/db/        对话持久化
backend/       FastAPI REST API
frontend/      React + TypeScript UI
```

## License

MIT
