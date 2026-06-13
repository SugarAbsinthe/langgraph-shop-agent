# langgraph-shop-agent

基于 LangGraph 的智能导购对话机器人，支持多阶段需求挖掘、产品检索与个性化推荐。

## 快速开始

```bash
# 安装依赖
pip install -e .

# 启动后端
python -m uvicorn backend.main:app --port 8000

# 启动前端（新终端）
cd frontend && npm install && npm run dev
```

浏览器打开 `http://localhost:5173`。

> 需要本地有 ChromaDB 产品向量数据，Redis 为可选项。

## 项目结构

```
src/agent/     LangGraph 编排、工具、Prompt
src/retrieval/ 检索与向量索引
src/profile/   画像存储
src/cache/     Redis 缓存
src/db/        对话持久化
backend/       FastAPI
frontend/      React
```

## License

MIT
