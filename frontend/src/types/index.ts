/* ---- Chat ---- */

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  conv_id: string;
  question: string;
  chat_history: ChatMessage[];
}

export interface ChatResponse {
  answer: string;
  stage: string;
  product_context: string;
  user_profile: string;
  tool_rounds: number;
}

/* ---- Conversations ---- */

export interface Conversation {
  id: string;
  title: string;
  model: string;
  created_at: string;
  updated_at: string;
}

export interface MessageRecord {
  id: number;
  conv_id: string;
  role: string;
  content: string;
  details: string | null;
  created_at: string;
}

/* ---- Health ---- */

export interface ComponentStatus {
  llm: boolean;
  chromadb: boolean;
  redis: boolean;
  database: boolean;
}

export interface HealthResponse {
  status: string;
  components: ComponentStatus;
}

/* ---- Stage labels ---- */

export const STAGE_LABELS: Record<string, string> = {
  discovery: "👋 发现需求",
  needs_elicitation: "🔍 需求挖掘",
  search: "🛒 产品搜索",
  comparison: "⚖️ 产品对比",
  objection_handling: "💬 异议处理",
  recommendation: "🎯 最终推荐",
  summary: "✅ 总结收尾",
};

/* ---- Profile keys ---- */

export const PROFILE_KEY_LABELS: Record<string, string> = {
  budget: "预算",
  primary_use: "用途",
  preferred_brand: "偏好品牌",
  mobility: "便携需求",
  must_have: "刚需",
  exclude_brand: "排除品牌",
  screen_preference: "屏幕偏好",
  battery_requirement: "续航要求",
  product_category: "产品品类",
};
