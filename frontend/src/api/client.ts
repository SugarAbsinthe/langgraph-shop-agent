/* ---- Thin fetch wrapper for ShopAgent API ---- */

const BASE = "/api";
const DEFAULT_TIMEOUT = 90_000; // 90s — agent init + LLM call can be slow

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT);

  try {
    const res = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      ...options,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  } finally {
    clearTimeout(timer);
  }
}

/* ---- Chat ---- */

export async function sendMessage(
  convId: string,
  question: string,
  chatHistory: { role: string; content: string }[],
) {
  return request<import("../types").ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({
      conv_id: convId,
      question,
      chat_history: chatHistory,
    }),
  });
}

/* ---- Conversations ---- */

export async function listConversations() {
  return request<import("../types").Conversation[]>("/conversations");
}

export async function createConversation(title = "新对话", model = "") {
  return request<import("../types").Conversation>("/conversations", {
    method: "POST",
    body: JSON.stringify({ title, model }),
  });
}

export async function deleteConversation(convId: string) {
  return request<{ deleted: string }>(`/conversations/${convId}`, {
    method: "DELETE",
  });
}

/* ---- Messages ---- */

export async function getMessages(convId: string) {
  return request<import("../types").MessageRecord[]>(
    `/conversations/${convId}/messages`,
  );
}

export async function saveMessage(
  convId: string,
  role: "user" | "assistant",
  content: string,
  details?: Record<string, unknown>,
) {
  return request<import("../types").MessageRecord>(
    `/conversations/${convId}/messages`,
    {
      method: "POST",
      body: JSON.stringify({ role, content, details }),
    },
  );
}

/* ---- Health ---- */

export async function healthCheck() {
  return request<import("../types").HealthResponse>("/health");
}
