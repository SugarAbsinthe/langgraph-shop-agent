import { useReducer, useCallback, useEffect } from "react";
import type { Conversation, ChatMessage, MessageRecord } from "../types";
import * as api from "../api/client";

/* ---- State ---- */

interface DisplayMessage {
  role: "user" | "assistant";
  content: string;
  details?: {
    stage?: string;
    tool_rounds?: number;
    has_product_context?: boolean;
  } | null;
}

interface AppState {
  conversations: Conversation[];
  currentConvId: string | null;
  messages: DisplayMessage[];
  lastStage: string;
  lastProfile: string;
  lastProductContext: string;
  lastToolRounds: number;
  isLoading: boolean;
  showDebug: boolean;
  error: string | null;
}

const initialState: AppState = {
  conversations: [],
  currentConvId: null,
  messages: [],
  lastStage: "discovery",
  lastProfile: "",
  lastProductContext: "",
  lastToolRounds: 0,
  isLoading: false,
  showDebug: false,
  error: null,
};

/* ---- Actions ---- */

type Action =
  | { type: "SET_CONVERSATIONS"; payload: Conversation[] }
  | { type: "SET_CURRENT_CONV"; payload: string }
  | { type: "SET_MESSAGES"; payload: DisplayMessage[] }
  | { type: "ADD_MESSAGE"; payload: DisplayMessage }
  | { type: "APPEND_TOKEN"; payload: string }
  | { type: "UPDATE_LAST_DETAILS"; payload: Record<string, unknown> }
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_RESPONSE"; payload: { stage: string; user_profile: string; product_context: string; tool_rounds: number } }
  | { type: "TOGGLE_DEBUG" }
  | { type: "SET_ERROR"; payload: string | null }
  | { type: "SET_STATUS"; payload: string | null }
  | { type: "REMOVE_CONVERSATION"; payload: string };

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "SET_CONVERSATIONS":
      return { ...state, conversations: action.payload };
    case "SET_CURRENT_CONV":
      return { ...state, currentConvId: action.payload, messages: [], error: null };
    case "SET_MESSAGES":
      return { ...state, messages: action.payload };
    case "ADD_MESSAGE":
      return { ...state, messages: [...state.messages, action.payload] };
    case "APPEND_TOKEN": {
      const msgs = [...state.messages];
      const last = msgs[msgs.length - 1];
      if (last && last.role === "assistant") {
        msgs[msgs.length - 1] = { ...last, content: last.content + action.payload };
      }
      return { ...state, messages: msgs };
    }
    case "UPDATE_LAST_DETAILS": {
      const msgs = [...state.messages];
      const last = msgs[msgs.length - 1];
      if (last && last.role === "assistant") {
        msgs[msgs.length - 1] = { ...last, details: { ...last.details, ...action.payload } };
      }
      return { ...state, messages: msgs };
    }
    case "SET_LOADING":
      return { ...state, isLoading: action.payload };
    case "SET_RESPONSE":
      return {
        ...state,
        lastStage: action.payload.stage,
        lastProfile: action.payload.user_profile,
        lastProductContext: action.payload.product_context,
        lastToolRounds: action.payload.tool_rounds,
      };
    case "TOGGLE_DEBUG":
      return { ...state, showDebug: !state.showDebug };
    case "SET_ERROR":
      return { ...state, error: action.payload };
    case "SET_STATUS":
      return { ...state, error: null };  // status messages are transient, just clear errors
    case "REMOVE_CONVERSATION":
      return {
        ...state,
        conversations: state.conversations.filter((c) => c.id !== action.payload),
        currentConvId: state.currentConvId === action.payload ? null : state.currentConvId,
      };
    default:
      return state;
  }
}

/* ---- Hook ---- */

export function useChat() {
  const [state, dispatch] = useReducer(reducer, initialState);

  /* Load conversation list on mount */
  useEffect(() => {
    api.listConversations().then((convs) => {
      dispatch({ type: "SET_CONVERSATIONS", payload: convs });
    }).catch(() => {});
  }, []);

  /* Select a conversation */
  const selectConversation = useCallback(async (convId: string) => {
    dispatch({ type: "SET_CURRENT_CONV", payload: convId });
    try {
      const msgs = await api.getMessages(convId);
      dispatch({
        type: "SET_MESSAGES",
        payload: msgs.map((m: MessageRecord) => ({
          role: m.role as "user" | "assistant",
          content: m.content,
          details: m.details ? JSON.parse(m.details) : null,
        })),
      });
    } catch {
      dispatch({ type: "SET_MESSAGES", payload: [] });
    }
  }, []);

  /* Create new conversation */
  const newConversation = useCallback(async () => {
    try {
      const conv = await api.createConversation();
      dispatch({ type: "SET_CONVERSATIONS", payload: [conv, ...state.conversations] });
      await selectConversation(conv.id);
    } catch (e: any) {
      dispatch({ type: "SET_ERROR", payload: e.message });
    }
  }, [state.conversations, selectConversation]);

  /* Delete conversation */
  const removeConversation = useCallback(async (convId: string) => {
    try {
      await api.deleteConversation(convId);
      dispatch({ type: "REMOVE_CONVERSATION", payload: convId });
    } catch (e: any) {
      dispatch({ type: "SET_ERROR", payload: e.message });
    }
  }, []);

  /* Send a message (streaming) */
  const sendMessage = useCallback(async (question: string) => {
    const convId = state.currentConvId;
    if (!convId || !question.trim()) return;

    dispatch({ type: "SET_LOADING", payload: true });
    dispatch({ type: "SET_ERROR", payload: null });
    dispatch({ type: "ADD_MESSAGE", payload: { role: "user", content: question } });
    api.saveMessage(convId, "user", question).catch(() => {});

    /* Add empty assistant placeholder for streaming */
    dispatch({ type: "ADD_MESSAGE", payload: { role: "assistant", content: "" } });

    const chatHistory: ChatMessage[] = state.messages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({ role: m.role, content: m.content }));

    let answerText = "";

    try {
      await api.sendMessageStream(convId, question, chatHistory, {
        onToken(content) {
          answerText += content;
          dispatch({ type: "APPEND_TOKEN", payload: content });
        },
        onStatus(message) {
          dispatch({ type: "SET_STATUS", payload: message });
        },
        onStage(stage) {
          dispatch({ type: "UPDATE_LAST_DETAILS", payload: { stage } });
        },
        onDone(meta) {
          api.saveMessage(convId, "assistant", answerText, {
            stage: meta.stage, tool_rounds: meta.tool_rounds,
          }).catch(() => {});
          dispatch({
            type: "UPDATE_LAST_DETAILS",
            payload: { stage: meta.stage, tool_rounds: meta.tool_rounds },
          });
          dispatch({
            type: "SET_RESPONSE",
            payload: {
              stage: meta.stage,
              user_profile: meta.user_profile || state.lastProfile,
              product_context: meta.product_context || state.lastProductContext,
              tool_rounds: meta.tool_rounds,
            },
          });
        },
        onError(message) {
          dispatch({ type: "SET_ERROR", payload: message });
        },
      });

      /* Refresh conversation list (title may have changed) */
      const convs = await api.listConversations();
      dispatch({ type: "SET_CONVERSATIONS", payload: convs });
    } catch (e: any) {
      dispatch({ type: "SET_ERROR", payload: e.message });
    } finally {
      dispatch({ type: "SET_LOADING", payload: false });
    }
  }, [state.currentConvId, state.messages, state.lastProfile, state.lastProductContext]);

  return {
    state,
    dispatch,
    selectConversation,
    newConversation,
    removeConversation,
    sendMessage,
  };
}
