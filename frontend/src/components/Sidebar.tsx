import { Plus, Trash2, MessageSquare, Layers } from "lucide-react";
import type { Conversation } from "../types";
import ProfilePanel from "./ProfilePanel";
import { STAGE_LABELS } from "../types";

interface SidebarProps {
  conversations: Conversation[];
  currentConvId: string | null;
  stage: string;
  profile: string;
  onNewChat: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

export default function Sidebar({
  conversations,
  currentConvId,
  stage,
  profile,
  onNewChat,
  onSelect,
  onDelete,
}: SidebarProps) {
  return (
    <aside className="w-72 bg-slate-50/80 border-r border-slate-200 flex flex-col h-full shrink-0">
      {/* New Chat */}
      <div className="p-3.5">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-indigo-500 hover:bg-indigo-600 text-white text-sm font-medium transition-all shadow-sm hover:shadow-md"
        >
          <Plus size={16} />
          新对话
        </button>
      </div>

      {/* Stage */}
      <div className="px-4 py-3 border-t border-slate-200/60">
        <p className="text-[11px] font-medium text-slate-400 uppercase tracking-wide mb-1.5">当前阶段</p>
        <div className="flex items-center gap-2">
          <Layers size={14} className="text-indigo-400" />
          <p className="text-sm text-slate-700 font-medium">
            {STAGE_LABELS[stage] || "👋 发现需求"}
          </p>
        </div>
      </div>

      {/* Profile */}
      <div className="px-4 py-3 border-t border-slate-200/60 max-h-48 overflow-y-auto">
        <p className="text-[11px] font-medium text-slate-400 uppercase tracking-wide mb-2">用户画像</p>
        <ProfilePanel profile={profile} />
      </div>

      {/* Conversations */}
      <div className="border-t border-slate-200 flex-1 overflow-y-auto flex flex-col">
        <div className="px-4 py-2.5">
          <p className="text-[11px] font-medium text-slate-400 uppercase tracking-wide">历史对话</p>
        </div>
        {conversations.length === 0 && (
          <p className="text-xs text-slate-400 px-4">暂无历史对话</p>
        )}
        <div className="flex-1 overflow-y-auto">
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className={`group flex items-center gap-2 px-4 py-2.5 mx-2 rounded-lg cursor-pointer transition-all ${
                conv.id === currentConvId
                  ? "bg-white shadow-sm border border-slate-200/60"
                  : "hover:bg-white/60"
              }`}
              onClick={() => onSelect(conv.id)}
            >
              <MessageSquare size={14} className={`shrink-0 ${conv.id === currentConvId ? "text-indigo-400" : "text-slate-400"}`} />
              <span className={`text-sm truncate flex-1 ${conv.id === currentConvId ? "text-slate-800 font-medium" : "text-slate-600"}`}>
                {conv.title || "新对话"}
              </span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(conv.id);
                }}
                className="opacity-0 group-hover:opacity-100 text-slate-300 hover:text-red-500 transition-all p-0.5"
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
