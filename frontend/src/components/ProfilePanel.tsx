import { PROFILE_KEY_LABELS } from "../types";

interface ProfilePanelProps {
  profile: string;
}

interface ProfileEntry {
  key: string;
  value: string;
  confidence: number;
}

function parseProfile(raw: string): ProfileEntry[] {
  if (!raw || raw === "(暂无画像)") return [];
  const entries: ProfileEntry[] = [];
  for (const line of raw.trim().split("\n")) {
    const cleaned = line.replace(/^-\s*/, "");
    const colonIdx = cleaned.indexOf(": ");
    if (colonIdx === -1) continue;
    const key = cleaned.substring(0, colonIdx);
    const rest = cleaned.substring(colonIdx + 2);
    const confMatch = rest.match(/\(置信度\s*(\d+)%\)/);
    const value = confMatch ? rest.substring(0, rest.lastIndexOf("(")).trim() : rest;
    const confidence = confMatch ? parseInt(confMatch[1]) : 100;
    entries.push({ key, value, confidence });
  }
  return entries;
}

function confidenceColor(pct: number): string {
  if (pct >= 75) return "bg-emerald-400";
  if (pct >= 50) return "bg-amber-400";
  return "bg-rose-300";
}

export default function ProfilePanel({ profile }: ProfilePanelProps) {
  const entries = parseProfile(profile);

  if (entries.length === 0) {
    return (
      <div className="text-xs text-slate-400 leading-relaxed">
        <p className="text-slate-500 font-medium mb-1">暂无画像数据</p>
        <p>开始对话后自动提取偏好</p>
      </div>
    );
  }

  return (
    <div className="space-y-2.5">
      {entries.map((entry) => (
        <div key={entry.key}>
          <div className="flex justify-between items-baseline mb-1">
            <span className="text-xs text-slate-500">
              {PROFILE_KEY_LABELS[entry.key] || entry.key}
            </span>
            <span className="text-xs font-medium text-slate-700">{entry.value}</span>
          </div>
          <div className="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${confidenceColor(entry.confidence)}`}
              style={{ width: `${entry.confidence}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
