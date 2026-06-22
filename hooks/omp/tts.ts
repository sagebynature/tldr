import { spawn } from "node:child_process";

function textFromContent(content) {
  if (typeof content === "string") {
    return content.trim();
  }
  if (!Array.isArray(content)) {
    return "";
  }
  return content
    .filter((part) => part && part.type === "text" && typeof part.text === "string")
    .map((part) => part.text)
    .join("")
    .trim();
}

export function latestAssistantText(entries) {
  if (!Array.isArray(entries)) {
    return "";
  }
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const entry = entries[index];
    const message = entry && typeof entry === "object" ? entry.message : null;
    if (!message || message.role !== "assistant") {
      continue;
    }
    const text = textFromContent(message.content);
    if (text) {
      return text;
    }
  }
  return "";
}

export default function ttsSummarizerHook(pi) {
  pi.on("turn_end", async (_event, ctx) => {
    const text = latestAssistantText(ctx?.sessionManager?.getEntries?.());
    if (!text) {
      return;
    }

    const ttsBin = process.env.OMP_TTS_BIN || "tts-summarizer";
    const sessionId = process.env.OMP_TTS_SESSION_ID || `omp:${ctx?.cwd || process.cwd()}`;
    const args = ["speak", "--session_id", sessionId, text];

    try {
      const child = spawn(ttsBin, args, {
        detached: true,
        stdio: "ignore",
      });
      child.on("error", () => {});
      child.unref();
    } catch {
      // ponytail: missing local tts-summarizer must not break an OMP turn.
    }
  });
}
