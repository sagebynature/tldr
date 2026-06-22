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

function messageFromEntry(value) {
  if (!value || typeof value !== "object") {
    return null;
  }
  if ("message" in value) {
    return value.message;
  }
  return value;
}

export function latestAssistantText(messages) {
  if (!Array.isArray(messages)) {
    return "";
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messageFromEntry(messages[index]);
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
  pi.on("agent_end", async (event, ctx) => {
    const text = latestAssistantText(event.messages);
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
