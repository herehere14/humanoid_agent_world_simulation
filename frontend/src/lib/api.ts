import type { EngineState } from '@/types';
import { useAppStore } from '@/store/useAppStore';

const BASE_URL = '/api';

export interface ChatRequest {
  message: string;
  task_type?: string;
  user_id?: string;
  include_base_comparison?: boolean;
}

// ─── SSE Chat Stream ──────────────────────────────────────────────────────────

export async function streamChat(
  req: ChatRequest,
  signal?: AbortSignal
): Promise<void> {
  const store = useAppStore.getState();

  const response = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      if (!line.trim()) continue;

      if (line.startsWith('data: ')) {
        const raw = line.slice(6);
        if (raw === '[DONE]') continue;

        try {
          const event = JSON.parse(raw) as { type: string; data: unknown; timestamp?: number };
          handleSSEEvent(event.type, event.data, store);
        } catch {
          // skip malformed
        }
      }
    }
  }
}

function handleSSEEvent(
  type: string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: any,
  store: ReturnType<typeof useAppStore.getState>
) {
  switch (type) {
    case 'routing':
      store.setTraceRouting(data);
      break;
    case 'branch_start':
      store.addTraceBranchStart(data.branch_name);
      break;
    case 'branch_complete':
      store.updateTraceBranchOutput(data.branch_name, data.output, data.reward);
      break;
    case 'evaluation':
      store.setTraceEvaluation(data);
      break;
    case 'optimization':
      store.setTraceOptimization(data);
      break;
    case 'memory_update':
      store.setTraceMemory(data);
      break;
    case 'token': {
      // Find the last assistant message and append token
      const msgs = store.messages;
      const lastAssistant = [...msgs].reverse().find((m) => m.role === 'assistant');
      if (lastAssistant) {
        store.appendToken(lastAssistant.id, data.token);
      }
      break;
    }
    case 'complete': {
      store.setTraceSelectedPath(data.selected_path ?? []);
      if (data.timings) store.setTraceTimings(data.timings);
      if (data.composer) store.setTraceComposer(data.composer);
      store.setTraceStage('done');

      // Update last assistant message
      const msgs = store.messages;
      const lastAssistant = [...msgs].reverse().find((m) => m.role === 'assistant');
      if (lastAssistant) {
        store.updateMessage(lastAssistant.id, {
          content: data.answer ?? lastAssistant.content,
          status: 'complete',
          baseAnswer: data.base_answer,
          improvementDelta: data.improvement_delta,
          winLabel: data.win_label,
          trace: {
            ...store.activeTrace,
            selectedPath: data.selected_path ?? [],
            timings: data.timings,
            composer: data.composer,
            stage: 'done',
          },
        });
      }

      // Add performance point
      if (data.improvement_delta !== undefined) {
        const adaptiveScore = data.evaluation?.reward_score ?? 0.7;
        const baseScore = adaptiveScore - (data.improvement_delta ?? 0);
        store.addPerformancePoint({
          taskIndex: store.taskCount,
          adaptiveScore,
          baseScore,
          delta: data.improvement_delta,
          label: `Task ${store.taskCount}`,
        });
      }

      store.setIsStreaming(false);
      break;
    }
    case 'error':
      store.setIsStreaming(false);
      store.setTraceStage('idle');
      break;
  }
}

// ─── REST Endpoints ───────────────────────────────────────────────────────────

export async function fetchEngineState(): Promise<EngineState> {
  const res = await fetch(`${BASE_URL}/state`);
  if (!res.ok) throw new Error('Failed to fetch engine state');
  return res.json();
}

export async function submitFeedback(
  taskId: string,
  score: number,
  comment?: string
): Promise<void> {
  await fetch(`${BASE_URL}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_id: taskId, score, comment }),
  });
}

export async function fetchRecentEvents(limit = 20): Promise<unknown[]> {
  const res = await fetch(`${BASE_URL}/events?limit=${limit}`);
  if (!res.ok) return [];
  return res.json();
}
