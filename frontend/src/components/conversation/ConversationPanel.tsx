import { useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Layers } from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';
import { MessageBubble } from './MessageBubble';
import { MessageInput } from './MessageInput';
import type { TaskType } from '@/types';

interface Props {
  onSend: (message: string, taskType: TaskType) => void;
  onAbort: () => void;
}

const EXAMPLES = [
  { label: 'Math',     text: 'What is the derivative of x³ · sin(x)?' },
  { label: 'Code',     text: 'Implement a binary search tree in Python' },
  { label: 'Planning', text: 'Plan a machine learning pipeline for fraud detection' },
  { label: 'Factual',  text: 'Explain how transformer attention works' },
];

export function ConversationPanel({ onSend, onAbort }: Props) {
  const { messages, isStreaming } = useAppStore();
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 py-8 space-y-6">
          {messages.length === 0 ? (
            <Welcome onExample={(text) => onSend(text, 'auto')} />
          ) : (
            <AnimatePresence initial={false}>
              {messages.map(m => <MessageBubble key={m.id} message={m} />)}
            </AnimatePresence>
          )}

          {/* Typing indicator */}
          {isStreaming && messages.at(-1)?.role === 'user' && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex gap-3"
            >
              <div className="w-8 h-8 rounded-full bg-primary-light border border-primary-mid flex items-center justify-center shrink-0 text-xs font-semibold text-primary">
                PF
              </div>
              <div className="bg-surface border border-border rounded-3xl rounded-tl-md shadow-card px-4 py-3 flex items-center gap-1.5">
                {[0,1,2].map(i => (
                  <motion.span
                    key={i}
                    className="w-1.5 h-1.5 rounded-full bg-text-muted"
                    animate={{ y: [-1, 1, -1], opacity: [0.4, 1, 0.4] }}
                    transition={{ duration: 0.7, repeat: Infinity, delay: i * 0.12 }}
                  />
                ))}
              </div>
            </motion.div>
          )}

          <div ref={endRef} />
        </div>
      </div>

      {/* Input */}
      <div className="max-w-3xl mx-auto w-full px-0 pb-0">
        <MessageInput onSend={onSend} onAbort={onAbort} />
      </div>
    </div>
  );
}

function Welcome({ onExample }: { onExample: (text: string) => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center justify-center pt-12 pb-8"
    >
      {/* Icon */}
      <div className="w-14 h-14 rounded-2xl bg-primary-light border border-primary-mid flex items-center justify-center mb-5">
        <Layers size={26} className="text-primary" />
      </div>

      <h1 className="text-2xl font-semibold text-text-base mb-2 tracking-tight">
        What can I help with?
      </h1>
      <p className="text-sm text-text-secondary text-center max-w-md mb-8 leading-relaxed">
        Unlike a standard chatbot, this system routes your query through specialised branches, evaluates outputs with reward signals, and continuously adapts.
      </p>

      {/* Example buttons */}
      <div className="grid grid-cols-2 gap-3 w-full max-w-lg">
        {EXAMPLES.map(ex => (
          <button
            key={ex.label}
            onClick={() => onExample(ex.text)}
            className="text-left p-4 bg-surface border border-border rounded-2xl hover:border-border-strong hover:shadow-card-md transition-all duration-150 group"
          >
            <div className="text-[11px] font-semibold text-text-muted uppercase tracking-wider mb-1.5">
              {ex.label}
            </div>
            <p className="text-sm text-text-secondary leading-snug line-clamp-2">{ex.text}</p>
          </button>
        ))}
      </div>
    </motion.div>
  );
}
