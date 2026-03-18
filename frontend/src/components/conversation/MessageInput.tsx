import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowUp, Square } from 'lucide-react';
import { clsx } from 'clsx';
import { useAppStore } from '@/store/useAppStore';
import type { TaskType } from '@/types';

const TYPES: { value: TaskType; label: string }[] = [
  { value: 'auto',     label: 'Auto'     },
  { value: 'math',     label: 'Math'     },
  { value: 'code',     label: 'Code'     },
  { value: 'planning', label: 'Planning' },
  { value: 'factual',  label: 'Factual'  },
  { value: 'creative', label: 'Creative' },
  { value: 'general',  label: 'General'  },
];

interface Props {
  onSend: (message: string, taskType: TaskType) => void;
  onAbort: () => void;
}

export function MessageInput({ onSend, onAbort }: Props) {
  const [value, setValue]       = useState('');
  const [taskType, setTaskType] = useState<TaskType>('auto');
  const ref   = useRef<HTMLTextAreaElement>(null);
  const { isStreaming } = useAppStore();

  // Auto-resize
  useEffect(() => {
    const ta = ref.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
  }, [value]);

  const send = () => {
    if (!value.trim() || isStreaming) return;
    onSend(value.trim(), taskType);
    setValue('');
  };

  return (
    <div className="px-4 pb-5">
      {/* Outer card */}
      <div className={clsx(
        'bg-surface border rounded-3xl shadow-card-md transition-all duration-200',
        isStreaming ? 'border-border' : 'border-border hover:border-border-strong focus-within:border-primary focus-within:shadow-input'
      )}>
        {/* Textarea */}
        <textarea
          ref={ref}
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
          disabled={isStreaming}
          placeholder="Message Prompt Forest…"
          rows={1}
          className="w-full bg-transparent resize-none outline-none px-5 pt-4 pb-2 text-sm text-text-base placeholder-text-placeholder min-h-[52px] max-h-40 leading-relaxed"
        />

        {/* Bottom bar */}
        <div className="flex items-center justify-between px-4 pb-3">
          {/* Task type pills */}
          <div className="flex items-center gap-1 flex-wrap">
            {TYPES.map(t => (
              <button
                key={t.value}
                onClick={() => setTaskType(t.value)}
                className={clsx(
                  'px-2.5 py-0.5 rounded-full text-[11px] font-medium transition-all duration-100',
                  taskType === t.value
                    ? 'bg-text-base text-white'
                    : 'text-text-muted hover:text-text-secondary hover:bg-elevated'
                )}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Send / Stop */}
          <AnimatePresence mode="wait">
            {isStreaming ? (
              <motion.button
                key="stop"
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                onClick={onAbort}
                className="w-8 h-8 rounded-full border border-border flex items-center justify-center text-text-secondary hover:bg-elevated transition-all"
              >
                <Square size={13} fill="currentColor" />
              </motion.button>
            ) : (
              <motion.button
                key="send"
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                whileTap={{ scale: 0.93 }}
                onClick={send}
                disabled={!value.trim()}
                className={clsx(
                  'w-8 h-8 rounded-full flex items-center justify-center transition-all duration-150',
                  value.trim()
                    ? 'bg-text-base text-white hover:bg-gray-800 shadow-card'
                    : 'bg-elevated text-text-muted cursor-not-allowed'
                )}
              >
                <ArrowUp size={15} />
              </motion.button>
            )}
          </AnimatePresence>
        </div>
      </div>

      <p className="text-center text-[11px] text-text-placeholder mt-2.5">
        Adaptive routing · Multi-branch execution · Live optimization
      </p>
    </div>
  );
}
