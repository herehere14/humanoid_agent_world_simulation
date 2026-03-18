import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Copy, ThumbsUp, ThumbsDown, ChevronDown } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { clsx } from 'clsx';
import type { Message } from '@/types';
import { Chip } from '@/components/ui';
import { formatBranchName } from '@/lib/mockData';

export function MessageBubble({ message }: { message: Message }) {
  const [copied, setCopied] = useState(false);
  const [showBase, setShowBase] = useState(false);
  const isUser = message.role === 'user';
  const isStreaming = message.status === 'streaming';

  const copy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease: 'easeOut' }}
      className={clsx('flex gap-3 group', isUser ? 'flex-row-reverse' : 'flex-row')}
    >
      {/* Avatar */}
      <div className={clsx(
        'w-8 h-8 rounded-full flex items-center justify-center shrink-0 text-xs font-semibold',
        isUser
          ? 'bg-text-base text-white'
          : 'bg-primary-light border border-primary-mid text-primary'
      )}>
        {isUser ? 'Y' : 'PF'}
      </div>

      {/* Bubble + meta */}
      <div className={clsx('flex flex-col gap-1.5 max-w-[78%]', isUser ? 'items-end' : 'items-start')}>
        {/* Bubble */}
        <div className={clsx(
          'px-4 py-3 text-sm leading-relaxed',
          isUser
            ? 'bg-text-base text-white rounded-3xl rounded-tr-md'
            : 'bg-surface border border-border rounded-3xl rounded-tl-md shadow-card'
        )}>
          {isUser ? (
            <p>{message.content}</p>
          ) : (
            <div className={clsx('prose-clean', isStreaming && 'typing-cursor')}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* Assistant meta row */}
        {!isUser && message.status === 'complete' && message.trace?.evaluation && (
          <div className="flex flex-wrap items-center gap-1.5">
            {message.trace.evaluation.selected_branch && (
              <Chip color="blue">
                {formatBranchName(message.trace.evaluation.selected_branch)}
              </Chip>
            )}
            {message.trace.evaluation.reward_score !== undefined && (
              <Chip
                color={message.trace.evaluation.reward_score >= 0.75 ? 'green' : message.trace.evaluation.reward_score >= 0.5 ? 'amber' : 'red'}
              >
                {(message.trace.evaluation.reward_score * 100).toFixed(0)}% reward
              </Chip>
            )}
            {message.improvementDelta !== undefined && message.improvementDelta > 0 && (
              <Chip color="green">
                +{(message.improvementDelta * 100).toFixed(1)}% vs base
              </Chip>
            )}
            {message.trace.timings?.total_ms && (
              <span className="text-[11px] text-text-muted">{message.trace.timings.total_ms}ms</span>
            )}
            {message.baseAnswer && (
              <button
                onClick={() => setShowBase(!showBase)}
                className="flex items-center gap-1 text-[11px] text-text-muted hover:text-text-secondary"
              >
                Base answer
                <ChevronDown size={10} className={clsx('transition-transform', showBase && 'rotate-180')} />
              </button>
            )}
          </div>
        )}

        {/* Base answer */}
        <AnimatePresence>
          {showBase && message.baseAnswer && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="w-full overflow-hidden"
            >
              <div className="bg-elevated border border-border rounded-2xl px-4 py-3 mt-1">
                <div className="flex items-center gap-1.5 mb-2">
                  <div className="w-2 h-2 rounded-full bg-text-muted" />
                  <span className="text-[11px] text-text-muted font-medium">Base model — no adaptive routing</span>
                </div>
                <div className="prose-clean text-sm text-text-secondary">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.baseAnswer}</ReactMarkdown>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Action buttons */}
        {!isUser && message.status === 'complete' && (
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <button onClick={copy} className="p-1 rounded-lg text-text-muted hover:text-text-secondary hover:bg-elevated transition-all">
              <Copy size={12} />
            </button>
            <button className="p-1 rounded-lg text-text-muted hover:text-success hover:bg-elevated transition-all">
              <ThumbsUp size={12} />
            </button>
            <button className="p-1 rounded-lg text-text-muted hover:text-danger hover:bg-elevated transition-all">
              <ThumbsDown size={12} />
            </button>
            {copied && <span className="text-[10px] text-success ml-1">Copied</span>}
          </div>
        )}
      </div>
    </motion.div>
  );
}
