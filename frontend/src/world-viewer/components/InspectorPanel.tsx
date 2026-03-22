/** Agent inspector panel — right-side detailed view of selected agent */

import { useWorldStore } from '../store';
import { getEmotionColor } from '../layout';
import type { AgentState, RelationshipEntry, MemoryEntry, FutureBranch } from '../types';

export function InspectorPanel() {
  const selectedAgentId = useWorldStore(s => s.selectedAgentId);
  const inspectorOpen = useWorldStore(s => s.inspectorOpen);
  const selectAgent = useWorldStore(s => s.selectAgent);
  const currentTick = useWorldStore(s => s.currentTick);
  const tickData = useWorldStore(s => s.snapshot?.ticks[s.currentTick] ?? null);
  const snapshot = useWorldStore(s => s.snapshot);

  const agent = selectedAgentId && tickData ? tickData.agent_states[selectedAgentId] : null;
  const meta = selectedAgentId && snapshot ? snapshot.agents[selectedAgentId] : null;

  if (!inspectorOpen || !agent || !meta) return null;

  return (
    <div className="inspector-panel">
      {/* Close button */}
      <button
        className="inspector-close"
        onClick={() => selectAgent(null)}
      >
        ✕
      </button>

      {/* Identity */}
      <section className="inspector-section">
        <h2 className="inspector-name">{agent.name}</h2>
        <p className="inspector-background">{meta.background}</p>
        <p className="inspector-temperament">{meta.temperament}</p>
        {meta.coalitions.length > 0 && (
          <div className="inspector-tags">
            {meta.coalitions.map(c => (
              <span key={c} className="tag tag-coalition">{c}</span>
            ))}
          </div>
        )}
        {meta.identity_tags.length > 0 && (
          <div className="inspector-tags">
            {meta.identity_tags.map(t => (
              <span key={t} className="tag tag-identity">{t}</span>
            ))}
          </div>
        )}
      </section>

      {/* Current State */}
      <section className="inspector-section">
        <h3 className="inspector-section-title">Current State</h3>
        <div className="state-row">
          <span className="state-label">Action</span>
          <span className="state-value state-action">{agent.action.replace(/_/g, ' ')}</span>
        </div>
        <div className="state-row">
          <span className="state-label">Location</span>
          <span className="state-value">{agent.location}</span>
        </div>
        <div className="state-row">
          <span className="state-label">Showing</span>
          <EmotionBadge emotion={agent.surface} valence={agent.valence} />
        </div>
        {agent.divergence > 0.15 && (
          <div className="state-row">
            <span className="state-label">Actually</span>
            <EmotionBadge emotion={agent.internal} valence={agent.valence} />
            <span className="divergence-indicator">
              mask {Math.round(agent.divergence * 100)}%
            </span>
          </div>
        )}
        <div className="heart-bars">
          <HeartBar label="Energy" value={agent.energy} color="#22c55e" />
          <HeartBar label="Valence" value={agent.valence} color="#60a5fa" />
          <HeartBar label="Arousal" value={agent.arousal} color="#f59e0b" />
          <HeartBar label="Tension" value={agent.tension} color="#ef4444" />
          <HeartBar label="Impulse" value={agent.impulse_control} color="#a78bfa" />
          <HeartBar label="Vuln." value={agent.vulnerability} color="#ec4899" />
        </div>
      </section>

      {/* Private Mind */}
      <section className="inspector-section">
        <h3 className="inspector-section-title">Private Mind</h3>
        <div className="private-mind">
          <div className="mind-concern">
            <span className="mind-label">Concern</span>
            <span className="mind-value">{agent.primary_concern}</span>
          </div>
          <div className="mind-interpretation">
            "{agent.interpretation}"
          </div>
          <div className="mind-row">
            <span className="mind-label">Blame</span>
            <span className="mind-value">{agent.blame_target}</span>
          </div>
          <div className="mind-row">
            <span className="mind-label">Seeks</span>
            <span className="mind-value">{agent.support_target}</span>
          </div>
          <div className="mind-row">
            <span className="mind-label">Priority</span>
            <span className="mind-value">{agent.priority_motive}</span>
          </div>
          <div className="mind-row">
            <span className="mind-label">Mask</span>
            <span className="mind-value">{agent.mask_style}</span>
          </div>
          <div className="mind-row">
            <span className="mind-label">Style</span>
            <span className="mind-value">{agent.action_style}</span>
          </div>
          <div className="mind-voice">
            <span className="mind-label">Inner voice</span>
            <p>"{agent.inner_voice}"</p>
          </div>
          {/* Pressure meters */}
          <div className="pressure-grid">
            <PressureMeter label="Economic" value={agent.economic_pressure} />
            <PressureMeter label="Loyalty" value={agent.loyalty_pressure} />
            <PressureMeter label="Secrecy" value={agent.secrecy_pressure} />
            <PressureMeter label="Opportunity" value={agent.opportunity_pressure} />
          </div>
        </div>
      </section>

      {/* Recent Memories */}
      <section className="inspector-section">
        <h3 className="inspector-section-title">Recent Memories</h3>
        <div className="memories-list">
          {agent.recent_memories.length === 0 && (
            <div className="empty-state">No memories yet</div>
          )}
          {agent.recent_memories.slice().reverse().slice(0, 8).map((mem, i) => (
            <MemoryItem key={i} memory={mem} />
          ))}
        </div>
      </section>

      {/* Relationships */}
      <section className="inspector-section">
        <h3 className="inspector-section-title">Relationships</h3>
        <div className="relationships-list">
          {(!agent.relationships || agent.relationships.length === 0) && (
            <div className="empty-state">No relationships yet</div>
          )}
          {agent.relationships?.slice(0, 6).map((rel, i) => (
            <RelationshipItem key={i} rel={rel} />
          ))}
        </div>
      </section>

      {/* Future Paths */}
      <section className="inspector-section">
        <h3 className="inspector-section-title">Future Paths</h3>
        <div className="futures-list">
          {agent.future_branches?.map((branch, i) => (
            <FutureItem key={i} branch={branch} index={i} />
          ))}
        </div>
      </section>
    </div>
  );
}

function EmotionBadge({ emotion, valence }: { emotion: string; valence: number }) {
  const color = getEmotionColor(valence, 0.5, 0.3);
  return (
    <span className="emotion-badge" style={{ borderColor: color, color }}>
      {emotion}
    </span>
  );
}

function HeartBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="heart-bar">
      <span className="heart-bar-label">{label}</span>
      <div className="heart-bar-track">
        <div
          className="heart-bar-fill"
          style={{ width: `${value * 100}%`, backgroundColor: color }}
        />
      </div>
      <span className="heart-bar-value">{(value * 100).toFixed(0)}</span>
    </div>
  );
}

function PressureMeter({ label, value }: { label: string; value: number }) {
  const intensity = value > 0.6 ? 'high' : value > 0.3 ? 'mid' : 'low';
  return (
    <div className={`pressure-meter pressure-${intensity}`}>
      <span className="pressure-label">{label}</span>
      <span className="pressure-value">{(value * 100).toFixed(0)}</span>
    </div>
  );
}

function MemoryItem({ memory }: { memory: MemoryEntry }) {
  const day = Math.floor(memory.tick / 24) + 1;
  const hour = memory.tick % 24;
  return (
    <div className="memory-item">
      <span className="memory-time">D{day} {hour}:00</span>
      <span className="memory-desc">{memory.description}</span>
      {memory.other && (
        <span className="memory-other">w/ {memory.other}</span>
      )}
    </div>
  );
}

function RelationshipItem({ rel }: { rel: RelationshipEntry }) {
  const sentiment =
    rel.trust > 0.5 && rel.warmth > 0.4 ? 'trust' :
    rel.rivalry > 0.3 || rel.conflict_events > 2 ? 'rival' :
    rel.resentment_toward > 0.3 ? 'resent' :
    rel.warmth > 0.3 ? 'warm' : 'neutral';

  return (
    <div className={`rel-item rel-${sentiment}`}>
      <div className="rel-header">
        <span className="rel-name">{rel.other_name}</span>
        <span className={`rel-badge rel-badge-${sentiment}`}>
          {sentiment === 'trust' ? 'trusts' :
           sentiment === 'rival' ? 'rivalry' :
           sentiment === 'resent' ? 'resents' :
           sentiment === 'warm' ? 'warm' : 'neutral'}
        </span>
      </div>
      <div className="rel-stats">
        {rel.trust !== 0 && <span>trust {rel.trust > 0 ? '+' : ''}{(rel.trust * 100).toFixed(0)}</span>}
        {rel.warmth !== 0 && <span>warmth {rel.warmth > 0 ? '+' : ''}{(rel.warmth * 100).toFixed(0)}</span>}
        {rel.resentment_toward > 0 && <span className="stat-negative">resent {(rel.resentment_toward * 100).toFixed(0)}</span>}
        {rel.grievance_toward > 0 && <span className="stat-negative">grievance {(rel.grievance_toward * 100).toFixed(0)}</span>}
        {rel.debt_toward > 0 && <span className="stat-warn">owes {(rel.debt_toward * 100).toFixed(0)}</span>}
        {rel.debt_from > 0 && <span className="stat-positive">owed {(rel.debt_from * 100).toFixed(0)}</span>}
        {rel.alliance_strength > 0.1 && <span className="stat-positive">alliance {(rel.alliance_strength * 100).toFixed(0)}</span>}
        {rel.rivalry > 0.1 && <span className="stat-negative">rivalry {(rel.rivalry * 100).toFixed(0)}</span>}
      </div>
      {(rel.support_events > 0 || rel.conflict_events > 0) && (
        <div className="rel-events">
          {rel.support_events > 0 && <span className="stat-positive">{rel.support_events} support</span>}
          {rel.conflict_events > 0 && <span className="stat-negative">{rel.conflict_events} conflict</span>}
          {rel.betrayal_events > 0 && <span className="stat-danger">{rel.betrayal_events} betrayal</span>}
        </div>
      )}
    </div>
  );
}

function FutureItem({ branch, index }: { branch: FutureBranch; index: number }) {
  const icons = ['→', '⚡', '🛟'];
  const colors = ['#60a5fa', '#f59e0b', '#22c55e'];
  return (
    <div className="future-item" style={{ borderLeftColor: colors[index] }}>
      <div className="future-label" style={{ color: colors[index] }}>
        {icons[index]} {branch.label}
      </div>
      <p className="future-summary">{branch.summary}</p>
    </div>
  );
}
