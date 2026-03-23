/**
 * Parallel Universe system.
 *
 * Each "universe" is a unique simulation run identified by a seed.
 * The seed shifts:
 *   - Personality parameters (arousal_rise_rate, vulnerability_weight, etc.)
 *   - Event timing (events fire earlier/later by +/- hours)
 *   - Relationship initialisation (trust/warmth jitter)
 *   - Which agents get laid off (different subsets)
 *   - Environmental conditions (weather, season tint)
 *
 * This creates infinite parallel timelines where butterfly effects
 * cascade differently. Two universes that start identical will diverge
 * because a single shifted parameter causes one agent to confront instead
 * of withdraw, which changes a relationship, which changes a coalition
 * vote, which changes the whole town's arc.
 *
 * Usage:
 *   const seed = generateUniverseSeed();
 *   const params = getUniverseParams(seed);
 *   // pass params to the simulation backend or mock generator
 */

export interface UniverseParams {
  seed: number;
  label: string;

  // Personality jitter — applied as multipliers to agent personality fields
  arousalJitter: number;        // ±0.15
  valenceJitter: number;        // ±0.1
  vulnerabilityJitter: number;  // ±0.2
  impulseJitter: number;        // ±0.1

  // Event timing shift — hours earlier or later
  eventTimingShift: number;     // -6 to +6

  // Relationship noise
  trustNoise: number;           // ±0.15
  warmthNoise: number;          // ±0.1

  // Layoff selection modifier — probability shift per agent
  layoffShuffle: boolean;       // true = randomise who gets laid off

  // Environmental
  weatherMood: 'clear' | 'overcast' | 'stormy' | 'golden';
  seasonTint: [number, number, number]; // RGB multiplier for the world

  // Chaos factor — scales how much each jitter actually applies
  chaosFactor: number;          // 0.0 = identical to base, 1.0 = max divergence
}

// Seeded PRNG
function mulberry32(seed: number) {
  return () => {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generateUniverseSeed(): number {
  return Math.floor(Math.random() * 2147483647);
}

export function getUniverseParams(seed: number): UniverseParams {
  const r = mulberry32(seed);

  const chaosFactor = 0.3 + r() * 0.7; // 0.3–1.0

  const weathers: UniverseParams['weatherMood'][] = ['clear', 'overcast', 'stormy', 'golden'];
  const weatherMood = weathers[Math.floor(r() * weathers.length)];

  const seasonTints: Record<string, [number, number, number]> = {
    clear:    [1.0, 1.0, 1.0],
    overcast: [0.85, 0.88, 0.95],
    stormy:   [0.7, 0.72, 0.8],
    golden:   [1.1, 1.0, 0.85],
  };

  // Generate a unique label
  const adjectives = [
    'Quiet', 'Turbulent', 'Fractured', 'Hopeful', 'Bitter', 'Scorched',
    'Frozen', 'Burning', 'Silent', 'Restless', 'Broken', 'Mended',
    'Dark', 'Bright', 'Shifting', 'Collapsing', 'Rising', 'Shattered',
    'Gentle', 'Volatile', 'Steady', 'Chaotic', 'Fragile', 'Hardened',
  ];
  const nouns = [
    'Timeline', 'Branch', 'Path', 'Thread', 'Fork', 'Divergence',
    'Reality', 'Strand', 'Echo', 'Fracture', 'Ripple', 'Cascade',
    'Current', 'Drift', 'Shift', 'Crossing', 'Rift', 'Horizon',
  ];
  const adj = adjectives[Math.floor(r() * adjectives.length)];
  const noun = nouns[Math.floor(r() * nouns.length)];
  const num = Math.floor(r() * 999) + 1;

  return {
    seed,
    label: `${adj} ${noun} #${num}`,
    arousalJitter:       (r() - 0.5) * 0.3 * chaosFactor,
    valenceJitter:       (r() - 0.5) * 0.2 * chaosFactor,
    vulnerabilityJitter: (r() - 0.5) * 0.4 * chaosFactor,
    impulseJitter:       (r() - 0.5) * 0.2 * chaosFactor,
    eventTimingShift:    Math.round((r() - 0.5) * 12 * chaosFactor),
    trustNoise:          (r() - 0.5) * 0.3 * chaosFactor,
    warmthNoise:         (r() - 0.5) * 0.2 * chaosFactor,
    layoffShuffle:       r() > (1 - chaosFactor * 0.5),
    weatherMood,
    seasonTint: seasonTints[weatherMood],
    chaosFactor,
  };
}

/** Apply universe parameters to a snapshot — mutates agent states in-place
 *  to create a divergent timeline. Called once when loading. */
export function applyUniverseToSnapshot(
  snapshot: any,
  params: UniverseParams,
): void {
  const r = mulberry32(params.seed + 12345);

  // Jitter initial agent states in tick 0
  if (snapshot.ticks.length > 0) {
    const t0 = snapshot.ticks[0];
    for (const state of Object.values(t0.agent_states) as any[]) {
      state.arousal  = clamp(state.arousal + params.arousalJitter * (0.5 + r() * 0.5), 0, 1);
      state.valence  = clamp(state.valence + params.valenceJitter * (0.5 + r() * 0.5), 0, 1);
      state.vulnerability = clamp(state.vulnerability + params.vulnerabilityJitter * r(), 0, 1);
      state.impulse_control = clamp(state.impulse_control + params.impulseJitter * r(), 0, 1);
      state.tension  = clamp(state.tension + (r() - 0.5) * 0.1 * params.chaosFactor, 0, 1);
      state.energy   = clamp(state.energy + (r() - 0.5) * 0.1 * params.chaosFactor, 0, 1);

      // Jitter relationships
      if (state.relationships) {
        for (const rel of state.relationships) {
          rel.trust  = clamp(rel.trust + params.trustNoise * (r() - 0.3), -1, 1);
          rel.warmth = clamp(rel.warmth + params.warmthNoise * (r() - 0.3), -1, 1);
        }
      }
    }
  }

  // ─── SCHEDULE DISRUPTION ─────────────────────────────────
  // This is what makes universes fundamentally different.
  // An agent who stays home instead of going to work misses the layoff
  // meeting, never gets fired, but also never bonds with coworkers at the bar.
  // That one change cascades through every relationship in the town.

  const allLocations = Object.keys(snapshot.locations);
  const allAgentIds = Object.keys(snapshot.agents);

  // Decide which agents have disrupted schedules in this universe
  const disruptedAgents = new Set<string>();
  const disruptionR = mulberry32(params.seed + 77777);
  for (const aid of allAgentIds) {
    // Higher chaos = more agents with disrupted days
    if (disruptionR() < 0.12 * params.chaosFactor) {
      disruptedAgents.add(aid);
    }
  }

  // For disrupted agents, pick which ticks they go somewhere else
  const locationOverrides = new Map<string, Map<number, string>>();
  for (const aid of disruptedAgents) {
    const overrides = new Map<number, string>();
    const oR = mulberry32(params.seed + hashStr(aid) + 999);

    // Pick 1-3 disruption days
    const disruptionDays = 1 + Math.floor(oR() * 3 * params.chaosFactor);
    for (let d = 0; d < disruptionDays; d++) {
      const startDay = 1 + Math.floor(oR() * 14); // within the 15-day sim
      const startTick = (startDay - 1) * 24;

      // What kind of disruption?
      const disruptionType = oR();
      if (disruptionType < 0.3) {
        // Stays home all day instead of going to work
        for (let h = 8; h < 18; h++) {
          overrides.set(startTick + h, 'home');
        }
      } else if (disruptionType < 0.5) {
        // Goes to bar instead of work
        for (let h = 10; h < 20; h++) {
          const barLoc = allLocations.find(l => /bar|pub|tap/i.test(l)) ?? 'bar';
          overrides.set(startTick + h, barLoc);
        }
      } else if (disruptionType < 0.7) {
        // Goes to park instead of work
        for (let h = 9; h < 16; h++) {
          const parkLoc = allLocations.find(l => /park|garden/i.test(l)) ?? 'park';
          overrides.set(startTick + h, parkLoc);
        }
      } else if (disruptionType < 0.85) {
        // Goes to church/community for support
        for (let h = 10; h < 14; h++) {
          const comLoc = allLocations.find(l => /church|community|chapel/i.test(l)) ?? 'church';
          overrides.set(startTick + h, comLoc);
        }
      } else {
        // Random location for the whole day
        const randLoc = allLocations[Math.floor(oR() * allLocations.length)];
        for (let h = 8; h < 20; h++) {
          overrides.set(startTick + h, randLoc);
        }
      }
    }
    locationOverrides.set(aid, overrides);
  }

  // ─── APPLY PER-TICK DIVERGENCE ─────────────────────────
  for (let i = 1; i < snapshot.ticks.length; i++) {
    const tick = snapshot.ticks[i];
    const amplification = 1 + i * 0.003 * params.chaosFactor;

    for (const state of Object.values(tick.agent_states) as any[]) {
      const agentR = mulberry32(params.seed + hashStr(state.id) * 1000 + i);

      // Apply schedule overrides — move agents to different locations
      const overrides = locationOverrides.get(state.id);
      if (overrides) {
        const override = overrides.get(i);
        if (override) {
          state.location = override;
          // Being somewhere unexpected changes the action
          if (state.action === 'WORK' && override !== state.location) {
            const altActions = ['IDLE', 'SOCIALIZE', 'RUMINATE', 'VENT'];
            state.action = altActions[Math.floor(agentR() * altActions.length)];
          }
        }
      }

      // Micro-jitter that compounds (butterfly effect)
      const jit = params.chaosFactor * 0.012 * amplification;
      state.arousal  = clamp(state.arousal + (agentR() - 0.5) * jit * 2, 0, 1);
      state.valence  = clamp(state.valence + (agentR() - 0.5) * jit * 1.5, 0, 1);
      state.tension  = clamp(state.tension + (agentR() - 0.5) * jit, 0, 1);
      state.energy   = clamp(state.energy + (agentR() - 0.5) * jit, 0, 1);
      state.vulnerability = clamp(state.vulnerability + (agentR() - 0.5) * jit, 0, 1);
      state.impulse_control = clamp(state.impulse_control + (agentR() - 0.5) * jit, 0, 1);

      // Action flips — more likely as chaos + amplification grow
      if (agentR() < 0.015 * params.chaosFactor * amplification) {
        // Actions are weighted by emotional state for realism
        const possibleActions: [string, number][] = [
          ['CONFRONT', state.tension > 0.3 ? 2 : 0.5],
          ['WITHDRAW', state.energy < 0.4 ? 2 : 0.5],
          ['LASH_OUT', state.impulse_control < 0.3 ? 2 : 0.3],
          ['FLEE', state.vulnerability > 0.4 ? 2 : 0.3],
          ['VENT', state.arousal > 0.3 ? 1.5 : 0.5],
          ['SOCIALIZE', state.valence > 0.4 ? 1.5 : 0.5],
          ['HELP_OTHERS', state.valence > 0.5 ? 1.5 : 0.3],
          ['RUMINATE', state.valence < 0.4 ? 1.5 : 0.3],
          ['SEEK_COMFORT', state.vulnerability > 0.3 ? 1.5 : 0.3],
          ['CELEBRATE', state.valence > 0.7 ? 2 : 0.1],
          ['IDLE', 0.5],
        ];
        const totalWeight = possibleActions.reduce((s, [, w]) => s + w, 0);
        let pick = agentR() * totalWeight;
        for (const [action, weight] of possibleActions) {
          pick -= weight;
          if (pick <= 0) { state.action = action; break; }
        }
      }

      // Emotional state label shifts
      if (agentR() < 0.008 * params.chaosFactor * amplification) {
        const emotions = ['angry', 'anxious', 'defeated', 'sad', 'neutral', 'content', 'excited'];
        state.internal = emotions[Math.floor(agentR() * emotions.length)];
        if (agentR() > 0.4) state.surface = state.internal;
        else state.divergence = clamp(state.divergence + 0.15, 0, 0.8);
      }
    }

    // Interaction type shifts
    for (const ix of tick.interactions) {
      const ixR = mulberry32(params.seed + i * 100 + hashStr(ix.agent_a));
      if (ixR() < 0.025 * params.chaosFactor * amplification) {
        // Weight interaction flips by the agents' states
        const agentA = tick.agent_states[ix.agent_a];
        if (agentA) {
          if (agentA.tension > 0.5 && ixR() < 0.5) ix.type = 'conflict';
          else if (agentA.valence > 0.5 && ixR() < 0.5) ix.type = 'support';
          else {
            const types = ['conflict', 'support', 'neutral', 'positive'];
            ix.type = types[Math.floor(ixR() * types.length)];
          }
        }
      }
    }

    // New interactions can emerge from disrupted schedules
    // (agents who are now at the same location unexpectedly)
    if (disruptedAgents.size > 0 && i % 3 === 0) {
      const tickAgents = Object.values(tick.agent_states) as any[];
      const byLoc = new Map<string, any[]>();
      for (const a of tickAgents) {
        if (!byLoc.has(a.location)) byLoc.set(a.location, []);
        byLoc.get(a.location)!.push(a);
      }
      for (const [loc, agents] of byLoc) {
        if (agents.length < 2) continue;
        const newIxR = mulberry32(params.seed + i * 200 + hashStr(loc));
        if (newIxR() < 0.15 * params.chaosFactor) {
          // Two random agents at this location interact
          const a = agents[Math.floor(newIxR() * agents.length)];
          const b = agents[Math.floor(newIxR() * agents.length)];
          if (a.id !== b.id) {
            const avgValence = (a.valence + b.valence) / 2;
            const type = avgValence < 0.3 ? 'conflict' : avgValence < 0.5 ? 'neutral' : 'support';
            tick.interactions.push({
              agent_a: a.id,
              agent_b: b.id,
              type,
              location: loc,
            });
          }
        }
      }
    }
  }

  // ─── EVENT DISRUPTION ─────────────────────────────────
  for (const tick of snapshot.ticks) {
    if (tick.events.length > 0) {
      // Some events might not fire in this universe
      tick.events = tick.events.filter(() => r() > 0.12 * params.chaosFactor);
    }
    // New events can appear in high-chaos universes
    if (r() < 0.005 * params.chaosFactor && allLocations.length > 0) {
      const loc = allLocations[Math.floor(r() * allLocations.length)];
      const newEvents = [
        { description: 'A power outage hits the neighborhood', kind: 'environmental' },
        { description: 'An anonymous tip reaches the local paper', kind: 'rumor_wave' },
        { description: 'A water main breaks, flooding the street', kind: 'environmental' },
        { description: 'Someone starts a heated argument in public', kind: 'conflict_flashpoint' },
        { description: 'A surprise community potluck is organized', kind: 'mutual_aid_hub' },
        { description: 'Police are seen interviewing witnesses outside', kind: 'whistleblower_leak' },
        { description: 'A car accident blocks the main road', kind: 'environmental' },
        { description: 'Someone spray-paints protest slogans on the wall', kind: 'organizing_meeting' },
      ];
      const evt = newEvents[Math.floor(r() * newEvents.length)];
      tick.events.push({ ...evt, location: loc, targets: null });
    }
  }
}

function hashStr(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}
