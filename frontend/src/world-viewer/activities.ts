/**
 * Personal activity system — expands broad simulation actions into
 * specific human activities based on personality, location, time, and emotion.
 *
 * "WORK at office" becomes "typing at computer" or "in a meeting" or "getting coffee".
 * "IDLE at home" becomes "watching TV" or "cooking dinner" or "scrolling phone".
 *
 * Each sub-activity has:
 *   - A unique pose override for the avatar
 *   - A prop to spawn near the agent (desk, TV, phone, book, cup, etc.)
 *   - A label for the UI
 */

export type SubActivity =
  // Work activities
  | 'typing'          // at desk, hands on keyboard
  | 'meeting'         // standing, gesturing, speaking
  | 'phone_call'      // one hand to ear
  | 'coffee_break'    // holding cup, standing
  | 'filing'          // looking down, shuffling papers
  | 'presenting'      // standing, arm raised toward imaginary board
  // Home activities
  | 'watching_tv'     // seated, reclined, head forward
  | 'cooking'         // standing, arms working in front
  | 'reading'         // seated, hands up holding book
  | 'scrolling_phone' // seated or standing, one hand up, head tilted down
  | 'cleaning'        // bending, arm sweeping
  | 'exercising'      // squatting / arm raising
  | 'staring_window'  // standing still, head turned sideways
  | 'pacing_home'     // walking in small circle
  | 'lying_awake'     // lying down but eyes-open head position
  | 'playing_guitar'  // seated, arms in guitar position
  // Bar activities
  | 'drinking'        // one hand up, tilting
  | 'playing_pool'    // leaning forward, arms extended
  | 'bar_chatting'    // seated at counter, turned sideways
  | 'sitting_booth'   // seated, slumped
  | 'dancing'         // bouncing, arms moving
  // Park activities
  | 'bench_sitting'   // seated, relaxed
  | 'jogging'         // run cycle
  | 'walking_dog'     // walking with arm extended down
  | 'feeding_birds'   // seated, arm tossing forward
  | 'meditating'      // cross-legged, hands on knees
  // Church/community activities
  | 'praying'         // head down, hands clasped
  | 'singing'         // head up, arms slightly out
  | 'listening'       // seated, head tilted, attentive
  // School activities
  | 'teaching'        // standing, arm gesturing at board
  | 'studying'        // seated, head down in book
  // Hospital
  | 'treating'        // standing, arms forward, caring gesture
  | 'waiting_room'    // seated, fidgeting
  // Generic fallbacks
  | 'standing_idle'   // default
  | 'sitting_idle';   // seated default

export interface ActivityInfo {
  sub: SubActivity;
  label: string;
  prop: PropType | null;
  seated: boolean;
}

export type PropType =
  | 'desk' | 'computer' | 'phone' | 'coffee_cup' | 'papers'
  | 'tv' | 'book' | 'guitar' | 'broom' | 'dumbbell'
  | 'beer' | 'pool_cue' | 'bird_seed'
  | 'none';

// Seeded per-agent randomness
function agentRand(agentId: string, tick: number, salt: number): number {
  let h = 5381;
  const s = agentId + tick + '' + salt;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return (Math.abs(h) % 1000) / 1000;
}

/**
 * Determine what specific activity an agent is doing right now.
 */
export function getSubActivity(
  agentId: string,
  action: string,
  location: string,
  hour: number,
  valence: number,
  energy: number,
  tension: number,
  temperament: string,
  tick: number,
): ActivityInfo {
  const r = (salt: number) => agentRand(agentId, tick, salt);
  const temp = temperament.toLowerCase();
  const isNight = hour >= 20 || hour < 6;
  const isEvening = hour >= 17 && hour < 20;
  const isMorning = hour >= 6 && hour < 9;

  // ─── WORK ────────────────────────────────────────────
  if (action === 'WORK') {
    if (/office|tower|trading|business/i.test(location)) {
      const roll = r(1);
      if (roll < 0.45) return { sub: 'typing', label: 'working at computer', prop: 'computer', seated: true };
      if (roll < 0.65) return { sub: 'meeting', label: 'in a meeting', prop: 'none', seated: false };
      if (roll < 0.78) return { sub: 'phone_call', label: 'on a phone call', prop: 'phone', seated: false };
      if (roll < 0.88) return { sub: 'coffee_break', label: 'getting coffee', prop: 'coffee_cup', seated: false };
      if (roll < 0.95) return { sub: 'filing', label: 'reviewing documents', prop: 'papers', seated: true };
      return { sub: 'presenting', label: 'presenting to team', prop: 'none', seated: false };
    }
    if (/school|lecture|library/i.test(location)) {
      return r(2) < 0.6
        ? { sub: 'teaching', label: 'teaching class', prop: 'none', seated: false }
        : { sub: 'studying', label: 'grading papers', prop: 'papers', seated: true };
    }
    if (/hospital|clinic/i.test(location)) {
      return { sub: 'treating', label: 'treating patients', prop: 'none', seated: false };
    }
    if (/factory|warehouse|dock/i.test(location)) {
      return { sub: 'standing_idle', label: 'working on floor', prop: 'none', seated: false };
    }
    if (/bar|pub|tap/i.test(location)) {
      return { sub: 'standing_idle', label: 'tending bar', prop: 'none', seated: false };
    }
    if (/market|shop|artisan/i.test(location)) {
      return { sub: 'standing_idle', label: 'manning the stall', prop: 'none', seated: false };
    }
    return { sub: 'typing', label: 'working', prop: 'computer', seated: true };
  }

  // ─── REST ────────────────────────────────────────────
  if (action === 'REST') {
    if (tension > 0.4 || valence < 0.25) {
      return { sub: 'lying_awake', label: 'lying awake, can\'t sleep', prop: 'none', seated: false };
    }
    return { sub: 'lying_awake', label: 'sleeping', prop: 'none', seated: false };
  }

  // ─── IDLE ────────────────────────────────────────────
  if (action === 'IDLE') {
    // Home idle — rich variety
    if (/home|residen|suburb|apartment/i.test(location)) {
      if (isEvening) {
        const roll = r(3);
        if (roll < 0.3) return { sub: 'watching_tv', label: 'watching TV', prop: 'tv', seated: true };
        if (roll < 0.5) return { sub: 'cooking', label: 'making dinner', prop: 'none', seated: false };
        if (roll < 0.65) return { sub: 'scrolling_phone', label: 'scrolling phone', prop: 'phone', seated: true };
        if (roll < 0.78) return { sub: 'reading', label: 'reading a book', prop: 'book', seated: true };
        if (/artistic|creative|music/i.test(temp)) {
          return { sub: 'playing_guitar', label: 'playing guitar', prop: 'guitar', seated: true };
        }
        return { sub: 'watching_tv', label: 'watching the news', prop: 'tv', seated: true };
      }
      if (isMorning) {
        const roll = r(4);
        if (roll < 0.4) return { sub: 'coffee_break', label: 'having morning coffee', prop: 'coffee_cup', seated: false };
        if (roll < 0.6) return { sub: 'scrolling_phone', label: 'checking phone', prop: 'phone', seated: true };
        if (roll < 0.75 && /exercis|active|coach/i.test(temp)) {
          return { sub: 'exercising', label: 'morning workout', prop: 'dumbbell', seated: false };
        }
        return { sub: 'cooking', label: 'making breakfast', prop: 'none', seated: false };
      }
      // Daytime at home (laid off, day off, etc)
      if (valence < 0.3) {
        const roll = r(5);
        if (roll < 0.35) return { sub: 'staring_window', label: 'staring out the window', prop: 'none', seated: false };
        if (roll < 0.6) return { sub: 'pacing_home', label: 'pacing the house', prop: 'none', seated: false };
        if (roll < 0.8) return { sub: 'scrolling_phone', label: 'doom-scrolling phone', prop: 'phone', seated: true };
        return { sub: 'lying_awake', label: 'lying on couch, staring at ceiling', prop: 'none', seated: false };
      }
      if (energy > 0.6) {
        return r(6) < 0.5
          ? { sub: 'cleaning', label: 'cleaning the house', prop: 'broom', seated: false }
          : { sub: 'exercising', label: 'working out', prop: 'dumbbell', seated: false };
      }
      return { sub: 'watching_tv', label: 'watching TV', prop: 'tv', seated: true };
    }

    // Park idle
    if (/park|garden|central/i.test(location)) {
      const roll = r(7);
      if (roll < 0.35) return { sub: 'bench_sitting', label: 'sitting on a bench', prop: 'none', seated: true };
      if (roll < 0.5) return { sub: 'jogging', label: 'jogging', prop: 'none', seated: false };
      if (roll < 0.65) return { sub: 'feeding_birds', label: 'feeding the birds', prop: 'bird_seed', seated: true };
      if (roll < 0.8) return { sub: 'meditating', label: 'meditating quietly', prop: 'none', seated: true };
      return { sub: 'walking_dog', label: 'walking around the park', prop: 'none', seated: false };
    }

    // Bar idle
    if (/bar|pub|tap|anchor/i.test(location)) {
      const roll = r(8);
      if (roll < 0.4) return { sub: 'drinking', label: 'nursing a drink', prop: 'beer', seated: true };
      if (roll < 0.6) return { sub: 'bar_chatting', label: 'chatting at the bar', prop: 'beer', seated: true };
      if (roll < 0.75) return { sub: 'playing_pool', label: 'playing pool', prop: 'pool_cue', seated: false };
      if (roll < 0.85 && energy > 0.5) return { sub: 'dancing', label: 'dancing', prop: 'none', seated: false };
      return { sub: 'sitting_booth', label: 'sitting alone in a booth', prop: 'beer', seated: true };
    }

    // Church idle
    if (/church|chapel|temple/i.test(location)) {
      return r(9) < 0.5
        ? { sub: 'praying', label: 'praying quietly', prop: 'none', seated: true }
        : { sub: 'listening', label: 'sitting in contemplation', prop: 'none', seated: true };
    }

    // Hospital idle
    if (/hospital|clinic/i.test(location)) {
      return { sub: 'waiting_room', label: 'waiting in the waiting room', prop: 'none', seated: true };
    }

    return { sub: 'standing_idle', label: 'hanging around', prop: 'none', seated: false };
  }

  // ─── RUMINATE ────────────────────────────────────────
  if (action === 'RUMINATE') {
    if (/home/i.test(location)) {
      const roll = r(10);
      if (roll < 0.3) return { sub: 'pacing_home', label: 'pacing back and forth', prop: 'none', seated: false };
      if (roll < 0.6) return { sub: 'staring_window', label: 'staring out the window, lost in thought', prop: 'none', seated: false };
      return { sub: 'sitting_idle', label: 'sitting alone, spiraling', prop: 'none', seated: true };
    }
    return { sub: 'standing_idle', label: 'lost in thought', prop: 'none', seated: false };
  }

  // ─── WITHDRAW ────────────────────────────────────────
  if (action === 'WITHDRAW') {
    return { sub: 'staring_window', label: 'withdrawn, avoiding everyone', prop: 'none', seated: false };
  }

  // ─── SOCIALIZE ───────────────────────────────────────
  if (action === 'SOCIALIZE') {
    if (/bar|pub/i.test(location)) {
      return r(11) < 0.6
        ? { sub: 'drinking', label: 'having drinks with friends', prop: 'beer', seated: true }
        : { sub: 'dancing', label: 'dancing with the group', prop: 'none', seated: false };
    }
    return { sub: 'standing_idle', label: 'chatting with people', prop: 'none', seated: false };
  }

  // ─── CELEBRATE ───────────────────────────────────────
  if (action === 'CELEBRATE') {
    if (/bar|pub/i.test(location)) {
      return { sub: 'dancing', label: 'celebrating at the bar', prop: 'beer', seated: false };
    }
    return { sub: 'standing_idle', label: 'celebrating', prop: 'none', seated: false };
  }

  // All other actions use the default pose system
  return { sub: 'standing_idle', label: action.toLowerCase().replace(/_/g, ' '), prop: 'none', seated: false };
}
