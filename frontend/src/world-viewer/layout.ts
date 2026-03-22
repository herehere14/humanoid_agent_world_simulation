/** Procedural town layout engine.
 *
 * Takes any set of locations from a simulation snapshot and generates
 * a coherent 3D town layout with appropriate building archetypes,
 * district grouping, roads, and agent placement.
 *
 * Works for 6 locations or 60 — no hardcoded positions.
 */

import type { LocationLayout, LocationMeta } from './types';

// ============================================================
// Building archetype classification
// ============================================================

export type BuildingArchetype =
  | 'office'      // corporate, business, trading
  | 'factory'     // industrial, warehouse, manufacturing
  | 'residential' // homes, apartments
  | 'civic'       // city hall, courthouse, government
  | 'education'   // school, university, library
  | 'medical'     // hospital, clinic
  | 'worship'     // church, temple, mosque
  | 'food'        // restaurant, café, food court, canteen
  | 'bar'         // bar, pub, tavern
  | 'market'      // market, shops, artisan
  | 'park'        // park, garden, plaza, waterfront
  | 'community'   // community center, union hall
  | 'port'        // docks, harbor, marina
  | 'generic';    // fallback

interface ArchetypeStyle {
  color: string;
  heightRange: [number, number];    // min, max building height
  footprintRange: [number, number]; // min, max footprint side
  flat?: boolean;                    // park-like flat area
  district?: string;                 // preferred district grouping
}

const ARCHETYPE_STYLES: Record<BuildingArchetype, ArchetypeStyle> = {
  office:      { color: '#64748b', heightRange: [7, 12], footprintRange: [10, 16], district: 'downtown' },
  factory:     { color: '#78716c', heightRange: [6, 9],  footprintRange: [14, 20], district: 'industrial' },
  residential: { color: '#a3866a', heightRange: [4, 6],  footprintRange: [12, 20], district: 'residential' },
  civic:       { color: '#d4c5a9', heightRange: [8, 12], footprintRange: [12, 16], district: 'civic' },
  education:   { color: '#dc2626', heightRange: [5, 8],  footprintRange: [10, 14], district: 'education' },
  medical:     { color: '#f0f4f8', heightRange: [6, 10], footprintRange: [10, 14], district: 'civic' },
  worship:     { color: '#7c6f64', heightRange: [8, 14], footprintRange: [7, 10],  district: 'community' },
  food:        { color: '#ea580c', heightRange: [3, 5],  footprintRange: [6, 10],  district: 'commercial' },
  bar:         { color: '#b45309', heightRange: [4, 6],  footprintRange: [6, 9],   district: 'commercial' },
  market:      { color: '#d97706', heightRange: [3, 5],  footprintRange: [8, 14],  district: 'commercial' },
  park:        { color: '#16a34a', heightRange: [0.2, 0.3], footprintRange: [14, 22], flat: true, district: 'green' },
  community:   { color: '#6366f1', heightRange: [4, 6],  footprintRange: [8, 12],  district: 'community' },
  port:        { color: '#475569', heightRange: [4, 6],  footprintRange: [10, 16], district: 'waterfront' },
  generic:     { color: '#94a3b8', heightRange: [4, 7],  footprintRange: [8, 12],  district: 'misc' },
};

/** Keyword patterns for classifying location archetype.
 *  Order matters — more specific patterns must come before general ones.
 *  e.g. "food_court" must match food before court→civic,
 *       "harbor_bar" must match bar before harbor→port.
 */
const ARCHETYPE_KEYWORDS: [BuildingArchetype, RegExp][] = [
  // Most specific first
  ['food',        /canteen|food.?court|cafeteria|café|cafe|restaurant|diner|kitchen|bistro|grill|eatery/i],
  ['bar',         /\bbar\b|pub\b|tavern|saloon|nightclub|lounge|anchor|mahoney|tap\b/i],
  ['factory',     /factor|warehouse|industr|chemical|manufactur|plant\b|mill\b|refiner/i],
  ['port',        /\bdock|harbor|harbour|marina|pier|wharf|waterfront|fish.?market/i],
  ['civic',       /city.?hall|courthouse|government|gov.?office|capitol|municipal|town.?hall/i],
  ['medical',     /hospital|clinic|medical|health.?center|emergency|pharma|doctor/i],
  ['education',   /school|university|college|lecture|library|campus|academy|institut/i],
  ['worship',     /church|temple|mosque|synagogue|chapel|cathedral|parish|pastor|worship/i],
  ['office',      /office|tower|corporate|business|trading|financial|firm|headquarter/i],
  ['residential', /home\b|house\b|residen|apartment|suburb|dwelling|quarters/i],
  ['market',      /market|shop\b|store\b|bazaar|mall\b|artisan|vendor|stall|boutique|retail/i],
  ['park',        /park\b|garden|plaza|square\b|green\b|field\b|pond|lake|river|beach|nature/i],
  ['community',   /community|union\b|center\b|centre|hall\b|club\b|recreation|gathering/i],
];

export function classifyArchetype(id: string, name: string, activity: string): BuildingArchetype {
  const text = `${id} ${name} ${activity}`.toLowerCase();
  for (const [archetype, pattern] of ARCHETYPE_KEYWORDS) {
    if (pattern.test(text)) return archetype;
  }
  return 'generic';
}

// ============================================================
// Seeded random for deterministic layouts
// ============================================================

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function hashString(str: string): number {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash + str.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

// ============================================================
// Procedural town generation
// ============================================================

/** Generate a complete town layout from location metadata */
export function generateTownLayout(
  locations: Record<string, LocationMeta>,
): LocationLayout[] {
  const entries = Object.values(locations);
  const n = entries.length;
  if (n === 0) return [];

  // Classify all locations
  const classified = entries.map(loc => ({
    ...loc,
    archetype: classifyArchetype(loc.id, loc.name, loc.default_activity),
  }));

  // Group by district preference
  const districts = new Map<string, typeof classified>();
  for (const loc of classified) {
    const dist = ARCHETYPE_STYLES[loc.archetype].district ?? 'misc';
    if (!districts.has(dist)) districts.set(dist, []);
    districts.get(dist)!.push(loc);
  }

  // Lay out districts in a radial pattern around center
  // Scale the town radius based on location count
  const townRadius = Math.max(30, Math.sqrt(n) * 14);
  const districtNames = [...districts.keys()];
  const districtCount = districtNames.length;

  // Assign each district an angular sector
  const districtAngles = new Map<string, number>();
  // Preferred angular positions for districts (clockwise from north)
  const PREFERRED_ANGLES: Record<string, number> = {
    downtown:    0,              // north-center
    civic:       Math.PI * 0.3,  // northeast
    commercial:  Math.PI * 0.6,  // east
    waterfront:  Math.PI * 0.85, // southeast
    industrial:  Math.PI * 1.1,  // south
    residential: Math.PI * 1.4,  // southwest
    community:   Math.PI * 1.65, // west
    education:   Math.PI * 1.85, // northwest
    green:       Math.PI * 0.45, // between civic and commercial
    misc:        Math.PI * 1.2,  // south-southwest
  };

  // Sort districts by preferred angle, then fill gaps
  const sortedDistricts = districtNames.sort((a, b) => {
    const aAngle = PREFERRED_ANGLES[a] ?? Math.PI;
    const bAngle = PREFERRED_ANGLES[b] ?? Math.PI;
    return aAngle - bAngle;
  });

  sortedDistricts.forEach((name, i) => {
    const preferred = PREFERRED_ANGLES[name];
    if (preferred !== undefined) {
      districtAngles.set(name, preferred);
    } else {
      districtAngles.set(name, (i / districtCount) * Math.PI * 2);
    }
  });

  // Place each location
  const layouts: LocationLayout[] = [];
  const occupied: Array<{ x: number; z: number; r: number }> = [];

  for (const [distName, locs] of districts) {
    const baseAngle = districtAngles.get(distName) ?? 0;
    const distRadius = townRadius * (0.4 + 0.3 * (locs.length > 3 ? 1 : 0.7));

    locs.forEach((loc, i) => {
      const style = ARCHETYPE_STYLES[loc.archetype];
      const rand = seededRandom(hashString(loc.id));

      // Size from archetype ranges
      const footprint = style.footprintRange[0] +
        rand() * (style.footprintRange[1] - style.footprintRange[0]);
      const height = style.heightRange[0] +
        rand() * (style.heightRange[1] - style.heightRange[0]);
      const depth = footprint * (0.7 + rand() * 0.4);

      // Position: spread within the district's sector
      const spreadAngle = Math.PI * 0.4 * (locs.length > 1 ? 1 : 0.3);
      const locAngle = baseAngle + (i / Math.max(locs.length - 1, 1) - 0.5) * spreadAngle;
      const locRadius = distRadius * (0.6 + rand() * 0.5);

      let x = Math.sin(locAngle) * locRadius;
      let z = -Math.cos(locAngle) * locRadius;

      // Collision avoidance: nudge away from occupied spots
      const minSep = footprint * 0.8 + 4;
      for (let attempt = 0; attempt < 8; attempt++) {
        let collision = false;
        for (const occ of occupied) {
          const dx = x - occ.x;
          const dz = z - occ.z;
          const dist = Math.sqrt(dx * dx + dz * dz);
          const needed = minSep + occ.r;
          if (dist < needed) {
            // Push away
            const pushDir = dist > 0.1 ? { x: dx / dist, z: dz / dist } : { x: rand() - 0.5, z: rand() - 0.5 };
            x += pushDir.x * (needed - dist + 2);
            z += pushDir.z * (needed - dist + 2);
            collision = true;
          }
        }
        if (!collision) break;
      }

      occupied.push({ x, z, r: footprint * 0.6 });

      layouts.push({
        id: loc.id,
        position: [Math.round(x * 10) / 10, 0, Math.round(z * 10) / 10],
        size: [
          Math.round(footprint * 10) / 10,
          Math.round(height * 10) / 10,
          Math.round(depth * 10) / 10,
        ],
        color: style.color,
        label: abbreviateLabel(loc.name),
      });
    });
  }

  return layouts;
}

/** Shorten location names for 3D labels */
function abbreviateLabel(name: string): string {
  // Remove common prefixes/suffixes
  return name
    .replace(/^(The|A)\s+/i, '')
    .replace(/\s+(Building|Complex|Area|District|Zone)$/i, '')
    .slice(0, 24);
}

// ============================================================
// Computed layout cache (regenerated when snapshot changes)
// ============================================================

let _cachedLayouts: LocationLayout[] = [];
let _cachedLayoutMap = new Map<string, LocationLayout>();
let _cachedLocationHash = '';

/** Initialize layout from snapshot locations. Call once when snapshot loads. */
export function initLayout(locations: Record<string, LocationMeta>): LocationLayout[] {
  const hash = Object.keys(locations).sort().join(',');
  if (hash === _cachedLocationHash && _cachedLayouts.length > 0) {
    return _cachedLayouts;
  }

  _cachedLayouts = generateTownLayout(locations);
  _cachedLayoutMap = new Map(_cachedLayouts.map(l => [l.id, l]));
  _cachedLocationHash = hash;
  return _cachedLayouts;
}

/** Get current layout list */
export function getLayouts(): LocationLayout[] {
  return _cachedLayouts;
}

/** Get layout for a specific location */
export function getLocationLayout(id: string): LocationLayout | undefined {
  return _cachedLayoutMap.get(id);
}

// ============================================================
// Agent positioning (works with any layout)
// ============================================================

/** Get a position for an agent at a location, with jitter to avoid stacking */
export function getAgentPosition(
  locationId: string,
  agentIndex: number,
  totalAtLocation: number,
): [number, number, number] {
  const layout = _cachedLayoutMap.get(locationId);
  if (!layout) {
    // Unknown location — scatter around origin
    const angle = (agentIndex / Math.max(totalAtLocation, 1)) * Math.PI * 2;
    return [Math.cos(angle) * 5, 0, Math.sin(angle) * 5];
  }

  const [cx, , cz] = layout.position;
  const [sx, , sz] = layout.size;

  // Spread agents around the building
  const spreadRadius = Math.max(sx, sz) * 0.6 + 2;

  if (totalAtLocation <= 1) {
    return [cx + spreadRadius * 0.5, 0, cz + spreadRadius * 0.3];
  }

  // Golden-angle spiral for uniform distribution
  const goldenAngle = 2.399963;
  const angle = agentIndex * goldenAngle;
  const radius = spreadRadius * 0.3 + Math.sqrt(agentIndex / totalAtLocation) * spreadRadius * 0.7;

  return [
    cx + Math.cos(angle) * radius,
    0,
    cz + Math.sin(angle) * radius,
  ];
}

// ============================================================
// Procedural environment helpers
// ============================================================

/** Generate tree positions that adapt to the town layout */
export function generateTreePositions(layouts: LocationLayout[]): [number, number, number][] {
  if (layouts.length === 0) return [];

  const trees: [number, number, number][] = [];
  const rand = seededRandom(42);

  // Find parks and green spaces — cluster trees there
  const parks = layouts.filter(l => ARCHETYPE_STYLES[classifyArchetype(l.id, l.label, '')].flat);
  for (const park of parks) {
    const [px, , pz] = park.position;
    const r = Math.max(park.size[0], park.size[2]) * 0.5;
    const treeCount = Math.floor(r * 1.5);
    for (let i = 0; i < treeCount; i++) {
      const angle = rand() * Math.PI * 2;
      const dist = r * 0.3 + rand() * r * 0.8;
      trees.push([px + Math.cos(angle) * dist, 0, pz + Math.sin(angle) * dist]);
    }
  }

  // Scatter trees along the perimeter and between buildings
  const bounds = getTownBounds(layouts);
  const roadSpacing = 15;
  for (let x = bounds.minX - 10; x < bounds.maxX + 10; x += roadSpacing) {
    for (let z = bounds.minZ - 10; z < bounds.maxZ + 10; z += roadSpacing) {
      // Don't place trees inside buildings
      const tooClose = layouts.some(l => {
        const dx = Math.abs(x - l.position[0]);
        const dz = Math.abs(z - l.position[2]);
        return dx < l.size[0] * 0.7 + 2 && dz < l.size[2] * 0.7 + 2;
      });
      if (!tooClose && rand() > 0.55) {
        trees.push([x + (rand() - 0.5) * 4, 0, z + (rand() - 0.5) * 4]);
      }
    }
  }

  return trees;
}

/** Generate road segments connecting buildings */
export function generateRoads(layouts: LocationLayout[]): Array<{
  start: [number, number];
  end: [number, number];
  width: number;
}> {
  if (layouts.length < 2) return [];

  const roads: Array<{ start: [number, number]; end: [number, number]; width: number }> = [];
  const bounds = getTownBounds(layouts);
  const cx = (bounds.minX + bounds.maxX) / 2;
  const cz = (bounds.minZ + bounds.maxZ) / 2;

  // Main cross-roads through center
  roads.push({ start: [bounds.minX - 10, cz], end: [bounds.maxX + 10, cz], width: 4 });
  roads.push({ start: [cx, bounds.minZ - 10], end: [cx, bounds.maxZ + 10], width: 4 });

  // Secondary roads connecting nearby buildings
  const sorted = [...layouts].sort((a, b) => {
    const da = Math.atan2(a.position[2] - cz, a.position[0] - cx);
    const db = Math.atan2(b.position[2] - cz, b.position[0] - cx);
    return da - db;
  });

  for (let i = 0; i < sorted.length; i++) {
    const a = sorted[i];
    const b = sorted[(i + 1) % sorted.length];
    roads.push({
      start: [a.position[0], a.position[2]],
      end: [b.position[0], b.position[2]],
      width: 2.5,
    });
  }

  return roads;
}

/** Get bounding box of all buildings */
export function getTownBounds(layouts: LocationLayout[]): {
  minX: number; maxX: number; minZ: number; maxZ: number;
  centerX: number; centerZ: number; radius: number;
} {
  if (layouts.length === 0) {
    return { minX: -50, maxX: 50, minZ: -50, maxZ: 50, centerX: 0, centerZ: 0, radius: 50 };
  }
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const l of layouts) {
    const halfW = l.size[0] / 2 + 5;
    const halfD = l.size[2] / 2 + 5;
    minX = Math.min(minX, l.position[0] - halfW);
    maxX = Math.max(maxX, l.position[0] + halfW);
    minZ = Math.min(minZ, l.position[2] - halfD);
    maxZ = Math.max(maxZ, l.position[2] + halfD);
  }
  const centerX = (minX + maxX) / 2;
  const centerZ = (minZ + maxZ) / 2;
  const radius = Math.max(maxX - minX, maxZ - minZ) / 2;
  return { minX, maxX, minZ, maxZ, centerX, centerZ, radius };
}

// ============================================================
// Behavior / emotion mapping (unchanged — works for any scenario)
// ============================================================

export type BehaviorClass =
  | 'idle' | 'walking' | 'working' | 'resting'
  | 'confronting' | 'fleeing' | 'collapsed'
  | 'socializing' | 'helping' | 'venting'
  | 'celebrating' | 'withdrawing' | 'ruminating';

export function actionToBehavior(action: string): BehaviorClass {
  switch (action) {
    case 'COLLAPSE': return 'collapsed';
    case 'LASH_OUT':
    case 'CONFRONT': return 'confronting';
    case 'FLEE': return 'fleeing';
    case 'WITHDRAW': return 'withdrawing';
    case 'SEEK_COMFORT': return 'socializing';
    case 'RUMINATE': return 'ruminating';
    case 'VENT': return 'venting';
    case 'SOCIALIZE': return 'socializing';
    case 'CELEBRATE': return 'celebrating';
    case 'HELP_OTHERS': return 'helping';
    case 'WORK': return 'working';
    case 'REST': return 'resting';
    case 'IDLE':
    default: return 'idle';
  }
}

export function getEmotionColor(valence: number, arousal: number, _vulnerability: number): string {
  if (valence < 0.25) {
    if (arousal > 0.5) return '#ef4444';
    return '#6b7280';
  }
  if (valence < 0.4) {
    if (arousal > 0.4) return '#f59e0b';
    return '#94a3b8';
  }
  if (valence > 0.7) {
    if (arousal > 0.5) return '#22c55e';
    return '#10b981';
  }
  return '#60a5fa';
}

export function getBehaviorPosture(behavior: BehaviorClass): { scaleY: number; lean: number } {
  switch (behavior) {
    case 'collapsed': return { scaleY: 0.5, lean: 0.3 };
    case 'confronting': return { scaleY: 1.05, lean: -0.15 };
    case 'fleeing': return { scaleY: 0.9, lean: 0.2 };
    case 'withdrawing': return { scaleY: 0.85, lean: 0.1 };
    case 'ruminating': return { scaleY: 0.88, lean: 0.15 };
    case 'celebrating': return { scaleY: 1.1, lean: -0.05 };
    case 'helping': return { scaleY: 1.0, lean: -0.1 };
    case 'venting': return { scaleY: 1.02, lean: -0.08 };
    case 'socializing': return { scaleY: 1.0, lean: 0 };
    case 'working': return { scaleY: 0.95, lean: 0.05 };
    case 'resting': return { scaleY: 0.7, lean: 0.2 };
    default: return { scaleY: 1.0, lean: 0 };
  }
}
