// Visual condition-tree builder for the desktop sidebar. Produces the
// recursive `conditions` JSON the Tauri start_search command accepts, so users
// don't have to hand-craft JSON. Mirrors the schema in
// src/mcseedfinder/criteria.py and crates/mcseedfinder-core/src/conditions.rs.

import React from "react";
import { BIOMES, BIOME_GROUPS, biomeLabel } from "./biomes";

// ---------------------------------------------------------------------------
// Tree types (with UI-only `id` field for stable React keys + path-free edits)
// ---------------------------------------------------------------------------

export type GroupKind = "all_of" | "any_of" | "none_of";
export type LeafKind =
  | "nearby_structure"
  | "cluster"
  | "spawn_biome"
  | "nearby_biomes"
  | "biome_area";
export type NodeKind = GroupKind | LeafKind;

export type GroupNode = { id: string; type: GroupKind; of: TreeNode[] };
export type NearbyStructureNode = {
  id: string;
  type: "nearby_structure";
  structure: string;
  max_distance: number;
  centre_x: number;
  centre_z: number;
};
export type ClusterNode = {
  id: string;
  type: "cluster";
  structures: string[];
  max_distance: number;
  min_count: number;
  centre_x: number;
  centre_z: number;
};
export type SpawnBiomeNode = {
  id: string;
  type: "spawn_biome";
  biomes: number[];
  spawn_radius: number;
};
export type NearbyBiomesNode = {
  id: string;
  type: "nearby_biomes";
  biomes: number[];
  radius: number;
  all: boolean;
  samples_per_axis: number;
};
export type BiomeAreaNode = {
  id: string;
  type: "biome_area";
  biomes: number[];
  radius: number;
  samples_per_axis: number;
  min_samples: number;
  centre_x: number;
  centre_z: number;
};

export type TreeNode =
  | GroupNode
  | NearbyStructureNode
  | ClusterNode
  | SpawnBiomeNode
  | NearbyBiomesNode
  | BiomeAreaNode;

const STRUCTURES = [
  "village",
  "pillager_outpost",
  "ocean_monument",
  "stronghold",
  "woodland_mansion",
  "desert_pyramid",
  "jungle_temple",
  "igloo",
  "swamp_hut",
  "shipwreck",
  "buried_treasure",
  "ruined_portal",
  "ancient_city",
  "trial_chambers",
];

let _idCounter = 0;
function newId(): string {
  _idCounter += 1;
  return `n${_idCounter}`;
}

// ---------------------------------------------------------------------------
// Constructors / converters
// ---------------------------------------------------------------------------

export function defaultLeaf(kind: LeafKind): TreeNode {
  const id = newId();
  switch (kind) {
    case "nearby_structure":
      return { id, type: "nearby_structure", structure: "village", max_distance: 1000, centre_x: 0, centre_z: 0 };
    case "cluster":
      return { id, type: "cluster", structures: ["swamp_hut"], max_distance: 128, min_count: 4, centre_x: 0, centre_z: 0 };
    case "spawn_biome":
      return { id, type: "spawn_biome", biomes: [1], spawn_radius: 64 };
    case "nearby_biomes":
      return { id, type: "nearby_biomes", biomes: [21], radius: 2000, all: false, samples_per_axis: 16 };
    case "biome_area":
      return { id, type: "biome_area", biomes: [1], radius: 1000, samples_per_axis: 16, min_samples: 8, centre_x: 0, centre_z: 0 };
  }
}

export function defaultGroup(kind: GroupKind): GroupNode {
  return { id: newId(), type: kind, of: [defaultLeaf("nearby_structure")] };
}

export function defaultRoot(): TreeNode {
  return defaultLeaf("nearby_structure");
}

// ---------------------------------------------------------------------------
// Preset templates (#8 — quad-hut / common multi-structure shapes)
// ---------------------------------------------------------------------------
//
// Each preset returns a fresh TreeNode (with fresh ids) ready to drop into
// the condition tree. The marquee one is the quad-witch-hut layout — the
// holy grail of Minecraft seed prospecting. Adding more presets is a one-
// line table entry below.

export type PresetTemplate = {
  key: string;
  label: string;
  description: string;
  build: () => TreeNode;
};

export const PRESET_TEMPLATES: PresetTemplate[] = [
  {
    key: "quad_hut",
    label: "Quad witch hut",
    description:
      "Four swamp huts within 128 blocks of (0, 0) — the classic XP / loot farm setup.",
    build: () => ({
      id: newId(),
      type: "cluster",
      structures: ["swamp_hut"],
      max_distance: 128,
      min_count: 4,
      centre_x: 0,
      centre_z: 0,
    }),
  },
  {
    key: "triple_village",
    label: "Triple village near spawn",
    description: "Three villages inside a 1500-block radius — quick trader hubs.",
    build: () => ({
      id: newId(),
      type: "cluster",
      structures: ["village"],
      max_distance: 1500,
      min_count: 3,
      centre_x: 0,
      centre_z: 0,
    }),
  },
  {
    key: "monument_mansion",
    label: "Monument + mansion near spawn",
    description:
      "Both an ocean monument and a woodland mansion within 4 000 blocks of origin.",
    build: () => ({
      id: newId(),
      type: "all_of",
      of: [
        {
          id: newId(),
          type: "nearby_structure",
          structure: "ocean_monument",
          max_distance: 4000,
          centre_x: 0,
          centre_z: 0,
        },
        {
          id: newId(),
          type: "nearby_structure",
          structure: "woodland_mansion",
          max_distance: 4000,
          centre_x: 0,
          centre_z: 0,
        },
      ],
    }),
  },
  {
    key: "deep_dark_stronghold",
    label: "Stronghold + ancient city",
    description:
      "A stronghold and an ancient city both within 3 000 blocks — speedrun-friendly.",
    build: () => ({
      id: newId(),
      type: "all_of",
      of: [
        {
          id: newId(),
          type: "nearby_structure",
          structure: "stronghold",
          max_distance: 3000,
          centre_x: 0,
          centre_z: 0,
        },
        {
          id: newId(),
          type: "nearby_structure",
          structure: "ancient_city",
          max_distance: 3000,
          centre_x: 0,
          centre_z: 0,
        },
      ],
    }),
  },
];

/** Convert a UI tree to the wire-format JSON the backend accepts.
 *
 * Defensively guards against pathological trees: the `seen` set detects
 * self-references (would infinite-recurse) and the `depth` cap catches
 * very deep trees before they blow the JS stack. Throws a clear error in
 * either case so the React error boundary surfaces something usable
 * instead of "Maximum call stack size exceeded" with no context.
 */
const NODE_MAX_DEPTH = 64;
export function nodeToWire(n: TreeNode): Record<string, unknown> {
  return nodeToWireImpl(n, new WeakSet(), 0);
}
function nodeToWireImpl(
  n: TreeNode,
  seen: WeakSet<TreeNode>,
  depth: number,
): Record<string, unknown> {
  if (depth > NODE_MAX_DEPTH) {
    throw new Error(
      `nodeToWire: tree exceeds max depth ${NODE_MAX_DEPTH} (suspect a cycle or pathological nesting)`,
    );
  }
  if (seen.has(n)) {
    throw new Error(
      `nodeToWire: cycle detected at node id=${n.id} type=${n.type} — refusing to serialize`,
    );
  }
  seen.add(n);
  switch (n.type) {
    case "all_of":
    case "any_of":
    case "none_of":
      return {
        type: n.type,
        of: n.of.map((c) => nodeToWireImpl(c, seen, depth + 1)),
      };
    case "nearby_structure":
      return { type: "nearby_structure", structure: n.structure, max_distance: n.max_distance, centre_x: n.centre_x, centre_z: n.centre_z };
    case "cluster":
      return { type: "cluster", structures: n.structures, max_distance: n.max_distance, min_count: n.min_count, centre_x: n.centre_x, centre_z: n.centre_z };
    case "spawn_biome":
      return { type: "spawn_biome", biomes: n.biomes, spawn_radius: n.spawn_radius };
    case "nearby_biomes":
      return { type: "nearby_biomes", biomes: n.biomes, radius: n.radius, all: n.all, samples_per_axis: n.samples_per_axis };
    case "biome_area":
      return { type: "biome_area", biomes: n.biomes, radius: n.radius, samples_per_axis: n.samples_per_axis, min_samples: n.min_samples, centre_x: n.centre_x, centre_z: n.centre_z };
  }
}

/**
 * Reverse of `nodeToWire`: rebuild a UI tree from the JSON shape the backend
 * accepts. Generates fresh UI ids for every node so a shared/imported spec
 * doesn't collide with what's already in the editor.
 */
export function wireToNode(wire: any): TreeNode {
  if (!wire || typeof wire !== "object") throw new Error("condition node must be an object");
  const t = wire.type as NodeKind;
  switch (t) {
    case "all_of":
    case "any_of":
    case "none_of": {
      const of = Array.isArray(wire.of) ? wire.of : [];
      return { id: newId(), type: t, of: of.map(wireToNode) };
    }
    case "nearby_structure":
      return {
        id: newId(),
        type: "nearby_structure",
        structure: String(wire.structure ?? "village"),
        max_distance: Number(wire.max_distance ?? 1500),
        centre_x: Number(wire.centre_x ?? 0),
        centre_z: Number(wire.centre_z ?? 0),
      };
    case "cluster":
      return {
        id: newId(),
        type: "cluster",
        structures: Array.isArray(wire.structures) ? wire.structures.map(String) : [],
        max_distance: Number(wire.max_distance ?? 1500),
        min_count: Number(wire.min_count ?? 4),
        centre_x: Number(wire.centre_x ?? 0),
        centre_z: Number(wire.centre_z ?? 0),
      };
    case "spawn_biome":
      return {
        id: newId(),
        type: "spawn_biome",
        biomes: Array.isArray(wire.biomes) ? wire.biomes.map(Number) : [],
        spawn_radius: Number(wire.spawn_radius ?? 64),
      };
    case "nearby_biomes":
      return {
        id: newId(),
        type: "nearby_biomes",
        biomes: Array.isArray(wire.biomes) ? wire.biomes.map(Number) : [],
        radius: Number(wire.radius ?? 2000),
        all: Boolean(wire.all),
        samples_per_axis: Number(wire.samples_per_axis ?? 16),
      };
    case "biome_area":
      return {
        id: newId(),
        type: "biome_area",
        biomes: Array.isArray(wire.biomes) ? wire.biomes.map(Number) : [],
        radius: Number(wire.radius ?? 1000),
        samples_per_axis: Number(wire.samples_per_axis ?? 16),
        min_samples: Number(wire.min_samples ?? 8),
        centre_x: Number(wire.centre_x ?? 0),
        centre_z: Number(wire.centre_z ?? 0),
      };
    default:
      throw new Error(`unknown condition type ${String(t)}`);
  }
}

/** Walk the tree and return a copy with `node.id` replaced by the patched version. */
function updateById(node: TreeNode, id: string, updater: (n: TreeNode) => TreeNode): TreeNode {
  if (node.id === id) return updater(node);
  if (node.type === "all_of" || node.type === "any_of" || node.type === "none_of") {
    return { ...node, of: node.of.map((c) => updateById(c, id, updater)) };
  }
  return node;
}

function removeById(node: TreeNode, id: string): TreeNode | null {
  if (node.id === id) return null;
  if (node.type === "all_of" || node.type === "any_of" || node.type === "none_of") {
    const filtered = node.of
      .map((c) => removeById(c, id))
      .filter((c): c is TreeNode => c !== null);
    return { ...node, of: filtered };
  }
  return node;
}

/** Morph a node to a new kind, preserving compatible fields where possible. */
function morphNode(node: TreeNode, newKind: NodeKind): TreeNode {
  if (newKind === node.type) return node;
  // Groups morph to groups while preserving children.
  if (
    (newKind === "all_of" || newKind === "any_of" || newKind === "none_of") &&
    (node.type === "all_of" || node.type === "any_of" || node.type === "none_of")
  ) {
    return { ...(node as GroupNode), type: newKind };
  }
  // Promote a leaf to a group, wrapping it as the single child.
  if (
    (newKind === "all_of" || newKind === "any_of" || newKind === "none_of") &&
    !(node.type === "all_of" || node.type === "any_of" || node.type === "none_of")
  ) {
    return { id: node.id, type: newKind, of: [node] };
  }
  // Group → leaf: drop children, use a default leaf with the kept id.
  if (
    (node.type === "all_of" || node.type === "any_of" || node.type === "none_of") &&
    (newKind === "nearby_structure" ||
      newKind === "cluster" ||
      newKind === "spawn_biome" ||
      newKind === "nearby_biomes" ||
      newKind === "biome_area")
  ) {
    return { ...defaultLeaf(newKind), id: node.id };
  }
  // Leaf → different leaf: keep id, replace.
  return { ...defaultLeaf(newKind as LeafKind), id: node.id };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

type Props = {
  root: TreeNode;
  onChange: (root: TreeNode) => void;
};

export function ConditionBuilder({ root, onChange }: Props) {
  const update = (id: string, updater: (n: TreeNode) => TreeNode) =>
    onChange(updateById(root, id, updater));
  const remove = (id: string) => {
    const next = removeById(root, id);
    onChange(next ?? defaultRoot()); // can't have an empty tree
  };

  return (
    <div className="conditionBuilder">
      <NodeView node={root} depth={0} update={update} remove={remove} isRoot />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Per-node renderer (recursive)
// ---------------------------------------------------------------------------

function NodeView({
  node,
  depth,
  update,
  remove,
  isRoot,
}: {
  node: TreeNode;
  depth: number;
  update: (id: string, updater: (n: TreeNode) => TreeNode) => void;
  remove: (id: string) => void;
  isRoot: boolean;
}) {
  return (
    <div
      className={`condNode condNode-${node.type}`}
      style={{ marginLeft: depth === 0 ? 0 : 12 }}
    >
      <div className="condHeader">
        <select
          value={node.type}
          onChange={(e) =>
            update(node.id, (n) => morphNode(n, e.target.value as NodeKind))
          }
        >
          <optgroup label="Logic">
            <option value="all_of">all of (AND)</option>
            <option value="any_of">any of (OR)</option>
            <option value="none_of">none of (NOR)</option>
          </optgroup>
          <optgroup label="Structures">
            <option value="nearby_structure">nearby structure</option>
            <option value="cluster">cluster</option>
          </optgroup>
          <optgroup label="Biomes">
            <option value="spawn_biome">spawn biome</option>
            <option value="nearby_biomes">nearby biomes</option>
            <option value="biome_area">biome area</option>
          </optgroup>
        </select>
        {!isRoot && (
          <button className="condRemove" onClick={() => remove(node.id)} title="Remove">
            ✕
          </button>
        )}
      </div>
      <div className="condBody">{renderBody(node, update)}</div>
      {(node.type === "all_of" || node.type === "any_of" || node.type === "none_of") && (
        <div className="condChildren">
          {node.of.map((child) => (
            <NodeView key={child.id} node={child} depth={depth + 1} update={update} remove={remove} isRoot={false} />
          ))}
          <button
            className="condAddChild"
            onClick={() =>
              update(node.id, (n) =>
                n.type === "all_of" || n.type === "any_of" || n.type === "none_of"
                  ? { ...n, of: [...n.of, defaultLeaf("nearby_structure")] }
                  : n,
              )
            }
          >
            + Add child
          </button>
        </div>
      )}
    </div>
  );
}

function renderBody(
  node: TreeNode,
  update: (id: string, updater: (n: TreeNode) => TreeNode) => void,
): React.ReactNode {
  if (node.type === "all_of" || node.type === "any_of" || node.type === "none_of") return null;

  if (node.type === "nearby_structure") {
    return (
      <>
        <LabeledField label="Structure">
          <select
            value={node.structure}
            onChange={(e) => update(node.id, (n) => ({ ...(n as NearbyStructureNode), structure: e.target.value }))}
          >
            {STRUCTURES.map((s) => (
              <option key={s} value={s}>{s.replace(/_/g, " ")}</option>
            ))}
          </select>
        </LabeledField>
        <NumField label="max distance" value={node.max_distance} onChange={(v) => update(node.id, (n) => ({ ...(n as NearbyStructureNode), max_distance: v }))} />
      </>
    );
  }
  if (node.type === "cluster") {
    return (
      <>
        <LabeledField label="Structures (one per line)">
          <textarea
            rows={2}
            value={node.structures.join("\n")}
            onChange={(e) =>
              update(node.id, (n) => ({
                ...(n as ClusterNode),
                structures: e.target.value.split(/\s*\n\s*/).filter(Boolean),
              }))
            }
          />
        </LabeledField>
        <div className="condRow">
          <NumField label="min count" value={node.min_count} onChange={(v) => update(node.id, (n) => ({ ...(n as ClusterNode), min_count: v }))} />
          <NumField label="radius" value={node.max_distance} onChange={(v) => update(node.id, (n) => ({ ...(n as ClusterNode), max_distance: v }))} />
        </div>
      </>
    );
  }
  if (node.type === "spawn_biome") {
    return (
      <>
        <BiomePicker value={node.biomes} onChange={(b) => update(node.id, (n) => ({ ...(n as SpawnBiomeNode), biomes: b }))} />
        <NumField label="spawn radius" value={node.spawn_radius} onChange={(v) => update(node.id, (n) => ({ ...(n as SpawnBiomeNode), spawn_radius: v }))} />
      </>
    );
  }
  if (node.type === "nearby_biomes") {
    return (
      <>
        <BiomePicker value={node.biomes} onChange={(b) => update(node.id, (n) => ({ ...(n as NearbyBiomesNode), biomes: b }))} />
        <div className="condRow">
          <NumField label="radius" value={node.radius} onChange={(v) => update(node.id, (n) => ({ ...(n as NearbyBiomesNode), radius: v }))} />
          <NumField label="samples/axis" value={node.samples_per_axis} onChange={(v) => update(node.id, (n) => ({ ...(n as NearbyBiomesNode), samples_per_axis: v }))} />
        </div>
        <label className="condCheck">
          <input
            type="checkbox"
            checked={node.all}
            onChange={(e) => update(node.id, (n) => ({ ...(n as NearbyBiomesNode), all: e.target.checked }))}
          />
          require all (else any)
        </label>
      </>
    );
  }
  if (node.type === "biome_area") {
    return (
      <>
        <BiomePicker value={node.biomes} onChange={(b) => update(node.id, (n) => ({ ...(n as BiomeAreaNode), biomes: b }))} />
        <div className="condRow">
          <NumField label="radius" value={node.radius} onChange={(v) => update(node.id, (n) => ({ ...(n as BiomeAreaNode), radius: v }))} />
          <NumField label="samples/axis" value={node.samples_per_axis} onChange={(v) => update(node.id, (n) => ({ ...(n as BiomeAreaNode), samples_per_axis: v }))} />
          <NumField label="min samples" value={node.min_samples} onChange={(v) => update(node.id, (n) => ({ ...(n as BiomeAreaNode), min_samples: v }))} />
        </div>
      </>
    );
  }
  return null;
}

// ---------------------------------------------------------------------------
// Small primitive fields
// ---------------------------------------------------------------------------

function LabeledField({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="condField">
      <span>{label}</span>
      {children}
    </label>
  );
}

function NumField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <LabeledField label={label}>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </LabeledField>
  );
}

function BiomePicker({
  value,
  onChange,
}: {
  value: number[];
  onChange: (next: number[]) => void;
}) {
  const set = new Set(value);
  function toggle(id: number) {
    const next = new Set(set);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    onChange(Array.from(next).sort((a, b) => a - b));
  }
  function addGroup(groupKey: string) {
    const ids = BIOME_GROUPS[groupKey];
    if (!ids) return;
    const next = new Set(set);
    for (const id of ids) next.add(id);
    onChange(Array.from(next).sort((a, b) => a - b));
  }
  return (
    <div className="biomePicker">
      <div className="biomeChips">
        {value.length === 0 ? (
          <span className="biomeChipEmpty">no biomes selected</span>
        ) : (
          value.map((id) => (
            <button key={id} className="biomeChip" onClick={() => toggle(id)} title="Remove">
              {biomeLabel(id)} ✕
            </button>
          ))
        )}
      </div>
      <div className="condRow">
        <select
          defaultValue=""
          onChange={(e) => {
            if (e.target.value) toggle(Number(e.target.value));
            e.target.value = "";
          }}
        >
          <option value="">+ add biome…</option>
          {BIOMES.map((b) => (
            <option key={b.id} value={b.id}>{b.label}</option>
          ))}
        </select>
        <select
          defaultValue=""
          onChange={(e) => {
            if (e.target.value) addGroup(e.target.value);
            e.target.value = "";
          }}
        >
          <option value="">+ add group…</option>
          {Object.keys(BIOME_GROUPS).map((g) => (
            <option key={g} value={g}>{g}</option>
          ))}
        </select>
      </div>
    </div>
  );
}
