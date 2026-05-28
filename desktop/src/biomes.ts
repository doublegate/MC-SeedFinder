// Subset of the project's biome catalog that the visual condition builder
// exposes. IDs mirror cubiomes' enum BiomeID (the values in src/mcseedfinder/
// biomes.py). Names are the readable forms shown in dropdowns.
//
// Kept small on purpose: it's the curated set the UI needs, not the full
// 100+ biome enum. To add a biome, look up its ID in biomes.py and copy a
// line here.

export type Biome = { id: number; key: string; label: string };

export const BIOMES: Biome[] = [
  // Oceans
  { id: 0, key: "ocean", label: "Ocean" },
  { id: 24, key: "deep_ocean", label: "Deep Ocean" },
  { id: 44, key: "warm_ocean", label: "Warm Ocean" },
  { id: 45, key: "lukewarm_ocean", label: "Lukewarm Ocean" },
  { id: 48, key: "deep_lukewarm_ocean", label: "Deep Lukewarm Ocean" },
  { id: 46, key: "cold_ocean", label: "Cold Ocean" },
  { id: 49, key: "deep_cold_ocean", label: "Deep Cold Ocean" },
  { id: 10, key: "frozen_ocean", label: "Frozen Ocean" },
  { id: 50, key: "deep_frozen_ocean", label: "Deep Frozen Ocean" },
  // Warm / dry
  { id: 2, key: "desert", label: "Desert" },
  { id: 35, key: "savanna", label: "Savanna" },
  { id: 36, key: "savanna_plateau", label: "Savanna Plateau" },
  { id: 37, key: "badlands", label: "Badlands" },
  { id: 165, key: "eroded_badlands", label: "Eroded Badlands" },
  { id: 38, key: "wooded_badlands", label: "Wooded Badlands" },
  // Warm / humid
  { id: 21, key: "jungle", label: "Jungle" },
  { id: 23, key: "sparse_jungle", label: "Sparse Jungle" },
  { id: 168, key: "bamboo_jungle", label: "Bamboo Jungle" },
  { id: 6, key: "swamp", label: "Swamp" },
  { id: 184, key: "mangrove_swamp", label: "Mangrove Swamp" },
  // Temperate
  { id: 1, key: "plains", label: "Plains" },
  { id: 129, key: "sunflower_plains", label: "Sunflower Plains" },
  { id: 4, key: "forest", label: "Forest" },
  { id: 132, key: "flower_forest", label: "Flower Forest" },
  { id: 27, key: "birch_forest", label: "Birch Forest" },
  { id: 29, key: "dark_forest", label: "Dark Forest" },
  { id: 192, key: "cherry_grove", label: "Cherry Grove" },
  { id: 186, key: "meadow", label: "Meadow" },
  // Cold
  { id: 5, key: "taiga", label: "Taiga" },
  { id: 32, key: "old_growth_pine_taiga", label: "Old Growth Pine Taiga" },
  { id: 3, key: "windswept_hills", label: "Windswept Hills" },
  { id: 34, key: "windswept_forest", label: "Windswept Forest" },
  { id: 189, key: "stony_peaks", label: "Stony Peaks" },
  // Snowy
  { id: 12, key: "snowy_plains", label: "Snowy Plains" },
  { id: 30, key: "snowy_taiga", label: "Snowy Taiga" },
  { id: 188, key: "snowy_slopes", label: "Snowy Slopes" },
  { id: 183, key: "frozen_peaks", label: "Frozen Peaks" },
  { id: 182, key: "jagged_peaks", label: "Jagged Peaks" },
  { id: 140, key: "ice_spikes", label: "Ice Spikes" },
  // Special
  { id: 14, key: "mushroom_fields", label: "Mushroom Fields" },
  { id: 16, key: "beach", label: "Beach" },
  { id: 7, key: "river", label: "River" },
];

// Quick-pick biome groups (mirror BIOME_GROUPS in src/mcseedfinder/biomes.py).
export const BIOME_GROUPS: Record<string, number[]> = {
  warm_dry: [2, 35, 36, 37, 165, 38],
  warm_humid: [21, 23, 168, 6, 184],
  temperate: [1, 129, 4, 132, 27, 29, 192, 186],
  cold: [5, 32, 3, 34, 189],
  snowy: [12, 30, 188, 183, 182, 140],
  ocean: [0, 24, 44, 45, 48, 46, 49, 10, 50],
  rare: [14, 140, 192, 165, 132],
};

export function biomeLabel(id: number): string {
  return BIOMES.find((b) => b.id === id)?.label ?? `#${id}`;
}
