//! Java Edition random-spread structure placement.

use crate::java_random::JavaRandom;

const PI: f64 = std::f64::consts::PI;
const REGION_MUL_X: i64 = 341_873_128_712;
const REGION_MUL_Z: i64 = 132_897_987_541;
const STRONGHOLD_RING_COUNTS: [usize; 8] = [3, 6, 10, 15, 21, 28, 36, 9];
const STRONGHOLD_RING_DISTANCES: [(i32, i32); 8] = [
    (1280, 2816),
    (4352, 5888),
    (7424, 8960),
    (10496, 12032),
    (13568, 15104),
    (16640, 18176),
    (19712, 21248),
    (22784, 24320),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpreadType {
    Linear,
    Triangular,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructureType {
    DesertPyramid,
    Igloo,
    JungleTemple,
    SwampHut,
    PillagerOutpost,
    Village,
    OceanRuin,
    Shipwreck,
    OceanMonument,
    WoodlandMansion,
    RuinedPortal,
    Stronghold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructureConfig {
    pub salt: i64,
    pub spacing: i32,
    pub separation: i32,
    pub spread_type: SpreadType,
}

impl StructureConfig {
    pub const fn chunk_range(self) -> i32 {
        self.spacing - self.separation
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructurePos {
    pub structure: StructureType,
    pub chunk_x: i32,
    pub chunk_z: i32,
}

impl StructurePos {
    pub const fn block_x(self) -> i32 {
        self.chunk_x * 16 + 8
    }

    pub const fn block_z(self) -> i32 {
        self.chunk_z * 16 + 8
    }
}

impl StructureType {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "desert_pyramid" => Some(Self::DesertPyramid),
            "igloo" => Some(Self::Igloo),
            "jungle_temple" => Some(Self::JungleTemple),
            "swamp_hut" => Some(Self::SwampHut),
            "pillager_outpost" => Some(Self::PillagerOutpost),
            "village" => Some(Self::Village),
            "ocean_ruin" => Some(Self::OceanRuin),
            "shipwreck" => Some(Self::Shipwreck),
            "ocean_monument" => Some(Self::OceanMonument),
            "woodland_mansion" => Some(Self::WoodlandMansion),
            "ruined_portal" => Some(Self::RuinedPortal),
            "stronghold" => Some(Self::Stronghold),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::DesertPyramid => "desert_pyramid",
            Self::Igloo => "igloo",
            Self::JungleTemple => "jungle_temple",
            Self::SwampHut => "swamp_hut",
            Self::PillagerOutpost => "pillager_outpost",
            Self::Village => "village",
            Self::OceanRuin => "ocean_ruin",
            Self::Shipwreck => "shipwreck",
            Self::OceanMonument => "ocean_monument",
            Self::WoodlandMansion => "woodland_mansion",
            Self::RuinedPortal => "ruined_portal",
            Self::Stronghold => "stronghold",
        }
    }
}

pub const fn structure_config(structure: StructureType) -> StructureConfig {
    match structure {
        StructureType::DesertPyramid => StructureConfig::linear(14_357_617, 32, 8),
        StructureType::Igloo => StructureConfig::linear(14_357_618, 32, 8),
        StructureType::JungleTemple => StructureConfig::linear(14_357_619, 32, 8),
        StructureType::SwampHut => StructureConfig::linear(14_357_620, 32, 8),
        StructureType::PillagerOutpost => StructureConfig::linear(165_745_296, 32, 8),
        StructureType::Village => StructureConfig::triangular(10_387_312, 34, 8),
        StructureType::OceanRuin => StructureConfig::linear(14_357_621, 20, 8),
        StructureType::Shipwreck => StructureConfig::linear(165_745_295, 24, 4),
        StructureType::OceanMonument => StructureConfig::triangular(10_387_313, 32, 5),
        StructureType::WoodlandMansion => StructureConfig::triangular(10_387_319, 80, 20),
        StructureType::RuinedPortal => StructureConfig::linear(34_222_645, 40, 15),
        StructureType::Stronghold => StructureConfig::linear(0, 1, 0),
    }
}

impl StructureConfig {
    const fn linear(salt: i64, spacing: i32, separation: i32) -> Self {
        Self {
            salt,
            spacing,
            separation,
            spread_type: SpreadType::Linear,
        }
    }

    const fn triangular(salt: i64, spacing: i32, separation: i32) -> Self {
        Self {
            salt,
            spacing,
            separation,
            spread_type: SpreadType::Triangular,
        }
    }
}

pub fn get_structure_pos(
    structure: StructureType,
    world_seed: i64,
    region_x: i64,
    region_z: i64,
) -> StructurePos {
    assert_ne!(
        structure,
        StructureType::Stronghold,
        "strongholds use iter_strongholds"
    );
    let cfg = structure_config(structure);
    let seed = world_seed
        .wrapping_add(cfg.salt)
        .wrapping_add(region_x.wrapping_mul(REGION_MUL_X))
        .wrapping_add(region_z.wrapping_mul(REGION_MUL_Z));
    let mut rng = JavaRandom::new(seed);
    let range = cfg.chunk_range();

    let (offset_x, offset_z) = match cfg.spread_type {
        SpreadType::Linear => (rng.next_int_bound(range), rng.next_int_bound(range)),
        SpreadType::Triangular => (
            (rng.next_int_bound(range) + rng.next_int_bound(range)) / 2,
            (rng.next_int_bound(range) + rng.next_int_bound(range)) / 2,
        ),
    };

    StructurePos {
        structure,
        chunk_x: (region_x as i32) * cfg.spacing + offset_x,
        chunk_z: (region_z as i32) * cfg.spacing + offset_z,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructureRequirement {
    pub structure: StructureType,
    pub max_distance: i32,
    pub centre_x: i32,
    pub centre_z: i32,
}

pub fn seed_matches_structure_requirements(
    seed: i64,
    requirements: &[StructureRequirement],
) -> bool {
    requirements
        .iter()
        .all(|req| has_structure_in_radius(seed, req))
}

pub fn has_structure_in_radius(seed: i64, req: &StructureRequirement) -> bool {
    if req.structure == StructureType::Stronghold {
        return iter_strongholds(seed, 1).any(|pos| {
            let dx = (pos.block_x() - req.centre_x) as i64;
            let dz = (pos.block_z() - req.centre_z) as i64;
            let max_dist_sq = (req.max_distance as i64) * (req.max_distance as i64);
            dx * dx + dz * dz <= max_dist_sq
        });
    }

    let cfg = structure_config(req.structure);
    let chunk_radius = req.max_distance / 16 + 1;
    let cx_min = req.centre_x.div_euclid(16) - chunk_radius;
    let cx_max = req.centre_x.div_euclid(16) + chunk_radius;
    let cz_min = req.centre_z.div_euclid(16) - chunk_radius;
    let cz_max = req.centre_z.div_euclid(16) + chunk_radius;
    let rx_min = floor_div(cx_min, cfg.spacing);
    let rx_max = floor_div(cx_max, cfg.spacing);
    let rz_min = floor_div(cz_min, cfg.spacing);
    let rz_max = floor_div(cz_max, cfg.spacing);
    let max_dist_sq = (req.max_distance as i64) * (req.max_distance as i64);

    for rx in rx_min..=rx_max {
        for rz in rz_min..=rz_max {
            let pos = get_structure_pos(req.structure, seed, rx as i64, rz as i64);
            let dx = (pos.block_x() - req.centre_x) as i64;
            let dz = (pos.block_z() - req.centre_z) as i64;
            if dx * dx + dz * dz <= max_dist_sq {
                return true;
            }
        }
    }
    false
}

fn floor_div(a: i32, b: i32) -> i32 {
    a.div_euclid(b)
}

pub fn iter_strongholds(seed: i64, max_rings: usize) -> impl Iterator<Item = StructurePos> {
    assert!(
        (1..=8).contains(&max_rings),
        "max_rings must be between 1 and 8"
    );

    let mut rng = JavaRandom::new(seed);
    let mut angle = rng.next_double() * PI * 2.0;
    let mut positions = Vec::new();

    for ring_idx in 0..max_rings {
        let count = STRONGHOLD_RING_COUNTS[ring_idx];
        let (d_min, d_max) = STRONGHOLD_RING_DISTANCES[ring_idx];
        for _ in 0..count {
            let distance = d_min as f64 + rng.next_double() * f64::from(d_max - d_min);
            let block_x = (angle.cos() * distance) as i32;
            let block_z = (angle.sin() * distance) as i32;
            positions.push(StructurePos {
                structure: StructureType::Stronghold,
                chunk_x: floor_div(block_x, 16),
                chunk_z: floor_div(block_z, 16),
            });
            angle += (2.0 * PI) / count as f64;
        }
    }

    positions.into_iter()
}

#[cfg(test)]
mod tests {
    use super::{get_structure_pos, StructureType};

    #[test]
    fn known_first_region_positions_seed_1() {
        let cases = [
            (StructureType::Village, (11, 16)),
            (StructureType::SwampHut, (9, 2)),
            (StructureType::DesertPyramid, (14, 10)),
            (StructureType::Shipwreck, (4, 8)),
            (StructureType::OceanMonument, (12, 23)),
            (StructureType::PillagerOutpost, (5, 20)),
            (StructureType::Igloo, (0, 10)),
            (StructureType::JungleTemple, (0, 18)),
        ];

        for (structure, expected) in cases {
            let pos = get_structure_pos(structure, 1, 0, 0);
            assert_eq!((pos.chunk_x, pos.chunk_z), expected, "{structure:?}");
        }
    }

    #[test]
    fn strongholds_count_and_first_ring_seed_1() {
        let first_ring: Vec<_> = super::iter_strongholds(1, 1).collect();
        let all: Vec<_> = super::iter_strongholds(1, 8).collect();
        assert_eq!(first_ring.len(), 3);
        assert_eq!(all.len(), 128);
    }
}
