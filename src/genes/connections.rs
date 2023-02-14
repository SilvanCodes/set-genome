use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::hash_map::DefaultHasher, hash::Hash, hash::Hasher};

use super::{Gene, Id};

/// Struct describing a ANN connection.
///
/// A connection is characterised by its input/origin/start, its output/destination/end and its weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub input: Id,
    pub output: Id,
    pub weight: f64,
    pub id_counter: u64,
}

impl Connection {
    pub fn new(input: Id, weight: f64, output: Id) -> Self {
        Self {
            input,
            output,
            weight,
            id_counter: 0,
        }
    }

    pub fn id(&self) -> (Id, Id) {
        (self.input, self.output)
    }

    pub fn next_id(&mut self) -> Id {
        let mut id_hasher = DefaultHasher::new();
        self.hash(&mut id_hasher);
        self.id_counter.hash(&mut id_hasher);
        self.id_counter += 1;
        Id(id_hasher.finish())
    }

    pub fn perturb_weight(&mut self, weight_cap: f64, rng: &mut impl Rng) {
        self.weight = Self::weight_perturbation(self.weight, weight_cap, rng);
    }

    pub fn weight_perturbation(weight: f64, weight_cap: f64, rng: &mut impl Rng) -> f64 {
        // approximatly normal distributed sample, see: https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution#Approximating_a_Normal_distribution
        let mut perturbation = (0..12).map(|_| rng.gen::<f64>()).sum::<f64>() - 6.0;

        while (weight + perturbation) > weight_cap || (weight + perturbation) < -weight_cap {
            perturbation = -perturbation / 2.0;
        }
        weight + perturbation
    }
}

impl Gene for Connection {}

impl PartialEq for Connection {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for Connection {}

impl Hash for Connection {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

impl PartialOrd for Connection {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Connection {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id().cmp(&other.id())
    }
}
