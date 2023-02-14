//! The `Gene` trait is a marker and in combination with the `Genes` struct describes common operations on collections (sets) of genes.
//!
//! The genome holds several fields with `Genes` of different types.

use rand::{prelude::IteratorRandom, prelude::SliceRandom, Rng};
use seahash::SeaHasher;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    hash::{BuildHasher, Hash, Hasher},
    iter::FromIterator,
    ops::Deref,
    ops::DerefMut,
};

mod connections;
mod id;
mod nodes;

pub use connections::Connection;
pub use id::Id;
pub use nodes::{
    activations::{self, Activation},
    Node,
};

pub trait Gene: Eq + Hash {}

impl<U: Gene, T: Eq + Hash + Deref<Target = U>> Gene for T {}

#[derive(Clone, Default)]
pub struct GeneHasher;

impl BuildHasher for GeneHasher {
    type Hasher = SeaHasher;

    fn build_hasher(&self) -> Self::Hasher {
        Self::Hasher::new()
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct Genes<T: Gene>(pub HashSet<T, GeneHasher>);

// see here: https://stackoverflow.com/questions/60882381/what-is-the-fastest-correct-way-to-detect-that-there-are-no-duplicates-in-a-json/60884343#60884343
impl<T: Gene> Hash for Genes<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let mut hash = 0;
        for gene in &self.0 {
            let mut gene_hasher = SeaHasher::new();
            gene.hash(&mut gene_hasher);
            hash ^= gene_hasher.finish();
        }
        state.write_u64(hash);
    }
}

impl<T: Gene> Default for Genes<T> {
    fn default() -> Self {
        Genes(Default::default())
    }
}

impl<T: Gene> Deref for Genes<T> {
    type Target = HashSet<T, GeneHasher>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Gene> DerefMut for Genes<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Gene> Genes<T> {
    pub fn iterate_with_random_offset(&self, rng: &mut impl Rng) -> impl Iterator<Item = &T> {
        self.iter()
            .cycle()
            .skip((rng.gen::<f64>() * self.len() as f64).floor() as usize)
            .take(self.len())
    }

    pub fn random(&self, rng: &mut impl Rng) -> Option<&T> {
        self.iter().choose(rng)
    }

    pub fn drain_into_random(&mut self, rng: &mut impl Rng) -> impl Iterator<Item = T> {
        let mut random_vec = self.drain().collect::<Vec<T>>();
        random_vec.shuffle(rng);
        random_vec.into_iter()
    }

    pub fn iterate_matching_genes<'a>(
        &'a self,
        other: &'a Genes<T>,
    ) -> impl Iterator<Item = (&'a T, &'a T)> {
        self.intersection(other)
            // we know item exists in other as we are iterating the intersection
            .map(move |item_self| (item_self, other.get(item_self).unwrap()))
    }

    pub fn iterate_unique_genes<'a>(&'a self, other: &'a Genes<T>) -> impl Iterator<Item = &'a T> {
        self.symmetric_difference(other)
    }
}

impl<T: Gene> FromIterator<T> for Genes<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Genes(iter.into_iter().collect())
    }
}

impl<T: Gene + Ord> Genes<T> {
    pub fn as_sorted_vec(&self) -> Vec<&T> {
        let mut vec: Vec<&T> = self.iter().collect();
        vec.sort_unstable();
        vec
    }
}

impl<T: Gene + Clone> Genes<T> {
    pub fn cross_in(&self, other: &Self, rng: &mut impl Rng) -> Self {
        self.iterate_matching_genes(other)
            .map(|(gene_self, gene_other)| {
                if rng.gen::<f64>() < 0.5 {
                    gene_self.clone()
                } else {
                    gene_other.clone()
                }
            })
            .chain(self.difference(other).cloned())
            .collect()
    }
}
