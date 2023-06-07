use seahash::SeaHasher;
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};

use self::activations::Activation;

use super::{Gene, Id};

pub mod activations;

/// Struct describing a ANN node.
///
/// A node is made up of an identifier and activation function.
/// See [`Activations`] for more information.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Node {
    pub id: Id,
    pub order: usize,
    pub activation: Activation,
    pub id_counter: u64,
}

impl Node {
    pub fn input(id: Id, order: usize) -> Self {
        Node {
            id,
            order,
            activation: Activation::Linear,
            id_counter: 0,
        }
    }

    pub fn output(id: Id, order: usize, activation: Activation) -> Self {
        Node {
            id,
            order,
            activation,
            id_counter: 0,
        }
    }

    pub fn hidden(id: Id, activation: Activation) -> Self {
        Node {
            id,
            order: 0,
            activation,
            id_counter: 0,
        }
    }

    pub fn next_id(&mut self) -> Id {
        let mut id_hasher = SeaHasher::new();
        self.id.hash(&mut id_hasher);
        self.id_counter.hash(&mut id_hasher);
        self.id_counter += 1;
        Id(id_hasher.finish())
    }
}

impl Gene for Node {
    fn recombine(&self, other: &Self) -> Self {
        Self {
            activation: other.activation,
            ..*self
        }
    }
}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Node {}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.order.cmp(&other.order)
    }
}
