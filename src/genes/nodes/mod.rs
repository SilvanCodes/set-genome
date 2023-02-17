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
    pub activation: Activation,
}

impl Node {
    pub fn new(id: Id, activation: Activation) -> Self {
        Node { id, activation }
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
        self.id.cmp(&other.id)
    }
}
