use serde::{Deserialize, Serialize};

pub mod id_generator;
pub mod id_iter;

/// Identity of ANN structure elements.
#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Id(pub usize);
