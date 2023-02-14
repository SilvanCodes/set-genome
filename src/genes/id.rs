use serde::{Deserialize, Serialize};

/// Identity of ANN structure elements.
#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Id(pub u64);
