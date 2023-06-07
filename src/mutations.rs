use serde::{Deserialize, Serialize};

use crate::{genes::Activation, genome::Genome};

pub use self::error::MutationError;

pub type MutationResult = Result<(), MutationError>;

mod add_connection;
mod add_node;
mod add_recurrent_connection;
mod change_activation;
mod change_weights;
mod duplicate_node;
mod error;
mod remove_connection;
mod remove_node;
mod remove_recurrent_connection;

/// Lists all possible mutations with their corresponding parameters.
///
/// Each mutation acts as a self-contained unit and has to be listed in the [`crate::Parameters::mutations`] field in order to take effect when calling [`crate::Genome::mutate_with`].
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Mutations {
    /// See [`Mutations::change_weights`].
    ChangeWeights {
        chance: f64,
        percent_perturbed: f64,
        standard_deviation: f64,
    },
    /// See [`Mutations::change_activation`].
    ChangeActivation {
        chance: f64,
        activation_pool: Vec<Activation>,
    },
    /// See [`Mutations::add_node`].
    AddNode {
        chance: f64,
        activation_pool: Vec<Activation>,
    },
    /// See [`Mutations::add_connection`].
    AddConnection { chance: f64 },
    /// See [`Mutations::add_recurrent_connection`].
    AddRecurrentConnection { chance: f64 },
    /// See [`Mutations::remove_node`].
    RemoveNode { chance: f64 },
    /// See [`Mutations::remove_connection`].
    RemoveConnection { chance: f64 },
    /// See [`Mutations::remove_recurrent_connection`].
    RemoveRecurrentConnection { chance: f64 },
    /// See [`Mutations::duplicate_node`].
    DuplicateNode { chance: f64 },
}

impl Mutations {
    /// Mutate a [`Genome`] but respects the associate `chance` field of the [`Mutations`] enum variants.
    /// The user needs to supply some RNG manually when using this method directly.
    /// Use [`crate::Genome::mutate`] as the default API.
    pub fn mutate(&self, genome: &mut Genome) -> MutationResult {
        match self {
            &Mutations::ChangeWeights {
                chance,
                percent_perturbed,
                standard_deviation,
            } => {
                if genome.rng.f64() < chance {
                    Self::change_weights(percent_perturbed, standard_deviation, genome);
                }
            }
            Mutations::AddNode {
                chance,
                activation_pool,
            } => {
                if genome.rng.f64() < *chance {
                    Self::add_node(activation_pool, genome)
                }
            }
            &Mutations::AddConnection { chance } => {
                if genome.rng.f64() < chance {
                    return Self::add_connection(genome);
                }
            }
            &Mutations::AddRecurrentConnection { chance } => {
                if genome.rng.f64() < chance {
                    return Self::add_recurrent_connection(genome);
                }
            }
            Mutations::ChangeActivation {
                chance,
                activation_pool,
            } => {
                if genome.rng.f64() < *chance {
                    Self::change_activation(activation_pool, genome)
                }
            }
            &Mutations::RemoveNode { chance } => {
                if genome.rng.f64() < chance {
                    return Self::remove_node(genome);
                }
            }
            &Mutations::RemoveConnection { chance } => {
                if genome.rng.f64() < chance {
                    return Self::remove_connection(genome);
                }
            }
            &Mutations::RemoveRecurrentConnection { chance } => {
                if genome.rng.f64() < chance {
                    return Self::remove_recurrent_connection(genome);
                }
            }
            &Mutations::DuplicateNode { chance } => {
                if genome.rng.f64() < chance {
                    return Self::duplicate_node(genome);
                }
            }
        }
        Ok(())
    }
}
