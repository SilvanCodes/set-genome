use rand::{rngs::SmallRng, thread_rng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::{genes::Activation, genome::Genome};

pub use self::error::MutationError;

pub type MutationResult = Result<(), MutationError>;

mod add_connection;
mod add_node;
mod add_recurrent_connection;
mod change_activation;
mod change_weights;
mod error;
mod remove_connection;
mod remove_node;
mod remove_recurrent_connection;

/// Lists all possible mutations with their corresponding parameters.
///
/// Each mutation acts as a self-contained unit and has to be listed in the [`crate::Parameters::mutations`] field in order to take effect when calling [`crate::Genome::mutate_with_context`].
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Mutations {
    /// See [`Mutations::change_weights`].
    ChangeWeights {
        chance: f64,
        percent_perturbed: f64,
        weight_cap: f64,
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
}

impl Mutations {
    /// Mutate a [`Genome`] but respects the associate `chance` field of the [`Mutations`] enum variants.
    /// The user needs to supply the [`GenomeRng`] and [`IdGenerator`] manually when using this method directly.
    /// Use the [`crate::GenomeContext`] and the genomes `<>_with_context` functions to avoid manually handling those.
    pub fn mutate(&self, genome: &mut Genome) -> MutationResult {
        // Seed RNG from hash of genome?
        // Same RNG seed for same structure.
        // Connection weights are not used to calculate hash.
        // Node activations are not used to calculate hash.
        //
        // When and why would I want the RNG to spit out identical values?

        let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

        match self {
            &Mutations::ChangeWeights {
                chance,
                percent_perturbed,
                weight_cap,
            } => {
                if rng.gen::<f64>() < chance {
                    Self::change_weights(percent_perturbed, weight_cap, genome, &mut rng);
                }
            }
            Mutations::AddNode {
                chance,
                activation_pool,
            } => {
                if rng.gen::<f64>() < *chance {
                    Self::add_node(activation_pool, genome, &mut rng)
                }
            }
            &Mutations::AddConnection { chance } => {
                if rng.gen::<f64>() < chance {
                    return Self::add_connection(genome, &mut rng);
                }
            }
            &Mutations::AddRecurrentConnection { chance } => {
                if rng.gen::<f64>() < chance {
                    return Self::add_recurrent_connection(genome, &mut rng);
                }
            }
            Mutations::ChangeActivation {
                chance,
                activation_pool,
            } => {
                if rng.gen::<f64>() < *chance {
                    Self::change_activation(activation_pool, genome, &mut rng)
                }
            }
            &Mutations::RemoveNode { chance } => {
                if rng.gen::<f64>() < chance {
                    return Self::remove_node(genome, &mut rng);
                }
            }
            &Mutations::RemoveConnection { chance } => {
                if rng.gen::<f64>() < chance {
                    return Self::remove_connection(genome, &mut rng);
                }
            }
            &Mutations::RemoveRecurrentConnection { chance } => {
                if rng.gen::<f64>() < chance {
                    return Self::remove_recurrent_connection(genome, &mut rng);
                }
            }
        }
        Ok(())
    }
}
