use serde::{Deserialize, Serialize};

use crate::{
    genes::{Activation, IdGenerator},
    genome::Genome,
    rng::GenomeRng,
};

mod add_connection;
mod add_node;
mod add_recurrent_connection;
mod change_activation;
mod change_weights;
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
    ChangeWeights { chance: f64, percent_perturbed: f64 },
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
    pub fn mutate(
        &self,
        genome: &mut Genome,
        rng: &mut GenomeRng,
        id_gen: &mut IdGenerator,
    ) -> Result<(), &'static str> {
        match self {
            &Mutations::ChangeWeights {
                chance,
                percent_perturbed,
            } => {
                if rng.gamble(chance) {
                    Self::change_weights(percent_perturbed, genome, rng);
                }
            }
            Mutations::AddNode {
                chance,
                activation_pool,
            } => {
                if rng.gamble(*chance) {
                    Self::add_node(activation_pool, genome, rng, id_gen)
                }
            }
            &Mutations::AddConnection { chance } => {
                if rng.gamble(chance) {
                    return Self::add_connection(genome, rng);
                }
            }
            &Mutations::AddRecurrentConnection { chance } => {
                if rng.gamble(chance) {
                    return Self::add_recurrent_connection(genome, rng);
                }
            }
            Mutations::ChangeActivation {
                chance,
                activation_pool,
            } => {
                if rng.gamble(*chance) {
                    Self::change_activation(activation_pool, genome, rng)
                }
            }
            &Mutations::RemoveNode { chance } => {
                if rng.gamble(chance) {
                    return Self::remove_node(genome, rng);
                }
            }
            &Mutations::RemoveConnection { chance } => {
                if rng.gamble(chance) {
                    return Self::remove_connection(genome, rng);
                }
            }
            &Mutations::RemoveRecurrentConnection { chance } => {
                if rng.gamble(chance) {
                    return Self::remove_recurrent_connection(genome, rng);
                }
            }
        }
        Ok(())
    }
}
