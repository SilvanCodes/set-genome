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
mod remove_node;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Mutations {
    ChangeWeights {
        chance: f64,
        percent_perturbed: f64,
    },
    ChangeActivation {
        chance: f64,
        activation_pool: Vec<Activation>,
    },
    AddNode {
        chance: f64,
        activation_pool: Vec<Activation>,
    },
    AddConnection {
        chance: f64,
    },
    AddRecurrentConnection {
        chance: f64,
    },
    RemoveNode {
        chance: f64,
    },
}

impl Mutations {
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
        }
        Ok(())
    }
}
