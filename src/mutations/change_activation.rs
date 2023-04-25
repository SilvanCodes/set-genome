use rand::{prelude::IteratorRandom, Rng};

use crate::{
    genes::{Activation, Node},
    genome::Genome,
};

use super::Mutations;

impl Mutations {
    /// This mutation changes the activation function of one random hidden node to any other choosen from `activation_pool`.
    /// If the pool is empty (the current activation function is excluded) nothing is changed.
    pub fn change_activation(
        activation_pool: &[Activation],
        genome: &mut Genome,
        rng: &mut impl Rng,
    ) {
        if let Some(node) = genome.hidden.random(rng) {
            let updated = Node::new(
                node.id,
                activation_pool
                    .iter()
                    .filter(|&&activation| activation != node.activation)
                    .choose(rng)
                    .cloned()
                    .unwrap_or(node.activation),
            );

            genome.hidden.replace(updated);
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::{activations::Activation, Genome, Mutations, Parameters};

    #[test]
    fn change_activation() {
        let mut genome = Genome::initialized(&Parameters::default());
        let activation_pool = Activation::all();

        Mutations::add_node(&activation_pool, &mut genome, &mut thread_rng());

        let old_activation = genome.hidden.iter().next().unwrap().activation;

        Mutations::change_activation(&activation_pool, &mut genome, &mut thread_rng());

        assert_ne!(
            genome.hidden.iter().next().unwrap().activation,
            old_activation
        );
    }
}
