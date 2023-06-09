use crate::{
    genes::{Activation, Node},
    genome::Genome,
};

use super::Mutations;

impl Mutations {
    /// This mutation changes the activation function of one random hidden node to any other choosen from `activation_pool`.
    /// If the pool is empty (the current activation function is excluded) nothing is changed.
    pub fn change_activation(activation_pool: &[Activation], genome: &mut Genome) {
        if let Some(node) = genome.hidden.random(&mut genome.rng) {
            let possible_activations = activation_pool
                .iter()
                .filter(|&&activation| activation != node.activation)
                .collect::<Vec<_>>();

            let updated = Node::hidden(
                node.id,
                genome
                    .rng
                    .choice(possible_activations)
                    .cloned()
                    .unwrap_or(node.activation),
            );

            genome.hidden.replace(updated);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{activations::Activation, Genome, Mutations, Parameters};

    #[test]
    fn change_activation() {
        let mut genome = Genome::initialized(&Parameters::default());
        let activation_pool = Activation::all();

        Mutations::add_node(&activation_pool, &mut genome);

        let old_activation = genome.hidden.iter().next().unwrap().activation;

        Mutations::change_activation(&activation_pool, &mut genome);

        assert_ne!(
            genome.hidden.iter().next().unwrap().activation,
            old_activation
        );
    }
}
