use rand::{prelude::IteratorRandom, rngs::SmallRng};

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
        rng: &mut SmallRng,
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
    use crate::{Genome, Parameters};

    #[test]
    fn change_activation() {
        let parameters = Parameters::default();

        let mut genome = Genome::initialized(&parameters.structure);

        genome.add_node_with_context(&parameters);

        let old_activation = genome.hidden.iter().next().unwrap().activation;

        genome.change_activation_with_context(&parameters);

        assert_ne!(
            genome.hidden.iter().next().unwrap().activation,
            old_activation
        );
    }
}
