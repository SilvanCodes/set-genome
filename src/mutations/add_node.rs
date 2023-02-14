use rand::{prelude::SliceRandom, rngs::SmallRng};

use crate::{
    genes::{Activation, Connection, Node},
    genome::Genome,
};

use super::Mutations;

impl Mutations {
    /// This mutation adds a new node to the genome by "splitting" an existing connection, i.e. the existing connection gets "re-routed" via the new node and the weight of the split connection is set to zero.
    /// The connection leading into the new node is of weight 1.0 and the connection originating from the new node has the same weight as the split connection (before it is zeroed).
    pub fn add_node(activation_pool: &[Activation], genome: &mut Genome, rng: &mut SmallRng) {
        // select an connection gene and split
        let mut random_connection = genome.feed_forward.random(rng).cloned().unwrap();

        let id = random_connection.next_id();

        // construct new node gene
        let new_node = Node::new(id, activation_pool.choose(rng).cloned().unwrap());

        // insert new connection pointing to new node
        assert!(genome.feed_forward.insert(Connection::new(
            random_connection.input,
            1.0,
            new_node.id,
        )));
        // insert new connection pointing from new node
        assert!(genome.feed_forward.insert(Connection::new(
            new_node.id,
            random_connection.weight,
            random_connection.output,
        )));
        // insert new node into genome
        assert!(genome.hidden.insert(new_node));

        // update weight to zero to 'deactivate' connnection
        random_connection.weight = 0.0;
        genome.feed_forward.replace(random_connection);
    }
}

#[cfg(test)]
mod tests {
    use crate::GenomeContext;

    #[test]
    fn add_random_node() {
        let mut gc = GenomeContext::default();

        let mut genome = gc.initialized_genome();

        genome.add_node_with_context(&mut gc);

        assert_eq!(genome.feed_forward.len(), 3);
    }
}
