use rand::Rng;

use crate::{genes::Connection, genome::Genome, rng::GenomeRng};

use super::Mutations;

impl Mutations {
    /// This mutation adds a new feed-forward connection to the genome, should it be possible.
    /// It is possible when [^any two nodes] are not yet connected with a recurrent connection.
    ///
    /// [^any two nodes]: "any two nodes" is technically not correct as the start node for the connection has to come from the intersection of input and hidden nodes and the end node has to come from the intersection of the hidden and output nodes.
    pub fn add_connection(genome: &mut Genome, rng: &mut GenomeRng) -> Result<(), &'static str> {
        let start_node_iterator = genome.inputs.iter().chain(genome.hidden.iter());
        let end_node_iterator = genome.hidden.iter().chain(genome.outputs.iter());

        for start_node in start_node_iterator
            // make iterator wrap
            .cycle()
            // randomly offset into the iterator to choose any node
            .skip(
                (rng.gen::<f64>() * (genome.inputs.len() + genome.hidden.len()) as f64).floor()
                    as usize,
            )
            // just loop every value once
            .take(genome.inputs.len() + genome.hidden.len())
        {
            if let Some(end_node) = end_node_iterator.clone().find(|&end_node| {
                end_node != start_node
                    && !genome.feed_forward.contains(&Connection::new(
                        start_node.id,
                        0.0,
                        end_node.id,
                    ))
                    && !genome.would_form_cycle(start_node, end_node)
            }) {
                // add new feed-forward connection
                assert!(genome.feed_forward.insert(Connection::new(
                    start_node.id,
                    rng.weight_perturbation(),
                    end_node.id,
                )));
                return Ok(());
            }
            // no possible connection end present
        }
        Err("no connection possible")
    }
}

#[cfg(test)]
mod tests {
    use crate::GenomeContext;

    #[test]
    fn add_random_connection() {
        let mut gc = GenomeContext::default();

        let mut genome = gc.uninitialized_genome();

        assert!(genome.add_connection_with_context(&mut gc).is_ok());

        assert_eq!(genome.feed_forward.len(), 1);
    }

    #[test]
    fn dont_add_same_connection_twice() {
        let mut gc = GenomeContext::default();

        let mut genome = gc.uninitialized_genome();

        assert!(genome.add_connection_with_context(&mut gc).is_ok());

        if let Err(message) = genome.add_connection_with_context(&mut gc) {
            assert_eq!(message, "no connection possible");
        } else {
            unreachable!()
        }

        assert_eq!(genome.feed_forward.len(), 1);
    }
}
