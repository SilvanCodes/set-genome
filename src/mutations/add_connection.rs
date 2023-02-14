use rand::Rng;

use crate::{genes::Connection, genome::Genome};

use super::{MutationError, MutationResult, Mutations};

impl Mutations {
    /// This mutation adds a new feed-forward connection to the genome, should it be possible.
    /// It is possible when any two nodes[^details] are not yet connected with a feed-forward connection.
    ///
    /// [^details]: "any two nodes" is technically not correct as the start node for the connection has to come from the intersection of input and hidden nodes and the end node has to come from the intersection of the hidden and output nodes.
    pub fn add_connection(genome: &mut Genome, rng: &mut impl Rng) -> MutationResult {
        // POTENTIAL BIAS: just chaining the iterators and starting "somewhere" in the iterator
        // seems like will at least in the long run heavily bias towards sampling hidden nodes.
        // This is because the amount of hidden nodes can grow while the number of inputs is fixed.
        // "starting somewhere" is ever more likely to hit a hidden node, which will then in expectation
        // be followed by (#hidden nodes / 2) more hidden nodes.
        // I should probably collect and shuffle for more of a fair distribution.
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
                    Connection::weight_perturbation(0.0, 1.0, rng),
                    end_node.id,
                )));
                return Ok(());
            }
        }
        // no possible connection end present
        Err(MutationError::CouldNotAddFeedForwardConnection)
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::{Genome, MutationError, Mutations, Parameters};

    #[test]
    fn add_random_connection() {
        let mut genome = Genome::uninitialized(&Parameters::default());

        assert!(Mutations::add_connection(&mut genome, &mut thread_rng()).is_ok());
        assert_eq!(genome.feed_forward.len(), 1);
    }

    #[test]
    fn dont_add_same_connection_twice() {
        let mut genome = Genome::uninitialized(&Parameters::default());

        Mutations::add_connection(&mut genome, &mut thread_rng()).expect("add_connection");

        if let Err(error) = Mutations::add_connection(&mut genome, &mut thread_rng()) {
            assert_eq!(error, MutationError::CouldNotAddFeedForwardConnection);
        } else {
            unreachable!()
        }

        assert_eq!(genome.feed_forward.len(), 1);
    }
}
