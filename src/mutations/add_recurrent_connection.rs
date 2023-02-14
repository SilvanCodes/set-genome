use rand::Rng;

use crate::{genes::Connection, genome::Genome};

use super::{MutationError, MutationResult, Mutations};

impl Mutations {
    /// This mutation adds a recurrent connection to the `genome` when possible.
    /// It is possible when any two nodes [^details] are not yet connected with a recurrent connection.
    ///
    /// [^details]: "any two nodes" is technically not correct as the end node has to come from the intersection of the hidden and output nodes.
    pub fn add_recurrent_connection(genome: &mut Genome, rng: &mut impl Rng) -> MutationResult {
        let start_node_iterator = genome
            .inputs
            .iter()
            .chain(genome.hidden.iter())
            .chain(genome.outputs.iter());
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
                    && !genome
                        .recurrent
                        .contains(&Connection::new(start_node.id, 0.0, end_node.id))
            }) {
                assert!(genome.recurrent.insert(Connection::new(
                    start_node.id,
                    Connection::weight_perturbation(0.0, 1.0, rng),
                    end_node.id,
                )));
                return Ok(());
            }
        }
        // no possible connection end present
        Err(MutationError::CouldNotAddRecurrentConnection)
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::{Genome, MutationError, Mutations, Parameters};

    #[test]
    fn add_random_connection() {
        let mut genome = Genome::initialized(&Parameters::default());

        Mutations::add_recurrent_connection(&mut genome, &mut thread_rng())
            .expect("y no add recurrent connection");

        assert_eq!(genome.recurrent.len(), 1);
    }

    #[test]
    fn dont_add_same_connection_twice() {
        let mut genome = Genome::initialized(&Parameters::default());

        Mutations::add_recurrent_connection(&mut genome, &mut thread_rng())
            .expect("y no add recurrent connection");

        if let Err(error) = Mutations::add_recurrent_connection(&mut genome, &mut thread_rng()) {
            assert_eq!(error, MutationError::CouldNotAddRecurrentConnection);
        } else {
            unreachable!()
        }

        assert_eq!(genome.recurrent.len(), 1);
    }
}
