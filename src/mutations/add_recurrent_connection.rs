use crate::{genes::Connection, genome::Genome};

use super::{MutationError, MutationResult, Mutations};

impl Mutations {
    /// This mutation adds a recurrent connection to the `genome` when possible.
    /// It is possible when any two nodes [^details] are not yet connected with a recurrent connection.
    ///
    /// [^details]: "any two nodes" is technically not correct as the end node has to come from the intersection of the hidden and output nodes.
    pub fn add_recurrent_connection(genome: &mut Genome) -> MutationResult {
        let mut possible_start_nodes = genome
            .inputs
            .iter()
            .chain(genome.hidden.iter())
            .chain(genome.outputs.iter())
            .collect::<Vec<_>>();
        genome.rng.shuffle(&mut possible_start_nodes);

        let mut possible_end_nodes = genome
            .hidden
            .iter()
            .chain(genome.outputs.iter())
            .collect::<Vec<_>>();
        genome.rng.shuffle(&mut possible_end_nodes);

        for start_node in possible_start_nodes {
            if let Some(end_node) = possible_end_nodes.iter().cloned().find(|&end_node| {
                !genome
                    .recurrent
                    .contains(&Connection::new(start_node.id, 0.0, end_node.id))
            }) {
                assert!(genome.recurrent.insert(Connection::new(
                    start_node.id,
                    Connection::weight_perturbation(0.0, 0.1, &genome.rng),
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
    use crate::{Genome, MutationError, Mutations, Parameters};

    #[test]
    fn add_random_connection() {
        let mut genome = Genome::initialized(&Parameters::default());

        Mutations::add_recurrent_connection(&mut genome).expect("y no add recurrent connection");

        assert_eq!(genome.recurrent.len(), 1);
    }

    #[test]
    fn dont_add_same_connection_twice() {
        let mut genome = Genome::initialized(&Parameters::default());

        // create all possible recurrent connections
        Mutations::add_recurrent_connection(&mut genome).expect("y no add recurrent connection");

        Mutations::add_recurrent_connection(&mut genome).expect("y no add recurrent connection");

        if let Err(error) = Mutations::add_recurrent_connection(&mut genome) {
            assert_eq!(error, MutationError::CouldNotAddRecurrentConnection);
        } else {
            unreachable!()
        }

        assert_eq!(genome.recurrent.len(), 2);
    }
}
