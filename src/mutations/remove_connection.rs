use crate::Genome;

use super::{MutationError, MutationResult, Mutations};

impl Mutations {
    /// Removes a connection, should this be possible without introducing dangling structure.
    /// Dangling means the in- or out-degree of any hidden node is zero, i.e. it neither can receive nor propagate a signal.
    /// If it is not possible, no connection will be removed.
    pub fn remove_connection(genome: &mut Genome) -> MutationResult {
        let mut feedforward_connections = genome.feed_forward.iter().collect::<Vec<_>>();

        genome.rng.shuffle(&mut feedforward_connections);

        if let Some(removable_connection) = feedforward_connections
            .into_iter()
            .find(|removal_candidate| {
                genome.has_alternative_input(removal_candidate.output, removal_candidate.input)
                    && genome
                        .has_alternative_output(removal_candidate.input, removal_candidate.output)
            })
            .cloned()
        {
            assert!(genome.feed_forward.remove(&removable_connection));
            Ok(())
        } else {
            Err(MutationError::CouldNotRemoveFeedForwardConnection)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        activations::Activation,
        genes::{Connection, Genes, Id, Node},
        mutations::MutationError,
        Genome, Mutations,
    };

    #[test]
    fn can_remove_connection() {
        let mut genome = Genome {
            inputs: Genes(vec![Node::input(Id(0), 0)].iter().cloned().collect()),
            hidden: Genes(
                vec![Node::hidden(Id(2), Activation::Linear)]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            outputs: Genes(
                vec![Node::output(Id(1), 0, Activation::Linear)]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            feed_forward: Genes(
                vec![
                    Connection::new(Id(0), 1.0, Id(1)),
                    Connection::new(Id(0), 1.0, Id(2)),
                    Connection::new(Id(2), 1.0, Id(1)),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            ..Default::default()
        };

        assert!(Mutations::remove_connection(&mut genome).is_ok());
    }

    #[test]
    fn can_not_remove_connection() {
        let mut genome = Genome {
            inputs: Genes(vec![Node::input(Id(0), 0)].iter().cloned().collect()),
            outputs: Genes(
                vec![Node::output(Id(1), 0, Activation::Linear)]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            feed_forward: Genes(
                vec![Connection::new(Id(0), 1.0, Id(1))]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            ..Default::default()
        };

        if let Err(error) = Mutations::remove_connection(&mut genome) {
            assert_eq!(error, MutationError::CouldNotRemoveFeedForwardConnection);
        } else {
            unreachable!()
        }
    }
}
