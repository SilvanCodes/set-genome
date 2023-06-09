use crate::Genome;

use super::{MutationError, MutationResult, Mutations};

impl Mutations {
    /// Removes a node and all incoming and outgoing connections, should this be possible without introducing dangling structure.
    /// Dangling means the in- or out-degree of any hidden node is zero, i.e. it neither can receive nor propagate a signal.
    /// If it is not possible, no node will be removed.
    pub fn remove_node(genome: &mut Genome) -> MutationResult {
        let mut hidden_nodes = genome.hidden.iter().collect::<Vec<_>>();

        genome.rng.shuffle(&mut hidden_nodes);

        if let Some(removable_node) = hidden_nodes
            .into_iter()
            .find(|removal_candidate| {
                genome
                    .connections()
                    // find all input nodes of removal candidate
                    .filter_map(|connection| {
                        if connection.output == removal_candidate.id {
                            Some(connection.input)
                        } else {
                            None
                        }
                    })
                    // make sure they have an alternative output
                    .all(|id| genome.has_alternative_output(id, removal_candidate.id))
                    && genome
                        .connections()
                        // find all output nodes of removal candidate
                        .filter_map(|connection| {
                            if connection.input == removal_candidate.id {
                                Some(connection.output)
                            } else {
                                None
                            }
                        })
                        // make sure they have an alternative input
                        .all(|id| genome.has_alternative_input(id, removal_candidate.id))
            })
            .cloned()
        {
            // remove all feed-forward connections involving the node to be removed
            genome.feed_forward.retain(|connection| {
                connection.input != removable_node.id && connection.output != removable_node.id
            });
            // remove all recurrent connections involving the node to be removed
            genome.recurrent.retain(|connection| {
                connection.input != removable_node.id && connection.output != removable_node.id
            });
            // remove the node to be removed
            assert!(genome.hidden.remove(&removable_node));
            Ok(())
        } else {
            Err(MutationError::CouldNotRemoveNode)
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
    fn can_remove_node() {
        let mut genome = Genome {
            inputs: Genes(vec![Node::input(Id(0), 0)].iter().cloned().collect()),
            hidden: Genes(
                vec![
                    Node::hidden(Id(2), Activation::Linear),
                    Node::hidden(Id(3), Activation::Linear),
                ]
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
                    Connection::new(Id(0), 1.0, Id(2)),
                    Connection::new(Id(0), 1.0, Id(3)),
                    Connection::new(Id(2), 1.0, Id(1)),
                    Connection::new(Id(3), 1.0, Id(1)),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            ..Default::default()
        };

        assert!(Mutations::remove_node(&mut genome).is_ok())
    }

    #[test]
    fn can_not_remove_node() {
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
                    Connection::new(Id(0), 1.0, Id(2)),
                    Connection::new(Id(2), 1.0, Id(1)),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            ..Default::default()
        };

        if let Err(error) = Mutations::remove_node(&mut genome) {
            assert_eq!(error, MutationError::CouldNotRemoveNode);
        } else {
            unreachable!()
        }
    }
}
