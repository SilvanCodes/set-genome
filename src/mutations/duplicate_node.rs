use rand::Rng;

use crate::{genes::Node, genome::Genome, MutationError};

use super::Mutations;

impl Mutations {
    /// This mutation adds a new node to the genome by "duplicating" an existing node, i.e. all incoming and outgoing connections to that node are duplicated as well.
    ///
    /// The weight of all outgoing connections is half the initial weight so the mutation starts out functionally equivalent to the genome without the mutation.
    /// By duplicating a node and its connections small local symetry can develop.
    /// It draws inspiration from the concept of gene duplication and cell division.
    pub fn duplicate_node(genome: &mut Genome, rng: &mut impl Rng) -> Result<(), MutationError> {
        // select an hiddden node gene to duplicate
        if let Some(mut random_hidden_node) = genome.hidden.random(rng).cloned() {
            let mut id = random_hidden_node.next_id();

            // avoid id collisions, will cause some kind of "divergent evolution" eventually
            while genome.contains(id) {
                id = random_hidden_node.next_id()
            }

            // construct new node gene
            let new_node = Node::new(id, random_hidden_node.activation);

            // duplicate outgoing feedforward connections
            let mut outgoing_feedforward_connections = genome
                .feed_forward
                .iter()
                .filter(|c| c.input == random_hidden_node.id)
                .cloned()
                .collect::<Vec<_>>();

            // duplicate incoming feedforward connections
            let incoming_feedforward_connections = genome
                .feed_forward
                .iter()
                .filter(|c| c.output == random_hidden_node.id)
                .cloned()
                .collect::<Vec<_>>();

            let mut new_feedworward_connections = Vec::with_capacity(
                outgoing_feedforward_connections.len() + incoming_feedforward_connections.len(),
            );

            // update weights
            for connection in outgoing_feedforward_connections.iter_mut() {
                connection.weight = connection.weight / 2.0;
                let mut new_connection = connection.clone();
                new_connection.input = new_node.id;
                new_feedworward_connections.push(new_connection);
            }

            // replace updated
            for connection in outgoing_feedforward_connections {
                assert!(genome.feed_forward.replace(connection).is_some())
            }

            // update ouputs
            for mut connection in incoming_feedforward_connections {
                connection.output = new_node.id;
                new_feedworward_connections.push(connection);
            }

            // insert all new connections
            for connection in new_feedworward_connections {
                assert!(genome.feed_forward.insert(connection))
            }

            // duplicate outgoing recurrent connections
            let mut outgoing_recurrent_connections = genome
                .recurrent
                .iter()
                .filter(|c| c.input == random_hidden_node.id && c.output != random_hidden_node.id)
                .cloned()
                .collect::<Vec<_>>();

            // duplicate incoming recurrent connections
            let incoming_recurrent_connections = genome
                .recurrent
                .iter()
                .filter(|c| c.output == random_hidden_node.id && c.input != random_hidden_node.id)
                .cloned()
                .collect::<Vec<_>>();

            let mut new_recurrent_connections = Vec::with_capacity(
                outgoing_recurrent_connections.len() + incoming_recurrent_connections.len(),
            );

            // update weights
            for connection in outgoing_recurrent_connections.iter_mut() {
                connection.weight = connection.weight / 2.0;
                let mut new_connection = connection.clone();
                new_connection.input = new_node.id;
                new_recurrent_connections.push(new_connection);
            }

            // replace updated
            for connection in outgoing_recurrent_connections {
                assert!(genome.recurrent.replace(connection).is_some())
            }

            // update ouputs
            for mut connection in incoming_recurrent_connections {
                connection.output = new_node.id;
                new_recurrent_connections.push(connection);
            }

            // insert all new connections
            for connection in new_recurrent_connections {
                assert!(genome.recurrent.insert(connection))
            }

            if let Some(self_loop) = genome
                .recurrent
                .iter()
                .find(|c| c.input == random_hidden_node.id && c.output == random_hidden_node.id)
            {
                let mut new_self_loop = self_loop.clone();
                new_self_loop.input = new_node.id;
                new_self_loop.output = new_node.id;
                assert!(genome.recurrent.insert(new_self_loop))
            }

            // replace selected node with updated id_counter
            assert!(genome.hidden.replace(random_hidden_node).is_some());

            // insert duplicated node
            assert!(genome.hidden.insert(new_node));
            Ok(())
        } else {
            Err(MutationError::CouldNotDuplicateNode)
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::{activations::Activation, Genome, Mutations, Parameters};

    #[test]
    fn duplicate_random_node() {
        let mut genome = Genome::initialized(&Parameters::default());
        assert_eq!(genome.feed_forward.len(), 1);

        Mutations::add_node(&Activation::all(), &mut genome, &mut thread_rng());
        assert_eq!(genome.hidden.len(), 1);
        assert_eq!(genome.feed_forward.len(), 3);

        // create all possible recurrent connections
        assert!(Mutations::add_recurrent_connection(&mut genome, &mut thread_rng()).is_ok());
        assert!(Mutations::add_recurrent_connection(&mut genome, &mut thread_rng()).is_ok());
        assert!(Mutations::add_recurrent_connection(&mut genome, &mut thread_rng()).is_ok());
        assert!(Mutations::add_recurrent_connection(&mut genome, &mut thread_rng()).is_ok());
        assert!(Mutations::add_recurrent_connection(&mut genome, &mut thread_rng()).is_ok());
        assert!(Mutations::add_recurrent_connection(&mut genome, &mut thread_rng()).is_ok());
        assert_eq!(genome.recurrent.len(), 6);

        assert!(Mutations::duplicate_node(&mut genome, &mut thread_rng()).is_ok());

        println!("{}", Genome::dot(&genome));

        assert_eq!(genome.feed_forward.len(), 5);
        assert_eq!(genome.recurrent.len(), 10);
        assert_eq!(genome.hidden.len(), 2);
    }

    #[test]
    fn same_structure_same_id() {
        let mut genome1 = Genome::initialized(&Parameters::default());
        let mut genome2 = Genome::initialized(&Parameters::default());

        Mutations::add_node(&Activation::all(), &mut genome1, &mut thread_rng());
        assert!(Mutations::duplicate_node(&mut genome1, &mut thread_rng()).is_ok());

        Mutations::add_node(&Activation::all(), &mut genome2, &mut thread_rng());
        assert!(Mutations::duplicate_node(&mut genome2, &mut thread_rng()).is_ok());

        assert_eq!(genome1.hidden, genome2.hidden);
    }
}
