use rand::Rng;

use crate::Genome;

use super::{MutationError, MutationResult, Mutations};

impl Mutations {
    /// Removes a recurrent connection if at least one is present in the genome.
    /// Does nothing when no recurrent connections exist.
    pub fn remove_recurrent_connection(genome: &mut Genome, rng: &mut impl Rng) -> MutationResult {
        if let Some(removable_connection) = &genome
            .recurrent
            .iter()
            // make iterator wrap
            .cycle()
            // randomly offset into the iterator to choose any node
            .skip((rng.gen::<f64>() * (genome.recurrent.len()) as f64).floor() as usize)
            .cloned()
            .next()
        {
            assert!(genome.recurrent.remove(removable_connection));
            Ok(())
        } else {
            Err(MutationError::CouldNotRemoveRecurrentConnection)
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::{
        activations::Activation,
        genes::{Connection, Genes, Id, Node},
        mutations::MutationError,
        Genome, Mutations,
    };

    #[test]
    fn can_remove_recurrent_connection() {
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
            recurrent: Genes(
                vec![Connection::new(Id(0), 1.0, Id(1))]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            ..Default::default()
        };

        assert!(Mutations::remove_recurrent_connection(&mut genome, &mut thread_rng()).is_ok())
    }

    #[test]
    fn can_not_remove_recurrent_connection() {
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

        if let Err(error) = Mutations::remove_recurrent_connection(&mut genome, &mut thread_rng()) {
            assert_eq!(error, MutationError::CouldNotRemoveRecurrentConnection);
        } else {
            unreachable!()
        }
    }
}
