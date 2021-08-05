use rand::Rng;

use crate::{Genome, Mutations};

impl Mutations {
    /// Removes a recurrent connection if at least one is present in the genome.
    /// Does nothing when no recurrent connections exist.
    pub fn remove_recurrent_connection(
        genome: &mut Genome,
        rng: &mut impl Rng,
    ) -> Result<(), &'static str> {
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
            Err("no recurrent connection is present")
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        activations::Activation,
        genes::{Connection, Genes, Id, Node},
        Genome, GenomeContext,
    };

    #[test]
    fn can_remove_recurrent_connection() {
        let mut gc = GenomeContext::default();

        let mut genome = Genome {
            inputs: Genes(
                vec![Node::new(Id(0), Activation::Linear)]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            outputs: Genes(
                vec![Node::new(Id(1), Activation::Linear)]
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

        assert!(genome
            .remove_recurrent_connection_with_context(&mut gc)
            .is_ok())
    }

    #[test]
    fn can_not_remove_recurrent_connection() {
        let mut gc = GenomeContext::default();

        let mut genome = Genome {
            inputs: Genes(
                vec![Node::new(Id(0), Activation::Linear)]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            outputs: Genes(
                vec![Node::new(Id(1), Activation::Linear)]
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

        assert!(genome
            .remove_recurrent_connection_with_context(&mut gc)
            .is_err())
    }
}
