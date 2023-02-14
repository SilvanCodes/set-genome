use rand::Rng;

use super::Mutations;
use crate::genome::Genome;

impl Mutations {
    /// This mutation alters `percent_perturbed` connection weights with perturbations sampled from calls to [`GenomeRng::weight_perturbation`].
    pub fn change_weights(
        percent_perturbed: f64,
        weight_cap: f64,
        genome: &mut Genome,
        rng: &mut impl Rng,
    ) {
        let change_feed_forward_amount =
            (percent_perturbed * genome.feed_forward.len() as f64).ceil() as usize;
        let change_recurrent_amount =
            (percent_perturbed * genome.recurrent.len() as f64).ceil() as usize;

        genome.feed_forward = genome
            .feed_forward
            .drain_into_random(rng)
            .enumerate()
            .map(|(index, mut connection)| {
                if index < change_feed_forward_amount {
                    connection.perturb_weight(weight_cap, rng);
                }
                connection
            })
            .collect();

        genome.recurrent = genome
            .recurrent
            .drain_into_random(rng)
            .enumerate()
            .map(|(index, mut connection)| {
                if index < change_recurrent_amount {
                    connection.perturb_weight(weight_cap, rng);
                }
                connection
            })
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::{Genome, Mutations, Parameters};

    #[test]
    fn change_weights() {
        let mut genome = Genome::initialized(&Parameters::default());

        let old_weight = genome.feed_forward.iter().next().unwrap().weight;

        Mutations::change_weights(1.0, 1.0, &mut genome, &mut thread_rng());

        assert!(
            (old_weight - genome.feed_forward.iter().next().unwrap().weight).abs() > f64::EPSILON
        );
    }
}
