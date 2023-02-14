use rand::rngs::SmallRng;

use super::Mutations;
use crate::genome::Genome;

impl Mutations {
    /// This mutation alters `percent_perturbed` connection weights with perturbations sampled from calls to [`GenomeRng::weight_perturbation`].
    pub fn change_weights(
        percent_perturbed: f64,
        weight_cap: f64,
        genome: &mut Genome,
        rng: &mut SmallRng,
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
    use crate::{Genome, Parameters};

    #[test]
    fn change_weights() {
        let parameters = Parameters::default();

        let mut genome = Genome::initialized(&parameters.structure);

        let old_weight = genome.feed_forward.iter().next().unwrap().weight;

        genome.change_weights_with_context(&parameters);

        assert!(
            (old_weight - genome.feed_forward.iter().next().unwrap().weight).abs() > f64::EPSILON
        );
    }
}
