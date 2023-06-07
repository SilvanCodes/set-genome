use super::Mutations;
use crate::genome::Genome;

impl Mutations {
    /// This mutation alters `percent_perturbed` connection weights sampled from a gaussian distribution with given `standard_deviation`.
    pub fn change_weights(percent_perturbed: f64, standard_deviation: f64, genome: &mut Genome) {
        let change_feed_forward_amount =
            (percent_perturbed * genome.feed_forward.len() as f64).ceil() as usize;
        let change_recurrent_amount =
            (percent_perturbed * genome.recurrent.len() as f64).ceil() as usize;

        genome.feed_forward = genome
            .feed_forward
            .drain_into_random(&genome.rng)
            .enumerate()
            .map(|(index, mut connection)| {
                if index < change_feed_forward_amount {
                    connection.perturb_weight(standard_deviation, &genome.rng);
                }
                connection
            })
            .collect();

        genome.recurrent = genome
            .recurrent
            .drain_into_random(&genome.rng)
            .enumerate()
            .map(|(index, mut connection)| {
                if index < change_recurrent_amount {
                    connection.perturb_weight(standard_deviation, &genome.rng);
                }
                connection
            })
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use crate::{Genome, Mutations, Parameters};

    #[test]
    fn change_weights() {
        let mut genome = Genome::initialized(&Parameters::default());

        let old_weight = genome.feed_forward.iter().next().unwrap().weight;

        Mutations::change_weights(1.0, 1.0, &mut genome);

        assert!(
            (old_weight - genome.feed_forward.iter().next().unwrap().weight).abs() > f64::EPSILON
        );
    }
}
