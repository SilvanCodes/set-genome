use rand::Rng;

use super::Mutations;
use crate::genome::Genome;

impl Mutations {
    /// This mutation alters `percent_perturbed` connection weights sampled from a gaussian distribution with given `standard_deviation`.
    pub fn change_weights(
        mutation_rate: f64,
        duplication_rate: f64,
        genome: &mut Genome,
        rng: &mut impl Rng,
    ) {
        genome.feed_forward = genome
            .feed_forward
            .drain()
            .map(|mut connection| {
                connection.mutate_resolution(duplication_rate, rng);
                connection.mutate_weight(mutation_rate, rng);
                connection
            })
            .collect();

        genome.recurrent = genome
            .recurrent
            .drain()
            .map(|mut connection| {
                connection.mutate_resolution(duplication_rate, rng);
                connection.mutate_weight(mutation_rate, rng);
                connection
            })
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::{Genome, Mutations, Parameters};

    // #[test]
    // fn change_weights() {
    //     let mut genome = Genome::initialized(&Parameters::default());

    //     let old_weight = genome.feed_forward.iter().next().unwrap().weight();

    //     Mutations::change_weights(1.0, &mut genome, &mut thread_rng());

    //     assert!(
    //         (old_weight - genome.feed_forward.iter().next().unwrap().weight()).abs() > f64::EPSILON
    //     );
    // }
}
