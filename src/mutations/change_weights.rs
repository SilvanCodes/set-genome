use super::Mutations;
use crate::{genome::Genome, rng::GenomeRng};

impl Mutations {
    pub fn change_weights(percent_perturbed: f64, genome: &mut Genome, rng: &mut GenomeRng) {
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
                    let mut perturbation = rng.weight_perturbation();
                    if (connection.weight + perturbation) > rng.cap
                        || (connection.weight + perturbation) < -rng.cap
                    {
                        perturbation = -perturbation;
                    }
                    connection.weight += perturbation;
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
                    let mut perturbation = rng.weight_perturbation();
                    if (connection.weight + perturbation) > rng.cap
                        || (connection.weight + perturbation) < -rng.cap
                    {
                        perturbation = -perturbation;
                    }
                    connection.weight += perturbation;
                }
                connection
            })
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        mutations::Mutations,
        parameters::{Parameters, Structure},
        GenomeContext,
    };

    #[test]
    fn change_weights() {
        let mut gc = GenomeContext::default();

        let mut genome = gc.initialized_genome();

        let old_weight = genome.feed_forward.iter().next().unwrap().weight;

        genome.change_weights_with_context(&mut gc);

        assert!(
            (old_weight - genome.feed_forward.iter().next().unwrap().weight).abs() > f64::EPSILON
        );
    }

    #[test]
    fn respect_weight_cap() {
        let parameters = Parameters {
            seed: None,
            structure: Structure::default(),
            mutations: vec![Mutations::ChangeWeights {
                chance: 1.0,
                percent_perturbed: 1.0,
            }],
        };

        let mut gc = GenomeContext::new(parameters);

        let mut genome = gc.initialized_genome();

        for _ in 0..1000 {
            genome.change_weights_with_context(&mut gc);
            let weight = genome.feed_forward.iter().next().unwrap().weight;
            assert!(weight < gc.rng.cap && weight > -gc.rng.cap);
        }
    }
}
