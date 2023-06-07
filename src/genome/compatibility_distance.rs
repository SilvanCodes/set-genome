use crate::Genome;

/// Mechanism to compute distances between genomes.
///
/// Compatibility distance is a concept introduced in [NEAT] and defines a distance metric between genomes.
/// It can be useful for other evolutionary mechanisms such as speciation.
///
/// Three aspects amount to the resulting difference:
/// - the amount of identical a.k.a shared connections between the genomes
/// - the total weight difference between shared connections
/// - the number of different activations in identical nodes
///
/// Each aspect gives a normalized value between 0 and 1 and is then weighted by the corresponding factor.
/// The computed difference is the normalized combination of the weighted aspects.
///
/// For details read [here] part 2.5.1.
///
/// [NEAT]: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
/// [here]: https://silvan.codes/assets/pdf/SET-NEAT_Thesis.pdf
///
/// # Example
/// ```
/// # use set_genome::{Genome, Parameters, CompatibilityDistance};
/// let distance = CompatibilityDistance::with_factors(1.0, 1.0, 1.0);
///
/// let parameters = Parameters::basic(10, 10);
///
/// // Randomly initialize two genomes.
/// let genome_one = Genome::initialized(&parameters);
/// let genome_two = Genome::initialized(&parameters);
///
/// // Both calls are equivalent.
/// assert!(distance.between(&genome_one, &genome_two) > 0.0);
/// assert!(CompatibilityDistance::compatability_distance(&genome_one, &genome_two, 1.0, 1.0, 1.0).0 > 0.0);
/// ```
pub struct CompatibilityDistance {
    factor_connections: f64,
    factor_weights: f64,
    factor_activations: f64,
}

impl CompatibilityDistance {
    pub fn with_factors(
        factor_connections: f64,
        factor_weights: f64,
        factor_activations: f64,
    ) -> Self {
        Self {
            factor_connections,
            factor_weights,
            factor_activations,
        }
    }

    pub fn between(&self, genome_0: &Genome, genome_1: &Genome) -> f64 {
        let Self {
            factor_connections,
            factor_weights,
            factor_activations,
        } = *self;

        CompatibilityDistance::compatability_distance(
            genome_0,
            genome_1,
            factor_connections,
            factor_weights,
            factor_activations,
        )
        .0
    }

    /// Directly compute the compatability distance.
    ///
    /// The result is a 4-tuple of:
    /// - the overall difference
    /// - the scaled connection difference
    /// - the scaled weight difference
    /// - the scaled activation difference
    ///
    /// # Example
    /// ```
    /// # use set_genome::{Genome, Parameters, CompatibilityDistance};
    /// let parameters = Parameters::basic(10, 10);
    ///
    /// // Randomly initialize two genomes.
    /// let genome_one = Genome::initialized(&parameters);
    /// let genome_two = Genome::initialized(&parameters);
    ///
    /// assert!(CompatibilityDistance::compatability_distance(&genome_one, &genome_two, 1.0, 1.0, 1.0).0 > 0.0);
    ///
    /// assert_eq!(CompatibilityDistance::compatability_distance(&genome_one, &genome_one, 1.0, 1.0, 1.0).1, 0.0);
    /// assert_eq!(CompatibilityDistance::compatability_distance(&genome_one, &genome_one, 1.0, 1.0, 1.0).2, 0.0);
    /// assert_eq!(CompatibilityDistance::compatability_distance(&genome_one, &genome_one, 1.0, 1.0, 1.0).3, 0.0);
    /// ```
    pub fn compatability_distance(
        genome_0: &Genome,
        genome_1: &Genome,
        factor_connections: f64,
        factor_weights: f64,
        factor_activations: f64,
    ) -> (f64, f64, f64, f64) {
        let mut weight_difference = 0.0;
        let mut activation_difference = 0.0;

        let matching_connections_count = (genome_0
            .feed_forward
            .iterate_matching_genes(&genome_1.feed_forward)
            .inspect(|(connection_0, connection_1)| {
                weight_difference += (connection_0.weight - connection_1.weight).abs();
            })
            .count()
            + genome_0
                .recurrent
                .iterate_matching_genes(&genome_1.recurrent)
                .inspect(|(connection_0, connection_1)| {
                    weight_difference += (connection_0.weight - connection_1.weight).abs();
                })
                .count()) as f64;

        let different_connections_count = (genome_0
            .feed_forward
            .iterate_unique_genes(&genome_1.feed_forward)
            .count()
            + genome_0
                .recurrent
                .iterate_unique_genes(&genome_1.recurrent)
                .count()) as f64;

        let matching_nodes_count = genome_0
            .hidden
            .iterate_matching_genes(&genome_1.hidden)
            .inspect(|(node_0, node_1)| {
                if node_0.activation != node_1.activation {
                    activation_difference += 1.0;
                }
            })
            .count() as f64;

        // Connection weights are capped between 1.0 and -1.0. So the maximum difference per matching connection is 2.0.
        let maximum_weight_difference = matching_connections_count * 2.0;

        // percent of different genes, considering all unique genes from both genomes
        let scaled_connection_difference = factor_connections * different_connections_count
            / (matching_connections_count + different_connections_count);

        // average weight differences , considering matching connection genes
        let scaled_weight_difference = factor_weights
            * if maximum_weight_difference > 0.0 {
                weight_difference / maximum_weight_difference
            } else {
                0.0
            };

        // percent of different activation functions, considering matching nodes genes
        let scaled_activation_difference = factor_activations
            * if matching_nodes_count > 0.0 {
                activation_difference / matching_nodes_count
            } else {
                0.0
            };

        let overall_scaled_distance = (scaled_connection_difference
            + scaled_weight_difference
            + scaled_activation_difference)
            / (factor_connections + factor_weights + factor_activations);

        (
            overall_scaled_distance,
            scaled_connection_difference,
            scaled_weight_difference,
            scaled_activation_difference,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        activations::Activation, genes::Genes,
        genome::compatibility_distance::CompatibilityDistance, Connection, Genome, Id, Node,
    };

    #[test]
    fn compatability_distance_same_genome() {
        let genome_0 = Genome {
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

        let genome_1 = genome_0.clone();

        let delta =
            CompatibilityDistance::compatability_distance(&genome_0, &genome_1, 1.0, 0.4, 0.0).0;

        assert!(delta.abs() < f64::EPSILON);
    }

    #[test]
    fn compatability_distance_different_weight_genome() {
        let genome_0 = Genome {
            inputs: Genes(vec![Node::input(Id(0), 0)].iter().cloned().collect()),
            outputs: Genes(
                vec![Node::output(Id(1), 0, Activation::Linear)]
                    .iter()
                    .cloned()
                    .collect(),
            ),

            feed_forward: Genes(
                vec![Connection::new(Id(0), 0.0, Id(1))]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            ..Default::default()
        };

        let mut genome_1 = genome_0.clone();

        genome_1
            .feed_forward
            .replace(Connection::new(Id(0), 1.0, Id(1)));

        let factor_weight = 2.0;

        let delta = CompatibilityDistance::compatability_distance(
            &genome_0,
            &genome_1,
            0.0,
            factor_weight,
            0.0,
        )
        .0;

        // factor 2 times 2 expressed difference over 2 possible difference over factor 2
        assert!((delta - factor_weight * 1.0 / 2.0 / factor_weight).abs() < f64::EPSILON);
    }

    #[test]
    fn compatability_distance_different_connection_genome() {
        let genome_0 = Genome {
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

        let mut genome_1 = genome_0.clone();

        genome_1
            .feed_forward
            .insert(Connection::new(Id(0), 1.0, Id(2)));
        genome_1
            .feed_forward
            .insert(Connection::new(Id(2), 2.0, Id(1)));

        let delta =
            CompatibilityDistance::compatability_distance(&genome_0, &genome_1, 2.0, 0.0, 0.0).0;

        // factor 2 times 2 different genes over 3 total genes over factor 2
        assert!((delta - 2.0 * 2.0 / 3.0 / 2.0).abs() < f64::EPSILON);
    }
}
