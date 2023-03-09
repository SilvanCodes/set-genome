use std::{
    collections::HashSet,
    hash::{Hash, Hasher},
};

use crate::{
    genes::{Activation, Connection, Genes, Id, Node},
    parameters::Structure,
};

use rand::{rngs::SmallRng, thread_rng, Rng, SeedableRng};
use seahash::SeaHasher;
use serde::{Deserialize, Serialize};

/// This is the core data structure this crate revoles around.
///
/// A genome can be changed by mutation (a random alteration of its structure) or by crossing in another genome (recombining their matching parts).
/// A lot of additional information explaining details of the structure can be found in the [thesis] that developed this idea.
/// More and more knowledge from there will find its way into this documentaion over time.
///
/// [thesis]: https://www.silvan.codes/SET-NEAT_Thesis.pdf
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Genome {
    pub inputs: Genes<Node>,
    pub hidden: Genes<Node>,
    pub outputs: Genes<Node>,
    pub feed_forward: Genes<Connection>,
    pub recurrent: Genes<Connection>,
}

impl Genome {
    /// Creates a new genome according to the [`Structure`] it is given.
    /// It generates all necessary identities based on an RNG seeded from a hash of the I/O configuration of the structure.
    /// This allows genomes of identical I/O configuration to be crossed over in a meaningful way.
    pub fn new(structure: &Structure) -> Self {
        let mut seed_hasher = SeaHasher::new();
        structure.number_of_inputs.hash(&mut seed_hasher);
        structure.number_of_outputs.hash(&mut seed_hasher);
        structure.seed.hash(&mut seed_hasher);

        let mut rng = SmallRng::seed_from_u64(seed_hasher.finish());

        Genome {
            inputs: (0..structure.number_of_inputs)
                .map(|_| Node::new(Id(rng.gen::<u64>()), Activation::Linear))
                .collect(),
            outputs: (0..structure.number_of_outputs)
                .map(|_| Node::new(Id(rng.gen::<u64>()), structure.outputs_activation))
                .collect(),
            ..Default::default()
        }
    }

    /// Returns an iterator over references to all node genes (input + hidden + output) in the genome.
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.inputs
            .iter()
            .chain(self.hidden.iter())
            .chain(self.outputs.iter())
    }

    pub fn contains(&self, id: Id) -> bool {
        let fake_node = &Node::new(id, Activation::Linear);
        self.inputs.contains(fake_node)
            || self.hidden.contains(fake_node)
            || self.outputs.contains(fake_node)
    }

    /// Returns an iterator over references to all connection genes (feed-forward + recurrent) in the genome.
    pub fn connections(&self) -> impl Iterator<Item = &Connection> {
        self.feed_forward.iter().chain(self.recurrent.iter())
    }

    /// Initializes a genome, i.e. connects the in the [`Structure`] configured percent of inputs to all outputs by creating connection genes with random weights.
    pub fn init(&mut self, structure: &Structure) {
        let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

        for input in self.inputs.iterate_with_random_offset(&mut rng).take(
            (structure.percent_of_connected_inputs * structure.number_of_inputs as f64).ceil()
                as usize,
        ) {
            // connect to every output
            for output in self.outputs.iter() {
                assert!(self.feed_forward.insert(Connection::new(
                    input.id,
                    Connection::weight_perturbation(0.0, 0.1, &mut rng),
                    output.id
                )));
            }
        }
    }

    /// Returns the sum of connection genes inside the genome (feed-forward + recurrent).
    pub fn len(&self) -> usize {
        self.feed_forward.len() + self.recurrent.len()
    }

    /// Is true when no connection genes are present in the genome.
    pub fn is_empty(&self) -> bool {
        self.feed_forward.is_empty() && self.recurrent.is_empty()
    }

    /// Cross-in another genome.
    /// For connection genes present in both genomes flip a coin to determine the weight inside the new genome.
    /// For node genes present in both genomes flip a coin to determine the activation function inside the new genome.
    /// Any structure not present in other is taken over unchanged from `self`.
    pub fn cross_in(&self, other: &Self) -> Self {
        // Instantiating an RNG for every call might slow things down.
        let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

        let feed_forward = self.feed_forward.cross_in(&other.feed_forward, &mut rng);
        let recurrent = self.recurrent.cross_in(&other.recurrent, &mut rng);
        let hidden = self.hidden.cross_in(&other.hidden, &mut rng);

        Genome {
            feed_forward,
            recurrent,
            hidden,
            // use input and outputs from fitter, but they should be identical with weaker
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
        }
    }

    /// Check if connecting `start_node` and `end_node` would introduce a circle into the ANN structure.
    /// Think about the ANN as a graph for this, if you follow the connection arrows, can you reach `start_node` from `end_node`?
    pub fn would_form_cycle(&self, start_node: &Node, end_node: &Node) -> bool {
        let mut to_visit = vec![end_node.id];
        let mut visited = HashSet::new();

        while let Some(node) = to_visit.pop() {
            if !visited.contains(&node) {
                visited.insert(node);
                for connection in self
                    .feed_forward
                    .iter()
                    .filter(|connection| connection.input == node)
                {
                    if connection.output == start_node.id {
                        return true;
                    } else {
                        to_visit.push(connection.output)
                    }
                }
            }
        }
        false
    }

    /// Check if a node gene has more than one connection gene pointing to it.
    pub fn has_alternative_input(&self, node: Id, exclude: Id) -> bool {
        self.connections()
            .filter(|connection| connection.output == node)
            .any(|connection| connection.input != exclude)
    }

    /// Check if a node gene has more than one connection gene leaving it.
    pub fn has_alternative_output(&self, node: Id, exclude: Id) -> bool {
        self.connections()
            .filter(|connection| connection.input == node)
            .any(|connection| connection.output != exclude)
    }

    /// Defines a distance metric between genomes, useful for other evolutionary mechanisms such as speciation used in [NEAT].
    /// Expects three factors to tune to importance of several aspects contributing to the distance metric, for details read [here].
    ///
    /// [NEAT]: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
    /// [here]: https://www.silvan.codes/SET-NEAT_Thesis.pdf
    pub fn compatability_distance(
        genome_0: &Self,
        genome_1: &Self,
        factor_connections: f64,
        factor_weights: f64,
        factor_activations: f64,
    ) -> (f64, f64, f64, f64) {
        let mut weight_difference_total = 0.0;
        let mut activation_difference = 0.0;

        let matching_connections_count_total = (genome_0
            .feed_forward
            .iterate_matching_genes(&genome_1.feed_forward)
            .inspect(|(connection_0, connection_1)| {
                weight_difference_total += (connection_0.weight - connection_1.weight).abs();
            })
            .count()
            + genome_0
                .recurrent
                .iterate_matching_genes(&genome_1.recurrent)
                .inspect(|(connection_0, connection_1)| {
                    weight_difference_total += (connection_0.weight - connection_1.weight).abs();
                })
                .count()) as f64;

        let different_connections_count_total = (genome_0
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

        let maximum_weight_difference = matching_connections_count_total * 2.0;

        // percent of different genes, considering all unique genes from both genomes
        let scaled_connection_difference = factor_connections * different_connections_count_total
            / (matching_connections_count_total + different_connections_count_total);

        // average weight differences , considering matching connection genes
        let scaled_weight_difference = factor_weights
            * if maximum_weight_difference > 0.0 {
                weight_difference_total / maximum_weight_difference
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

        (
            (scaled_connection_difference
                + scaled_weight_difference
                + scaled_activation_difference)
                / (factor_connections + factor_weights + factor_activations),
            scaled_connection_difference,
            scaled_weight_difference,
            scaled_activation_difference,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::hash::{Hash, Hasher};

    use rand::thread_rng;
    use seahash::SeaHasher;

    use super::Genome;
    use crate::{
        genes::{Activation, Connection, Genes, Id, Node},
        Mutations, Parameters,
    };

    #[test]
    fn find_alternative_input() {
        let genome = Genome {
            inputs: Genes(
                vec![
                    Node::new(Id(0), Activation::Linear),
                    Node::new(Id(1), Activation::Linear),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            outputs: Genes(
                vec![Node::new(Id(2), Activation::Linear)]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            feed_forward: Genes(
                vec![
                    Connection::new(Id(0), 1.0, Id(2)),
                    Connection::new(Id(1), 1.0, Id(2)),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            ..Default::default()
        };

        assert!(genome.has_alternative_input(Id(2), Id(1)))
    }

    #[test]
    fn find_no_alternative_input() {
        let genome = Genome {
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

        assert!(!genome.has_alternative_input(Id(1), Id(0)))
    }

    #[test]
    fn find_alternative_output() {
        let genome = Genome {
            inputs: Genes(
                vec![Node::new(Id(0), Activation::Linear)]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            outputs: Genes(
                vec![
                    Node::new(Id(2), Activation::Linear),
                    Node::new(Id(1), Activation::Linear),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            feed_forward: Genes(
                vec![
                    Connection::new(Id(0), 1.0, Id(1)),
                    Connection::new(Id(0), 1.0, Id(2)),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            ..Default::default()
        };

        assert!(genome.has_alternative_output(Id(0), Id(1)))
    }

    #[test]
    fn find_no_alternative_output() {
        let genome = Genome {
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

        assert!(!genome.has_alternative_output(Id(0), Id(1)))
    }

    #[test]
    fn crossover() {
        let parameters = Parameters::default();

        let mut genome_0 = Genome::initialized(&parameters);
        let mut genome_1 = Genome::initialized(&parameters);

        let rng = &mut thread_rng();

        // mutate genome_0
        Mutations::add_node(&Activation::all(), &mut genome_0, rng);

        // mutate genome_1
        Mutations::add_node(&Activation::all(), &mut genome_1, rng);
        Mutations::add_node(&Activation::all(), &mut genome_1, rng);

        // shorter genome is fitter genome
        let offspring = genome_0.cross_in(&genome_1);

        assert_eq!(offspring.hidden.len(), 1);
        assert_eq!(offspring.feed_forward.len(), 3);
    }

    #[test]
    fn detect_no_cycle() {
        let parameters = Parameters::default();

        let genome = Genome::initialized(&parameters);

        let input = genome.inputs.iter().next().unwrap();
        let output = genome.outputs.iter().next().unwrap();

        assert!(!genome.would_form_cycle(&input, &output));
    }

    #[test]
    fn detect_cycle() {
        let parameters = Parameters::default();

        let genome = Genome::initialized(&parameters);

        let input = genome.inputs.iter().next().unwrap();
        let output = genome.outputs.iter().next().unwrap();

        assert!(genome.would_form_cycle(&output, &input));
    }

    #[test]
    fn crossover_no_cycle() {
        // assumption:
        // crossover of equal fitness genomes should not produce cycles
        // prerequisits:
        // genomes with equal fitness (0.0 in this case)
        // "mirrored" structure as simplest example

        let mut genome_0 = Genome {
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
            hidden: Genes(
                vec![
                    Node::new(Id(2), Activation::Tanh),
                    Node::new(Id(3), Activation::Tanh),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            feed_forward: Genes(
                vec![
                    Connection::new(Id(0), 1.0, Id(2)),
                    Connection::new(Id(2), 1.0, Id(1)),
                    Connection::new(Id(0), 1.0, Id(3)),
                    Connection::new(Id(3), 1.0, Id(1)),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            ..Default::default()
        };

        let mut genome_1 = genome_0.clone();

        // insert connectio one way in genome0
        genome_0
            .feed_forward
            .insert(Connection::new(Id(2), 1.0, Id(3)));

        // insert connection the other way in genome1
        genome_1
            .feed_forward
            .insert(Connection::new(Id(3), 1.0, Id(2)));

        let offspring = genome_0.cross_in(&genome_1);

        for connection0 in offspring.feed_forward.iter() {
            for connection1 in offspring.feed_forward.iter() {
                assert!(
                    !(connection0.input == connection1.output
                        && connection0.output == connection1.input)
                )
            }
        }
    }

    #[test]
    fn compatability_distance_same_genome() {
        let genome_0 = Genome {
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

        let genome_1 = genome_0.clone();

        let delta = Genome::compatability_distance(&genome_0, &genome_1, 1.0, 0.4, 0.0).0;

        assert!(delta.abs() < f64::EPSILON);
    }

    #[test]
    fn compatability_distance_different_weight_genome() {
        let genome_0 = Genome {
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

        let delta = Genome::compatability_distance(&genome_0, &genome_1, 0.0, factor_weight, 0.0).0;

        // factor 2 times 2 expressed difference over 2 possible difference over factor 2
        assert!((delta - factor_weight * 1.0 / 2.0 / factor_weight).abs() < f64::EPSILON);
    }

    #[test]
    fn compatability_distance_different_connection_genome() {
        let genome_0 = Genome {
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

        let mut genome_1 = genome_0.clone();

        genome_1
            .feed_forward
            .insert(Connection::new(Id(0), 1.0, Id(2)));
        genome_1
            .feed_forward
            .insert(Connection::new(Id(2), 2.0, Id(1)));

        let delta = Genome::compatability_distance(&genome_0, &genome_1, 2.0, 0.0, 0.0).0;

        // factor 2 times 2 different genes over 3 total genes over factor 2
        assert!((delta - 2.0 * 2.0 / 3.0 / 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn hash_genome() {
        let genome_0 = Genome {
            inputs: Genes(
                vec![
                    Node::new(Id(1), Activation::Linear),
                    Node::new(Id(0), Activation::Linear),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            outputs: Genes(
                vec![Node::new(Id(2), Activation::Linear)]
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

        let genome_1 = Genome {
            inputs: Genes(
                vec![
                    Node::new(Id(0), Activation::Linear),
                    Node::new(Id(1), Activation::Linear),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            outputs: Genes(
                vec![Node::new(Id(2), Activation::Linear)]
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

        assert_eq!(genome_0, genome_1);

        let mut hasher = SeaHasher::new();
        genome_0.hash(&mut hasher);
        let genome_0_hash = hasher.finish();

        let mut hasher = SeaHasher::new();
        genome_1.hash(&mut hasher);
        let genome_1_hash = hasher.finish();

        assert_eq!(genome_0_hash, genome_1_hash);
    }
}
