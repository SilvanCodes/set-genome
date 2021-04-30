use std::collections::HashSet;

use crate::{
    genes::{Activation, Connection, Genes, Id, IdGenerator, Node},
    parameters::Structure,
    rng::GenomeRng,
};

use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub inputs: Genes<Node>,
    pub hidden: Genes<Node>,
    pub outputs: Genes<Node>,
    pub feed_forward: Genes<Connection>,
    pub recurrent: Genes<Connection>,
}

impl Genome {
    pub fn new(id_gen: &mut IdGenerator, structure: &Structure) -> Self {
        Genome {
            inputs: (0..structure.inputs)
                .map(|_| Node::new(id_gen.next_id(), Activation::Linear))
                .collect(),
            outputs: (0..structure.outputs)
                .map(|_| Node::new(id_gen.next_id(), structure.outputs_activation))
                .collect(),
            ..Default::default()
        }
    }

    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.inputs
            .iter()
            .chain(self.hidden.iter())
            .chain(self.outputs.iter())
    }

    pub fn connections(&self) -> impl Iterator<Item = &Connection> {
        self.feed_forward.iter().chain(self.recurrent.iter())
    }

    pub fn init(&mut self, rng: &mut GenomeRng, structure: &Structure) {
        for input in self
            .inputs
            .iterate_with_random_offset(rng)
            .take((structure.inputs_connected_percent * structure.inputs as f64).ceil() as usize)
        {
            // connect to every output
            for output in self.outputs.iter() {
                assert!(self.feed_forward.insert(Connection::new(
                    input.id,
                    rng.weight_perturbation(),
                    output.id
                )));
            }
        }
    }

    pub fn len(&self) -> usize {
        self.feed_forward.len() + self.recurrent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.feed_forward.is_empty() && self.recurrent.is_empty()
    }

    pub fn cross_in(&self, other: &Self, rng: &mut impl Rng) -> Self {
        let feed_forward = self.feed_forward.cross_in(&other.feed_forward, rng);
        let recurrent = self.recurrent.cross_in(&other.recurrent, rng);
        let hidden = self.hidden.cross_in(&other.hidden, rng);

        Genome {
            feed_forward,
            recurrent,
            hidden,
            // use input and outputs from fitter, but they should be identical with weaker
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
        }
    }

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

    pub fn has_alternative_input(&self, node: Id, exclude: Id) -> bool {
        self.connections()
            .filter(|connection| connection.output == node)
            .any(|connection| connection.input != exclude)
    }

    pub fn has_alternative_output(&self, node: Id, exclude: Id) -> bool {
        self.connections()
            .filter(|connection| connection.input == node)
            .any(|connection| connection.output != exclude)
    }

    pub fn compatability_distance(
        genome_0: &Self,
        genome_1: &Self,
        factor_genes: f64,
        factor_weights: f64,
        factor_activations: f64,
        weight_cap: f64,
    ) -> (f64, f64, f64, f64) {
        let mut weight_difference_total = 0.0;
        let mut activation_difference = 0.0;

        let matching_genes_count_total = (genome_0
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

        let different_genes_count_total = (genome_0
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

        let maximum_weight_difference = matching_genes_count_total * 2.0 * weight_cap;

        // percent of different genes, considering all unique genes from both genomes
        let gene_diff = factor_genes * different_genes_count_total
            / (matching_genes_count_total + different_genes_count_total);

        // average weight differences , considering matching connection genes
        let weight_diff = factor_weights
            * if maximum_weight_difference > 0.0 {
                weight_difference_total / maximum_weight_difference
            } else {
                0.0
            };

        // percent of different activation functions, considering matching nodes genes
        let activation_diff = factor_activations
            * if matching_nodes_count > 0.0 {
                activation_difference / matching_nodes_count
            } else {
                0.0
            };

        (
            (gene_diff + weight_diff + activation_diff)
                / (factor_genes + factor_weights + factor_activations),
            gene_diff,
            weight_diff,
            activation_diff,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Genome;
    use crate::{
        genes::{Activation, Connection, Genes, Id, Node},
        GenomeContext,
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
        let mut gc = GenomeContext::default();

        let mut genome_0 = gc.initialized_genome();
        let mut genome_1 = gc.initialized_genome();

        // mutate genome_0
        genome_0.add_node_with_context(&mut gc);

        // mutate genome_1
        genome_1.add_node_with_context(&mut gc);
        genome_1.add_node_with_context(&mut gc);

        // shorter genome is fitter genome
        let offspring = genome_0.cross_in(&genome_1, &mut gc.rng);

        assert_eq!(offspring.hidden.len(), 1);
        assert_eq!(offspring.feed_forward.len(), 3);
    }

    #[test]
    fn detect_no_cycle() {
        let gc = GenomeContext::default();

        let genome = gc.initialized_genome();

        let input = genome.inputs.iter().next().unwrap();
        let output = genome.outputs.iter().next().unwrap();

        assert!(!genome.would_form_cycle(&input, &output));
    }

    #[test]
    fn detect_cycle() {
        let gc = GenomeContext::default();

        let genome = gc.initialized_genome();

        let input = genome.inputs.iter().next().unwrap();
        let output = genome.outputs.iter().next().unwrap();

        assert!(genome.would_form_cycle(&output, &input));
    }

    #[test]
    fn crossover_no_cycle() {
        let mut gc = GenomeContext::default();

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

        let offspring = genome_0.cross_in(&genome_1, &mut gc.rng);

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

        let delta =
            Genome::compatability_distance(&genome_0, &genome_1, 1.0, 0.4, 0.0, f64::INFINITY).0;

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
            .replace(Connection::new(Id(0), 2.0, Id(1)));

        let delta = Genome::compatability_distance(&genome_0, &genome_1, 0.0, 2.0, 0.0, 2.0).0;

        // factor 1 times 1 expressed difference over 4 possible difference over factor 1
        assert!((delta - 1.0 * 1.0 / 4.0 / 1.0).abs() < f64::EPSILON);
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

        let delta =
            Genome::compatability_distance(&genome_0, &genome_1, 2.0, 0.0, 0.0, f64::INFINITY).0;

        // factor 2 times 2 different genes over 3 total genes over factor 2
        assert!((delta - 2.0 * 2.0 / 3.0 / 2.0).abs() < f64::EPSILON);
    }
}
