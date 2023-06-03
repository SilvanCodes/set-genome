use std::{
    collections::HashSet,
    hash::{Hash, Hasher},
};

use crate::{
    genes::{Activation, Connection, Genes, Id, Node},
    parameters::Structure,
};

use rand::{rngs::SmallRng, seq::SliceRandom, thread_rng, Rng, SeedableRng};
use seahash::SeaHasher;
use serde::{Deserialize, Serialize};

mod compatibility_distance;

pub use compatibility_distance::CompatibilityDistance;

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
        let rng = &mut SmallRng::from_rng(thread_rng()).unwrap();

        let mut possible_inputs = self.inputs.iter().collect::<Vec<_>>();
        possible_inputs.shuffle(rng);

        for input in possible_inputs.iter().take(
            (structure.percent_of_connected_inputs * structure.number_of_inputs as f64).ceil()
                as usize,
        ) {
            // connect to every output
            for output in self.outputs.iter() {
                assert!(self.feed_forward.insert(Connection::from_u64(
                    input.id,
                    rng.gen(),
                    output.id
                )));
            }
        }
    }

    /// Connects each output to a random input.
    ///
    /// This is the minimum required connectivity for the genome to be evaluatable.
    pub fn mimimum_init(&mut self) {
        let rng = &mut SmallRng::from_rng(thread_rng()).unwrap();

        for output in self.outputs.iter() {
            assert!(self.feed_forward.insert(Connection::from_u64(
                self.inputs.random(rng).unwrap().id,
                rng.gen(),
                output.id
            )));
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

    /// Get the encoded neural network as a string in [DOT][1] format.
    ///
    /// The output can be [visualized here][2] for example.
    ///
    /// [1]: https://www.graphviz.org/doc/info/lang.html
    /// [2]: https://dreampuf.github.io/GraphvizOnline
    pub fn dot(genome: &Self) -> String {
        let mut dot = "digraph {\n".to_owned();
        dot.push_str("\tgraph [splines=curved ranksep=8]\n");

        dot.push_str("\tsubgraph cluster_inputs {\n");
        dot.push_str("\t\tgraph [label=\"Inputs\"]\n");
        dot.push_str("\t\tnode [color=\"#D6B656\", fillcolor=\"#FFF2CC\", style=\"filled\"]\n");
        dot.push_str("\n");
        for node in genome.inputs.iter() {
            // fill color: FFF2CC
            // line color: D6B656

            dot.push_str(&format!(
                "\t\t{} [label={:?}];\n",
                node.id.0, node.activation
            ));
        }
        dot.push_str("\t}\n");

        dot.push_str("\tsubgraph hidden {\n");
        dot.push_str("\t\tgraph [label=\"Hidden\" rank=\"same\"]\n");
        dot.push_str("\t\tnode [color=\"#6C8EBF\", fillcolor=\"#DAE8FC\", style=\"filled\"]\n");
        dot.push_str("\n");
        for node in genome.hidden.iter() {
            // fill color: DAE8FC
            // line color: 6C8EBF

            dot.push_str(&format!(
                "\t\t{} [label={:?}];\n",
                node.id.0, node.activation
            ));
        }
        dot.push_str("\t}\n");

        dot.push_str("\tsubgraph cluster_outputs {\n");
        dot.push_str("\t\tgraph [label=\"Outputs\" labelloc=\"b\"]\n");
        dot.push_str("\t\tnode [color=\"#9673A6\", fillcolor=\"#E1D5E7\", style=\"filled\"]\n");
        dot.push_str("\n");
        for node in genome.outputs.iter() {
            // fill color: E1D5E7
            // line color: 9673A6

            dot.push_str(&format!(
                "\t\t{} [label={:?}];\n",
                node.id.0, node.activation
            ));
        }
        dot.push_str("\t}\n");

        dot.push_str("\n");

        dot.push_str("\tsubgraph feedforward_connections {\n");
        dot.push_str("\t\tedge [label=\"\"]\n");
        dot.push_str("\n");
        for connection in genome.feed_forward.iter() {
            dot.push_str(&format!(
                "\t\t{0} -> {1} [arrowsize={3:?} penwidth={3:?} tooltip={2:?} labeltooltip={2:?}];\n",
                connection.input.0,
                connection.output.0,
                connection.weight(),
                connection.weight().abs() * 0.95 + 0.05
            ));
        }
        dot.push_str("\t}\n");

        dot.push_str("\tsubgraph recurrent_connections {\n");
        dot.push_str("\t\tedge [label=\"\" color=\"#FF8000\"]\n");
        dot.push_str("\n");
        for connection in genome.recurrent.iter() {
            // color: FF8000

            dot.push_str(&format!(
                "\t\t{0} -> {1} [arrowsize={3:?} penwidth={3:?} tooltip={2:?} labeltooltip={2:?}];\n",
                connection.input.0,
                connection.output.0,
                connection.weight(),
                connection.weight().abs() * 0.95 + 0.05
            ));
        }
        dot.push_str("\t}\n");

        dot.push_str("}\n");
        dot
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
        Mutations, Parameters, Structure,
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

    #[test]
    fn create_dot_from_genome() {
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
            hidden: Genes(
                vec![Node::new(Id(2), Activation::Tanh)]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            feed_forward: Genes(
                vec![
                    Connection::new(Id(0), 0.25795942718883524, Id(2)),
                    Connection::new(Id(2), -0.09736946507786626, Id(1)),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            recurrent: Genes(
                vec![Connection::new(Id(1), 0.19777863112749228, Id(2))]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            ..Default::default()
        };

        // let dot = "digraph {\n\t0 [label=Linear color=\"#D6B656\" fillcolor=\"#FFF2CC\" style=\"filled\"];\n\t2 [label=Tanh color=\"#6C8EBF\" fillcolor=\"#DAE8FC\" style=\"filled\"];\n\t1 [label=Linear color=\"#9673A6\" fillcolor=\"#E1D5E7\" style=\"filled\"];\n\t0 -> 2 [label=0.25795942718883524];\n\t2 -> 1 [label=0.09736946507786626];\n\t1 -> 2 [label=0.19777863112749228 color=\"#FF8000\"];\n}\n";

        let dot = "digraph {
\tgraph [splines=curved ranksep=8]
\tsubgraph cluster_inputs {
\t\tgraph [label=\"Inputs\"]
\t\tnode [color=\"#D6B656\", fillcolor=\"#FFF2CC\", style=\"filled\"]

\t\t0 [label=Linear];
\t}
\tsubgraph hidden {
\t\tgraph [label=\"Hidden\" rank=\"same\"]
\t\tnode [color=\"#6C8EBF\", fillcolor=\"#DAE8FC\", style=\"filled\"]

\t\t2 [label=Tanh];
\t}
\tsubgraph cluster_outputs {
\t\tgraph [label=\"Outputs\" labelloc=\"b\"]
\t\tnode [color=\"#9673A6\", fillcolor=\"#E1D5E7\", style=\"filled\"]

\t\t1 [label=Linear];
\t}

\tsubgraph feedforward_connections {
\t\tedge [label=\"\"]

\t\t0 -> 2 [arrowsize=0.2875 penwidth=0.2875 tooltip=0.25 labeltooltip=0.25];
\t\t2 -> 1 [arrowsize=0.05 penwidth=0.05 tooltip=0.0 labeltooltip=0.0];
\t}
\tsubgraph recurrent_connections {
\t\tedge [label=\"\" color=\"#FF8000\"]

\t\t1 -> 2 [arrowsize=0.22812499999999997 penwidth=0.22812499999999997 tooltip=0.1875 labeltooltip=0.1875];
\t}
}
";
        assert_eq!(&Genome::dot(&genome), dot)
    }

    #[test]
    fn print_big_dot() {
        let parameters = Parameters {
            structure: Structure {
                number_of_inputs: 10,
                number_of_outputs: 10,
                percent_of_connected_inputs: 0.2,
                ..Default::default()
            },
            mutations: vec![
                Mutations::ChangeWeights {
                    mutation_rate: 0.1,
                    duplication_rate: 0.0,
                },
                Mutations::ChangeActivation {
                    chance: 0.05,
                    activation_pool: vec![
                        Activation::Linear,
                        Activation::Sigmoid,
                        Activation::Tanh,
                        Activation::Gaussian,
                        Activation::Step,
                        Activation::Sine,
                        Activation::Cosine,
                        Activation::Inverse,
                        Activation::Absolute,
                        Activation::Relu,
                    ],
                },
                Mutations::AddNode {
                    chance: 0.005,
                    activation_pool: vec![
                        Activation::Linear,
                        Activation::Sigmoid,
                        Activation::Tanh,
                        Activation::Gaussian,
                        Activation::Step,
                        Activation::Sine,
                        Activation::Cosine,
                        Activation::Inverse,
                        Activation::Absolute,
                        Activation::Relu,
                    ],
                },
                Mutations::AddConnection { chance: 0.01 },
                Mutations::AddRecurrentConnection { chance: 0.01 },
            ],
        };
        let mut genome = Genome::initialized(&parameters);

        for _ in 0..1000 {
            genome.mutate(&parameters);
        }

        print!("{}", Genome::dot(&genome));
    }
}
