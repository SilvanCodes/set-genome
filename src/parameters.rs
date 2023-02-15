use crate::{genes::Activation, mutations::Mutations};
use config::{Config, ConfigError, File};
use serde::{Deserialize, Serialize};

/// This struct captures configuration about the basic ANN structure and [available mutations].
///
/// It can be constructed manually or from a `.toml` file.
///
/// # Examples
///
/// ## Code
///
/// The following lists everything that is possible to specify:
/// ```
/// use set_genome::{Parameters, Structure, Mutations, activations::Activation};
///
/// let parameters = Parameters {
///     structure: Structure {
///         number_of_inputs: 25,
///         number_of_outputs: 3,
///         percent_of_connected_inputs: 1.0,
///         outputs_activation: Activation::Tanh,
///         seed: 42
///     },
///     mutations: vec![
///         Mutations::ChangeWeights {
///         chance: 1.0,
///         percent_perturbed: 0.5,
///         weight_cap: 1.0,
///         },
///         Mutations::ChangeActivation {
///             chance: 0.05,
///             activation_pool: vec![
///                 Activation::Linear,
///                 Activation::Sigmoid,
///                 Activation::Tanh,
///                 Activation::Gaussian,
///                 Activation::Step,
///                 Activation::Sine,
///                 Activation::Cosine,
///                 Activation::Inverse,
///                 Activation::Absolute,
///                 Activation::Relu,
///             ],
///         },
///         Mutations::AddNode {
///             chance: 0.005,
///             activation_pool: vec![
///                 Activation::Linear,
///                 Activation::Sigmoid,
///                 Activation::Tanh,
///                 Activation::Gaussian,
///                 Activation::Step,
///                 Activation::Sine,
///                 Activation::Cosine,
///                 Activation::Inverse,
///                 Activation::Absolute,
///                 Activation::Relu,
///             ],
///         },
///         Mutations::RemoveNode { chance: 0.001 },
///         Mutations::AddConnection { chance: 0.1 },
///         Mutations::RemoveConnection { chance: 0.001 },
///         Mutations::AddRecurrentConnection { chance: 0.01 },
///         Mutations::RemoveRecurrentConnection { chance: 0.001 },
///     ],
/// };
/// ```
///
/// ## Configuration
///
/// Write a config file like so:
/// ```toml
/// [structure]
/// number_of_inputs = 9
/// number_of_outputs = 2
/// percent_of_connected_inputs = 1.0
/// outputs_activation = "Tanh"
/// weight_std_dev = 0.1
/// weight_cap = 1.0
///
/// [[mutations]]
/// type = "add_connection"
/// chance = 0.1
///
/// [[mutations]]
/// type = "add_recurrent_connection"
/// chance = 0.01
///
/// [[mutations]]
/// type = "add_node"
/// chance = 0.005
/// activation_pool = [
///     "Sigmoid",
///     "Tanh",
///     "Relu",
///     "Linear",
///     "Gaussian",
///     "Step",
///     "Sine",
///     "Cosine",
///     "Inverse",
///     "Absolute",
/// ]
///
/// [[mutations]]
/// type = "remove_node"
/// chance = 0.001
///
/// [[mutations]]
/// type = "change_weights"
/// chance = 1.0
/// percent_perturbed = 0.5
///
/// [[mutations]]
/// type = "change_activation"
/// chance = 0.05
/// activation_pool = [
///     "Sigmoid",
///     "Tanh",
///     "Relu",
///     "Linear",
///     "Gaussian",
///     "Step",
///     "Sine",
///     "Cosine",
///     "Inverse",
///     "Absolute",
/// ]
///
/// [[mutations]]
/// type = "remove_connection"
/// chance = 0.001
///
/// [[mutations]]
/// type = "remove_recurrent_connection"
/// chance = 0.001
/// ```
///
/// And then read the file:
///
/// ```text
/// // let parameters = Parameters::new("path/to/file");
/// ```
///
/// [available mutations]: `Mutations`
///
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Parameters {
    /// Describes basic structure of the ANN.
    pub structure: Structure,
    /// List of mutations that execute on [`crate::Genome::mutate_with`]
    pub mutations: Vec<Mutations>,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            structure: Structure::default(),
            mutations: vec![
                Mutations::ChangeWeights {
                    chance: 1.0,
                    percent_perturbed: 0.5,
                    weight_cap: 1.0,
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
                Mutations::AddConnection { chance: 0.1 },
                Mutations::AddRecurrentConnection { chance: 0.01 },
            ],
        }
    }
}

impl Parameters {
    /// The basic parameters allow for the mutations of weights (100% of the time 50% of the weights are mutated), new nodes (1% chance) and new connections (10% chance) and are meant to quickly get a general feel for how this crate works.
    /// All nodes use the [`Activation::Tanh`] function.
    pub fn basic(number_of_inputs: usize, number_of_outputs: usize) -> Self {
        Self {
            structure: Structure::basic(number_of_inputs, number_of_outputs),
            mutations: vec![
                Mutations::ChangeWeights {
                    chance: 1.0,
                    percent_perturbed: 0.5,
                    weight_cap: 1.0,
                },
                Mutations::AddNode {
                    chance: 0.01,
                    activation_pool: vec![Activation::Tanh],
                },
                Mutations::AddConnection { chance: 0.1 },
            ],
        }
    }
}

/// This struct describes the invariants of the ANN structure.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Structure {
    /// Number of input nodes.
    pub number_of_inputs: usize,
    /// Number of output nodes.
    pub number_of_outputs: usize,
    /// Percent of input nodes initially connected to all poutput nodes.
    pub percent_of_connected_inputs: f64,
    /// Activation function for all output nodes.
    pub outputs_activation: Activation,
    /// Seed to generate the initial node ids.
    pub seed: u64,
}

impl Default for Structure {
    fn default() -> Self {
        Self {
            number_of_inputs: 1,
            number_of_outputs: 1,
            percent_of_connected_inputs: 1.0,
            outputs_activation: Activation::Tanh,
            seed: 42,
        }
    }
}

impl Structure {
    /// The basic structure connects every input to every output, uses a standard deviation of 0.1 for sampling weight mutations and caps weights between [-1, 1].
    pub fn basic(number_of_inputs: usize, number_of_outputs: usize) -> Self {
        Self {
            number_of_inputs,
            number_of_outputs,
            ..Default::default()
        }
    }
}

impl Parameters {
    pub fn new(path: &str) -> Result<Self, ConfigError> {
        let mut s = Config::new();

        // Start off by merging in the "default" configuration file
        s.merge(File::with_name(path))?;

        // You can deserialize (and thus freeze) the entire configuration as
        s.try_into()
    }
}
