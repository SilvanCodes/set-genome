use crate::{genes::Activation, mutations::Mutations};
use config::{Config, ConfigError, File};
use serde::{Deserialize, Serialize};

/// This struct captures configuration about the basic ANN structure and available mutations.
///
/// It can be constructed manually or from a `.toml` file.
///
/// # Examples
///
/// Manual:
/// ```
/// use set_genome::{Parameters, Structure, Mutations};
///
/// let parameters = Parameters {
///     seed: None,
///     structure: Structure {
///         inputs: 25,
///         inputs_connected_percent: 1.0,
///         outputs: 3,
///         outputs_activation: Activation::Tanh,
///         weight_std_dev: 0.1,
///         weight_cap: 1.0,
///     },
///     mutations: vec![
///         Mutations::ChangeWeights {
///         chance: 1.0,
///         percent_perturbed: 0.5,
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
///         Mutations::AddRecurrentConnection { chance: 0.01 },
///     ],
/// }
/// ```
///
/// Write a config file like so:
/// ```toml
/// [structure]
/// inputs = 9
/// outputs = 2
/// inputs_connected_percent = 1.0
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
/// ```
/// And then read the file:
///
/// ```text
/// // let parameters = Parameters::new("path/to/file");
/// ```
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Parameters {
    /// Seed for the RNG.
    pub seed: Option<u64>,
    /// Describes basic structure of the ANN.
    pub structure: Structure,
    /// List of mutations that execute on [`crate::Genome::mutate_with_context`]
    pub mutations: Vec<Mutations>,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            seed: Some(42),
            structure: Structure::default(),
            mutations: vec![
                Mutations::ChangeWeights {
                    chance: 1.0,
                    percent_perturbed: 0.5,
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

/// This struct describes the invariants of the ANN structure.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Structure {
    /// Number of input nodes.
    pub inputs: usize,
    /// Percent of input nodes initially connected to all poutput nodes.
    pub inputs_connected_percent: f64,
    /// Number of output nodes.
    pub outputs: usize,
    /// Activation function for all output nodes.
    pub outputs_activation: Activation,
    /// Standard deviation of a normal distribution that provides samples for weight perturbations.
    pub weight_std_dev: f64,
    /// Constrains connection weights to the range [-weight_cap, weight_cap].
    pub weight_cap: f64,
}

impl Default for Structure {
    fn default() -> Self {
        Self {
            inputs: 1,
            inputs_connected_percent: 1.0,
            outputs: 1,
            outputs_activation: Activation::Tanh,
            weight_std_dev: 0.1,
            weight_cap: 1.0,
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
