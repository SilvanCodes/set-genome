use crate::{genes::Activation, mutations::Mutations};
use config::{Config, ConfigError, File};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Parameters {
    pub seed: Option<u64>,
    pub structure: Structure,
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

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Structure {
    pub inputs: usize,
    pub inputs_connected_percent: f64,
    pub outputs: usize,
    pub outputs_activation: Activation,
    pub weight_std_dev: f64,
}

impl Default for Structure {
    fn default() -> Self {
        Self {
            inputs: 1,
            inputs_connected_percent: 1.0,
            outputs: 1,
            outputs_activation: Activation::Tanh,
            weight_std_dev: 0.1,
        }
    }
}

/* #[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Mutations {
    pub chance_node: f64,
    pub chance_connection: f64,
    pub chance_connection_recurrent: f64,
    pub chance_activation_change: f64,
    pub chance_weight_perturbation: f64,
    pub weight_perturbation_percent: f64,
    pub weight_perturbation_std_dev: f64,
    pub weight_perturbation_cap: f64,
}

impl Default for Mutations {
    fn default() -> Self {
        Self {
            chance_node: 0.005,
            chance_connection: 0.1,
            chance_connection_recurrent: 0.1,
            chance_activation_change: 0.05,
            chance_weight_perturbation: 1.0,
            weight_perturbation_percent: 0.5,
            weight_perturbation_std_dev: 0.1,
            weight_perturbation_cap: 1.0,
        }
    }
} */

impl Parameters {
    pub fn new(path: &str) -> Result<Self, ConfigError> {
        let mut s = Config::new();

        // Start off by merging in the "default" configuration file
        s.merge(File::with_name(path))?;

        // You can deserialize (and thus freeze) the entire configuration as
        s.try_into()
    }
}
