//! This crate is supposed to act as the representation/reproduction aspect in neuroevolution algorithms and may be combined with arbitrary selection mechanisms.
//!
//! # What you can do with this crate
//! ```
//! # use set_genome::GenomeContext;
//! # use favannat::{
//! #   matrix::feedforward::fabricator::MatrixFeedForwardFabricator,
//! #    network::{Evaluator, Fabricator},
//! # };
//! # use nalgebra::dmatrix;
//! // Setup a genome context for networks with 10 inputs and 10 outputs.
//! let mut genome_context = GenomeContext::basic(10, 10);
//!
//! // Initialize a genome.
//! let mut genome = genome_context.initialized_genome();
//!
//! // Mutate a genome.
//! genome.mutate_with_context(&mut genome_context);
//!
//! // Get a phenotype of the genome.
//! let network = MatrixFeedForwardFabricator::fabricate(&genome).expect("Cool network.");
//!
//! // Evaluate a network on an input.
//! let output = network.evaluate(dmatrix![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
//! ```
//!
//! # Getting started
//!
//! Head over to [`GenomeContext`] to understand how to use this crate in detail.
//!
//! # SET genome
//!
//! SET stands for **S**et **E**ncoded **T**opology and this crate implements a genetic data structure, the [`Genome`], using this set encoding to describe artificial neural networks (ANNs).
//! Further this crate defines operations on this genome, namely [`Mutations`] and [crossover]. Mutations alter a genome by adding or removing genes, crossover recombines two genomes.
//! To have an intuitive definition of crossover for network structures the [NEAT algorithm] defined a procedure and has to be understood as a mental predecessor to this SET encoding,
//! which very much is a formalization and progression of the ideas NEAT introduced regarding the genome.
//! The thesis describing this genome and other ideas can be found [here], a paper focusing just on the SET encoding will follow soon.
//!
//! # Features
//!
//! This crate exposes the 'favannat' feature. [favannat] is a library to translate the genome into an executable form and also to execute it.
//! It can be seen as a phenotype of the genome.
//! The feature is enabled by default as probably you want to evaluate your evolved genomes, but disabling it is as easy as this:
//!
//! ```toml
//! [dependencies]
//! set-genome = { version = "x.x.x", default-features = false }
//! ```
//!
//! If you are interested how they connect, [see here].
//! favannat can be used to evaluate other data structures of yours, too, if they are [`favannat::network::NetworkLike`]. ;)
//!
//! [crossover]: `Genome::cross_in`
//! [NEAT algorithm]: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
//! [here]: https://www.silvan.codes/SET-NEAT_Thesis.pdf
//! [favannat]: https://docs.rs/favannat
//! [see here]: https://github.com/SilvanCodes/set-genome/blob/main/src/favannat_impl.rs

pub use genes::{activations, Connection, Id, Node};
pub use genome::Genome;
pub use mutations::{MutationError, MutationResult, Mutations};
pub use parameters::{Parameters, Structure};
use rand::{rngs::SmallRng, thread_rng, SeedableRng};

#[cfg(feature = "favannat")]
mod favannat_impl;
mod genes;
mod genome;
mod mutations;
mod parameters;

/// This struct simplifies operations on the [`Genome`].
///
/// The [`GenomeContext`] wraps all required building blocks to create and initialize genomes while maintaining consistent identities of their parts across operations.
/// It is used in a simplified API to perform operations on genomes, as it handles all necessary moving parts for you.
///
/// # Examples
///
/// Creating a default genome context:
/// ```
/// use set_genome::GenomeContext;
///
/// let genome_context = GenomeContext::default();
/// ```
///
/// Creating the context like this is unusual as we most likely want to pass it parameters fitting our situation.
///
/// Suppose we know our task has ten inputs and two outputs, which translate to the input and output layer of our ANN.
/// Further we want 100% of our inputs nodes to be initially connected to the outputs and the outputs shall use the [`activations::Activation::Tanh`] function.
/// Also the weights of our connections are supposed to be capped between \[-1, 1\] and change by deltas sampled from a normal distribution with 0.1 standard deviation.
///
/// ```
/// use set_genome::{GenomeContext, activations::Activation, Parameters, Structure};
///
/// let parameters = Parameters {
///     structure: Structure {
///         // ten inputs
///         number_of_inputs: 10,
///         // two outputs
///         number_of_outputs: 2,
///         // 100% connected
///         percent_of_connected_inputs: 1.0,
///         // specified output activation
///         outputs_activation: Activation::Tanh,
///         // seed for initial genome construction
///         seed: 42
///     },
///     mutations: vec![],
/// };
///
/// let genome_context = GenomeContext::new(parameters);
/// ```
/// This allows us to ask this context for an initialized genome which conforms to our description above:
///
/// ```
/// # use set_genome::{GenomeContext, activations::Activation, Parameters, Structure};
/// #
/// # let parameters = Parameters {
/// #     structure: Structure {
/// #         // ten inputs
/// #         number_of_inputs: 10,
/// #         // two outputs
/// #         number_of_outputs: 2,
/// #         // 100% connected
/// #         percent_of_connected_inputs: 1.0,
/// #         // specified output activation
/// #         outputs_activation: Activation::Tanh,
///           // seed for initial genome construction
///           seed: 42
/// #     },
/// #     mutations: vec![],
/// # };
/// #
/// # let genome_context = GenomeContext::new(parameters);
/// let genome_with_connections = genome_context.initialized_genome();
/// ```
/// "Initialized" here means the configured percent of connections have been constructed with random weights.
/// "Uninitialized" thereby implys no connections have been constructed, such a genome is also available:
///
/// ```
/// # use set_genome::{GenomeContext, activations::Activation, Parameters, Structure};
/// #
/// # let parameters = Parameters {
/// #     structure: Structure {
/// #         // ten inputs
/// #         number_of_inputs: 10,
/// #         // two outputs
/// #         number_of_outputs: 2,
/// #         // 100% connected
/// #         percent_of_connected_inputs: 1.0,
/// #         // specified output activation
/// #         outputs_activation: Activation::Tanh,
///           // seed for initial genome construction
///           seed: 42
///
/// #     },
/// #     mutations: vec![],
/// # };
/// #
/// # let genome_context = GenomeContext::new(parameters);
/// let genome_without_connections = genome_context.uninitialized_genome();
/// ```
/// Setting the `percent_of_connected_inputs` field in the [`parameters::Structure`] parameter to zero makes the
/// "initialized" and "uninitialized" genome look the same.
///
/// So we got ourselves a genome, let's mutate it: [`Genome::mutate_with_context`].
///
/// The possible mutations:
///
/// - [`Mutations::add_connection`]
/// - [`Mutations::add_node`]
/// - [`Mutations::add_recurrent_connection`]
/// - [`Mutations::change_activation`]
/// - [`Mutations::change_weights`]
/// - [`Mutations::remove_node`]
/// - [`Mutations::remove_connection`]
/// - [`Mutations::remove_recurrent_connection`]
///
/// To evaluate the function encoded in the genome check [this crate].
///
/// [thesis]: https://www.silvan.codes/SET-NEAT_Thesis.pdf
/// [this crate]: https://crates.io/crates/favannat
///
pub struct GonnerGenomeContext {
    pub parameters: Parameters,
    initialized_genome: Genome,
    uninitialized_genome: Genome,
}

impl Genome {
    /// Initialization connects the configured percent of inputs nodes to output nodes, i.e. it creates connection genes with random weights.
    pub fn init_with_context(&mut self, parameters: &Parameters) {
        self.init(&parameters.structure)
    }

    /// Apply all mutations listed in the [parameters of the context] with respect to their chance of happening.
    /// If a mutation is listed multiple times it is applied multiple times.
    ///
    /// This will probably be the most common way to apply mutations to a genome.
    ///
    /// # Examples
    ///
    /// ```
    /// use set_genome::GenomeContext;
    ///
    /// // Create a `GenomeContext`.
    /// let mut genome_context = GenomeContext::default();
    ///
    /// // Create an initialized `Genome`.
    /// let mut genome = genome_context.initialized_genome();
    ///
    /// // Randomly mutate the genome according to the available mutations listed in the parameters of the context and their corresponding chances .
    /// genome.mutate_with_context(&mut genome_context);
    /// ```
    ///
    /// [parameters of the context]: `Parameters`
    ///
    pub fn mutate_with_context(&mut self, parameters: &Parameters) -> MutationResult {
        for mutation in &parameters.mutations {
            // gamble for application of mutation right here instead of in mutate() ??
            mutation.mutate(self)?
        }
        Ok(())
    }

    /// Calls [`Mutations::add_node`] with `self`, should [`Mutations::AddNode`] be listed in the context.
    /// It needs to be listed as it provides parameters.
    pub fn add_node_with_context(&mut self, parameters: &Parameters) {
        let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

        for mutation in &parameters.mutations {
            if let Mutations::AddNode {
                activation_pool, ..
            } = mutation
            {
                Mutations::add_node(activation_pool, self, &mut rng)
            }
        }
    }

    /// Calls the [`Mutations::remove_node`] with `self`.
    pub fn remove_node_with_context(&mut self) -> MutationResult {
        let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

        Mutations::remove_node(self, &mut rng)
    }

    /// Calls the [`Mutations::remove_connection`] with `self`.
    pub fn remove_connection_with_context(&mut self) -> MutationResult {
        let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

        Mutations::remove_connection(self, &mut rng)
    }

    /// Calls the [`Mutations::remove_recurrent_connection`] with `self`.
    pub fn remove_recurrent_connection_with_context(&mut self) -> MutationResult {
        let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

        Mutations::remove_recurrent_connection(self, &mut rng)
    }

    /// Calls the [`Mutations::add_connection`] with `self`.
    pub fn add_connection_with_context(&mut self) -> MutationResult {
        let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

        Mutations::add_connection(self, &mut rng)
    }

    /// Calls the [`Mutations::add_recurrent_connection`] with `self`.
    pub fn add_recurrent_connection_with_context(&mut self) -> MutationResult {
        let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

        Mutations::add_recurrent_connection(self, &mut rng)
    }

    /// Calls [`Mutations::change_activation`] with `self`, should [`Mutations::ChangeActivation`] be listed in the context.
    /// It needs to be listed as it provides parameters.
    pub fn change_activation_with_context(&mut self, parameters: &Parameters) {
        let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

        for mutation in &parameters.mutations {
            if let Mutations::ChangeActivation {
                activation_pool, ..
            } = mutation
            {
                Mutations::change_activation(activation_pool, self, &mut rng)
            }
        }
    }

    /// Calls [`Mutations::change_weights`] with `self`, should [`Mutations::ChangeWeights`] be listed in the context.
    /// It needs to be listed as it provides parameters.
    pub fn change_weights_with_context(&mut self, parameters: &Parameters) {
        let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

        for mutation in &parameters.mutations {
            if let Mutations::ChangeWeights {
                percent_perturbed,
                weight_cap,
                ..
            } = *mutation
            {
                Mutations::change_weights(percent_perturbed, weight_cap, self, &mut rng)
            }
        }
    }
}
