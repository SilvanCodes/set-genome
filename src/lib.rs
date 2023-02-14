//! This crate is supposed to act as the representation/reproduction aspect in neuroevolution algorithms and may be combined with arbitrary selection mechanisms.
//!
//! # What you can do with this crate
//! ```
//! # use set_genome::{Genome, Parameters};
//! # use favannat::{
//! #   matrix::feedforward::fabricator::MatrixFeedForwardFabricator,
//! #    network::{Evaluator, Fabricator},
//! # };
//! # use nalgebra::dmatrix;
//! // Setup a genome context for networks with 10 inputs and 10 outputs.
//! let mut parameters = Parameters::basic(10, 10);
//!
//! // Initialize a genome.
//! let mut genome = Genome::initialized(&parameters);
//!
//! // Mutate a genome.
//! genome.mutate(&parameters);
//!
//! // Get a phenotype of the genome.
//! let network = MatrixFeedForwardFabricator::fabricate(&genome).expect("Cool network.");
//!
//! // Evaluate a network on an input.
//! let output = network.evaluate(dmatrix![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
//! ```
//!
//! # SET genome
//!
//! SET stands for **S**et **E**ncoded **T**opology and this crate implements a genetic data structure, the [`Genome`], using this set encoding to describe artificial neural networks (ANNs).
//! Further this crate defines operations on this genome, namely [`Mutations`] and [crossover]. Mutations alter a genome by adding or removing genes, crossover recombines two genomes.
//! To have an intuitive definition of crossover for network structures the [NEAT algorithm] defined a procedure and has to be understood as a mental predecessor to this SET encoding,
//! which very much is a formalization and progression of the ideas NEAT introduced regarding the genome.
//! The thesis describing this genome and other ideas can be found [here], a paper focusing just on the SET encoding will follow soon.
//!
//! # Getting started
//!
//! We start by defining our parameters:
//!
//! Suppose we know our task has ten inputs and two outputs, which translate to the input and output layer of our ANN.
//! Further we want 100% of our inputs nodes to be initially connected to the outputs and the outputs shall use the [`activations::Activation::Tanh`] function.
//! Also the weights of our connections are supposed to be capped between \[-1, 1\] and change by deltas sampled from a normal distribution with 0.1 standard deviation.
//!
//! ```
//! use set_genome::{activations::Activation, Parameters, Structure};
//!
//! let parameters = Parameters {
//!     structure: Structure {
//!         // ten inputs
//!         number_of_inputs: 10,
//!         // two outputs
//!         number_of_outputs: 2,
//!         // 100% connected
//!         percent_of_connected_inputs: 1.0,
//!         // specified output activation
//!         outputs_activation: Activation::Tanh,
//!         // seed for initial genome construction
//!         seed: 42
//!     },
//!     mutations: vec![],
//! };
//! ```
//! This allows us to create an initialized genome which conforms to our description above:
//!
//! ```
//! # use set_genome::{Genome, activations::Activation, Parameters, Structure};
//! #
//! # let parameters = Parameters {
//! #     structure: Structure {
//! #         // ten inputs
//! #         number_of_inputs: 10,
//! #         // two outputs
//! #         number_of_outputs: 2,
//! #         // 100% connected
//! #         percent_of_connected_inputs: 1.0,
//! #         // specified output activation
//! #         outputs_activation: Activation::Tanh,
//!           // seed for initial genome construction
//!           seed: 42
//! #     },
//! #     mutations: vec![],
//! # };
//! #
//! let genome_with_connections = Genome::initialized(&parameters);
//! ```
//! "Initialized" here means the configured percent of connections have been constructed with random weights.
//! "Uninitialized" thereby implys no connections have been constructed, such a genome is also available:
//!
//! ```
//! # use set_genome::{Genome, activations::Activation, Parameters, Structure};
//! #
//! # let parameters = Parameters {
//! #     structure: Structure {
//! #         // ten inputs
//! #         number_of_inputs: 10,
//! #         // two outputs
//! #         number_of_outputs: 2,
//! #         // 100% connected
//! #         percent_of_connected_inputs: 1.0,
//! #         // specified output activation
//! #         outputs_activation: Activation::Tanh,
//!           // seed for initial genome construction
//!           seed: 42
//!
//! #     },
//! #     mutations: vec![],
//! # };
//! #
//! let genome_without_connections = Genome::uninitialized(&parameters);
//! ```
//! Setting the `percent_of_connected_inputs` field in the [`parameters::Structure`] parameter to zero makes the
//! "initialized" and "uninitialized" genome look the same.
//!
//! So we got ourselves a genome, let's mutate it: [`Genome::mutate`].
//!
//! The possible mutations:
//!
//! - [`Mutations::add_connection`]
//! - [`Mutations::add_node`]
//! - [`Mutations::add_recurrent_connection`]
//! - [`Mutations::change_activation`]
//! - [`Mutations::change_weights`]
//! - [`Mutations::remove_node`]
//! - [`Mutations::remove_connection`]
//! - [`Mutations::remove_recurrent_connection`]
//!
//! //! # Features
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
//! [thesis]: https://www.silvan.codes/SET-NEAT_Thesis.pdf
//! [this crate]: https://crates.io/crates/favannat
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

impl Genome {
    /// Initialization connects the configured percent of inputs nodes to output nodes, i.e. it creates connection genes with random weights.
    pub fn uninitialized(parameters: &Parameters) -> Self {
        Self::new(&parameters.structure)
    }

    pub fn initialized(parameters: &Parameters) -> Self {
        let mut genome = Genome::new(&parameters.structure);
        genome.init(&parameters.structure);
        genome
    }

    /// Apply all mutations listed in the [`Parameters`] with respect to their chance of happening.
    /// If a mutation is listed multiple times it is applied multiple times.
    ///
    /// This will probably be the most common way to apply mutations to a genome.
    ///
    /// # Examples
    ///
    /// ```
    /// use set_genome::{Genome, Parameters};
    ///
    /// // Create parameters, usually read from a configuration file.
    /// let parameters = Parameters::default();
    ///
    /// // Create an initialized `Genome`.
    /// let mut genome = Genome::initialized(&parameters);
    ///
    /// // Randomly mutate the genome according to the available mutations listed in the parameters of the context and their corresponding chances .
    /// genome.mutate(&parameters);
    /// ```
    ///
    pub fn mutate(&mut self, parameters: &Parameters) -> MutationResult {
        let rng = &mut SmallRng::from_rng(thread_rng()).unwrap();

        for mutation in &parameters.mutations {
            // gamble for application of mutation right here instead of in mutate() ??
            mutation.mutate(self, rng)?
        }
        Ok(())
    }
}
