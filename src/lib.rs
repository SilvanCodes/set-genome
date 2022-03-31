//! This crate is supposed to act as the representation/reproduction aspect in neuroevolution algorithms and may be combined with arbitrary selection mechanisms.
//!
//! # Getting started
//! Head over to [`GenomeContext`] to understand how to use this crate.
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

pub use genes::{activations, Connection, Id, IdGenerator, Node};
pub use genome::Genome;
pub use mutations::{MutationError, MutationResult, Mutations};
pub use parameters::{Parameters, Structure};
pub use rng::GenomeRng;

#[cfg(feature = "favannat")]
mod favannat_impl;
mod genes;
mod genome;
mod mutations;
mod parameters;
mod rng;

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
///     seed: None,
///     structure: Structure {
///         // ten inputs
///         number_of_inputs: 10,
///         // two outputs
///         number_of_outputs: 2,
///         // 100% connected
///         percent_of_connected_inputs: 1.0,
///         // specified output activation
///         outputs_activation: Activation::Tanh,
///         // delta distribution
///         weight_std_dev: 0.1,
///         // intervall constraint, applies as [-weight_cap, weight_cap]
///         weight_cap: 1.0,
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
/// #     seed: None,
/// #     structure: Structure {
/// #         // ten inputs
/// #         number_of_inputs: 10,
/// #         // two outputs
/// #         number_of_outputs: 2,
/// #         // 100% connected
/// #         percent_of_connected_inputs: 1.0,
/// #         // specified output activation
/// #         outputs_activation: Activation::Tanh,
/// #         // delta distribution
/// #         weight_std_dev: 0.1,
/// #         // intervall constraint, applies as [-weight_cap, weight_cap]
/// #         weight_cap: 1.0,
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
/// #     seed: None,
/// #     structure: Structure {
/// #         // ten inputs
/// #         number_of_inputs: 10,
/// #         // two outputs
/// #         number_of_outputs: 2,
/// #         // 100% connected
/// #         percent_of_connected_inputs: 1.0,
/// #         // specified output activation
/// #         outputs_activation: Activation::Tanh,
/// #         // delta distribution
/// #         weight_std_dev: 0.1,
/// #         // intervall constraint, applies as [-weight_cap, weight_cap]
/// #         weight_cap: 1.0,
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
pub struct GenomeContext {
    pub id_gen: IdGenerator,
    pub rng: GenomeRng,
    pub parameters: Parameters,
    initialized_genome: Genome,
    uninitialized_genome: Genome,
}

impl GenomeContext {
    /// Returns a new `GenomeContext` from the parameters.
    pub fn new(parameters: Parameters) -> Self {
        let mut id_gen = IdGenerator::default();
        let mut rng = GenomeRng::new(
            parameters.seed.unwrap_or(42),
            parameters.structure.weight_std_dev,
            parameters.structure.weight_cap,
        );

        let uninitialized_genome = Genome::new(&mut id_gen, &parameters.structure);

        let mut initialized_genome = uninitialized_genome.clone();
        initialized_genome.init(&mut rng, &parameters.structure);

        Self {
            id_gen,
            rng,
            parameters,
            initialized_genome,
            uninitialized_genome,
        }
    }

    /// Returns an initialized genome, see [`Genome::init_with_context`].
    pub fn initialized_genome(&self) -> Genome {
        self.initialized_genome.clone()
    }

    /// Returns an uninitialized genome, see [`Genome::init_with_context`].
    pub fn uninitialized_genome(&self) -> Genome {
        self.uninitialized_genome.clone()
    }
}

impl Default for GenomeContext {
    fn default() -> Self {
        Self::new(Parameters::default())
    }
}

impl Genome {
    /// Initialization connects the configured percent of inputs nodes to output nodes, i.e. it creates connection genes with random weights.
    pub fn init_with_context(&mut self, context: &mut GenomeContext) {
        self.init(&mut context.rng, &context.parameters.structure)
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
    pub fn mutate_with_context(&mut self, context: &mut GenomeContext) -> MutationResult {
        for mutation in &context.parameters.mutations {
            mutation.mutate(self, &mut context.rng, &mut context.id_gen)?
        }
        Ok(())
    }

    /// Calls [`Mutations::add_node`] with `self`, should [`Mutations::AddNode`] be listed in the context.
    /// It needs to be listed as it provides parameters.
    pub fn add_node_with_context(&mut self, context: &mut GenomeContext) {
        for mutation in &context.parameters.mutations {
            if let Mutations::AddNode {
                activation_pool, ..
            } = mutation
            {
                Mutations::add_node(activation_pool, self, &mut context.rng, &mut context.id_gen)
            }
        }
    }

    /// Calls the [`Mutations::remove_node`] with `self`.
    pub fn remove_node_with_context(&mut self, context: &mut GenomeContext) -> MutationResult {
        Mutations::remove_node(self, &mut context.rng)
    }

    /// Calls the [`Mutations::remove_connection`] with `self`.
    pub fn remove_connection_with_context(
        &mut self,
        context: &mut GenomeContext,
    ) -> MutationResult {
        Mutations::remove_connection(self, &mut context.rng)
    }

    /// Calls the [`Mutations::remove_recurrent_connection`] with `self`.
    pub fn remove_recurrent_connection_with_context(
        &mut self,
        context: &mut GenomeContext,
    ) -> MutationResult {
        Mutations::remove_recurrent_connection(self, &mut context.rng)
    }

    /// Calls the [`Mutations::add_connection`] with `self`.
    pub fn add_connection_with_context(&mut self, context: &mut GenomeContext) -> MutationResult {
        Mutations::add_connection(self, &mut context.rng)
    }

    /// Calls the [`Mutations::add_recurrent_connection`] with `self`.
    pub fn add_recurrent_connection_with_context(
        &mut self,
        context: &mut GenomeContext,
    ) -> MutationResult {
        Mutations::add_recurrent_connection(self, &mut context.rng)
    }

    /// Calls [`Mutations::change_activation`] with `self`, should [`Mutations::ChangeActivation`] be listed in the context.
    /// It needs to be listed as it provides parameters.
    pub fn change_activation_with_context(&mut self, context: &mut GenomeContext) {
        for mutation in &context.parameters.mutations {
            if let Mutations::ChangeActivation {
                activation_pool, ..
            } = mutation
            {
                Mutations::change_activation(activation_pool, self, &mut context.rng)
            }
        }
    }

    /// Calls [`Mutations::change_weights`] with `self`, should [`Mutations::ChangeWeights`] be listed in the context.
    /// It needs to be listed as it provides parameters.
    pub fn change_weights_with_context(&mut self, context: &mut GenomeContext) {
        for mutation in &context.parameters.mutations {
            if let Mutations::ChangeWeights {
                percent_perturbed, ..
            } = *mutation
            {
                Mutations::change_weights(percent_perturbed, self, &mut context.rng)
            }
        }
    }
}
