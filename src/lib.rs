use genes::Connection;

pub use genes::{Activation, IdGenerator};
pub use genome::Genome;
pub use mutations::Mutations;
pub use parameters::Parameters;
pub use rng::GenomeRng;

mod favannat_impl;
mod genes;
mod genome;
mod mutations;
mod parameters;
mod rng;

pub struct GenomeContext {
    pub id_gen: IdGenerator,
    pub rng: GenomeRng,
    pub parameters: Parameters,
    initialized_genome: Genome,
    uninitialized_genome: Genome,
}

impl GenomeContext {
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

    pub fn initialized_genome(&self) -> Genome {
        self.initialized_genome.clone()
    }

    pub fn uninitialized_genome(&mut self) -> Genome {
        self.uninitialized_genome.clone()
    }
}

impl Default for GenomeContext {
    fn default() -> Self {
        Self::new(Parameters::default())
    }
}

impl Genome {
    pub fn init_with_context(&mut self, context: &mut GenomeContext) {
        for input in self
            .inputs
            .iterate_with_random_offset(&mut context.rng)
            .take(
                (context.parameters.structure.inputs_connected_percent
                    * context.parameters.structure.inputs as f64)
                    .ceil() as usize,
            )
        {
            // connect to every output
            for output in self.outputs.iter() {
                assert!(self.feed_forward.insert(Connection::new(
                    input.id,
                    context.rng.weight_perturbation(),
                    output.id
                )));
            }
        }
    }

    pub fn mutate_with_context(&mut self, context: &mut GenomeContext) {
        for mutation in &context.parameters.mutations {
            mutation.mutate(self, &mut context.rng, &mut context.id_gen);
        }
    }

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

    pub fn add_connection_with_context(
        &mut self,
        context: &mut GenomeContext,
    ) -> Result<(), &'static str> {
        Mutations::add_connection(self, &mut context.rng)
    }

    pub fn add_recurrent_connection_with_context(
        &mut self,
        context: &mut GenomeContext,
    ) -> Result<(), &'static str> {
        Mutations::add_recurrent_connection(self, &mut context.rng)
    }

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
