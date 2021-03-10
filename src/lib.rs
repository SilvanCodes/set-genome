use genes::IdGenerator;
use genome::Genome;
use mutations::Mutations;
use parameters::Parameters;
use rng::GenomeRng;

mod favannat_impl;
mod genes;
mod genome;
mod mutations;
mod parameters;
mod rng;

pub struct GenomeContext {
    id_gen: IdGenerator,
    rng: GenomeRng,
    parameters: Parameters,
    initialized_genome: Genome,
    uninitialized_genome: Genome,
}

impl GenomeContext {
    pub fn new(parameters: Parameters) -> Self {
        let mut id_gen = IdGenerator::default();
        let mut rng = GenomeRng::new(
            parameters.seed.unwrap_or(42),
            parameters.structure.weight_std_dev,
        );

        let uninitialized_genome = Genome::new(&mut id_gen, &parameters);

        let mut initialized_genome = uninitialized_genome.clone();
        initialized_genome.init(&mut rng, &parameters);

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
    pub fn mutate_with_context(&mut self, context: &mut GenomeContext) {
        self.mutate(
            &mut context.rng,
            &mut context.id_gen,
            &context.parameters.mutations,
        )
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
                percent_perturbed,
                weight_cap,
                ..
            } = *mutation
            {
                Mutations::change_weights(percent_perturbed, weight_cap, self, &mut context.rng)
            }
        }
    }
}
