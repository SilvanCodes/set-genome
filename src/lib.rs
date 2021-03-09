use genes::IdGenerator;
use genome::Genome;
use parameters::Parameters;
use rng::GenomeRng;

mod genes;
mod genome;
mod parameters;
mod rng;

pub struct GenomeContext {
    id_gen: IdGenerator,
    rng: GenomeRng,
    parameters: Parameters,
    initial_genome: Genome,
}

impl GenomeContext {
    pub fn new(parameters: Parameters) -> Self {
        let mut id_gen = IdGenerator::default();
        let mut rng = GenomeRng::new(
            parameters.seed.unwrap_or(42),
            parameters.mutations.weight_perturbation_std_dev,
        );

        let mut initial_genome = Genome::new(&mut id_gen, &parameters);
        initial_genome.init(&mut rng, &parameters);

        Self {
            id_gen,
            rng,
            parameters,
            initial_genome,
        }
    }

    pub fn initialized_genome(&self) -> Genome {
        self.initial_genome.clone()
    }
}

impl Default for GenomeContext {
    fn default() -> Self {
        Self::new(Parameters::default())
    }
}

impl Genome {
    pub fn mutate_with_context(&mut self, context: &mut GenomeContext) {
        self.mutate(&mut context.rng, &mut context.id_gen, &context.parameters)
    }

    pub fn add_node_with_context(&mut self, context: &mut GenomeContext) {
        self.add_node(&mut context.rng, &mut context.id_gen, &context.parameters)
    }

    pub fn add_connection_with_context(&mut self, context: &mut GenomeContext) {
        self.add_connection(&mut context.rng, &context.parameters);
    }

    pub fn change_activation_with_context(&mut self, context: &mut GenomeContext) {
        self.change_activation(&mut context.rng, &context.parameters)
    }

    pub fn change_weights_with_context(&mut self, context: &mut GenomeContext) {
        self.change_weights(&mut context.rng, &context.parameters)
    }
}
