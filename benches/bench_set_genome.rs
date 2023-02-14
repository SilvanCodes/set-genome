use criterion::{criterion_group, criterion_main, Criterion};
use rand::{rngs::SmallRng, SeedableRng};
use set_genome::{activations::Activation, GenomeContext, Mutations, Parameters};

pub fn crossover_same_genome_benchmark(c: &mut Criterion) {
    let gc = GenomeContext::default();
    let mut rng = SmallRng::from_entropy();

    let genome_0 = gc.initialized_genome();
    let genome_1 = gc.initialized_genome();

    c.bench_function("crossover same genome", |b| {
        b.iter(|| genome_0.cross_in(&genome_1, &mut rng))
    });
}

pub fn crossover_highly_mutated_genomes_benchmark(c: &mut Criterion) {
    let parameters = Parameters {
        seed: None,
        structure: Default::default(),
        mutations: vec![
            Mutations::AddNode {
                chance: 1.0,
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
            Mutations::AddConnection { chance: 1.0 },
        ],
    };

    let gc = GenomeContext::new(parameters);
    let mut rng = SmallRng::from_entropy();

    let mut genome_0 = gc.initialized_genome();
    let mut genome_1 = gc.initialized_genome();

    for _ in 0..100 {
        genome_0.mutate_with_context(&gc);
        genome_1.mutate_with_context(&gc);
    }

    c.bench_function("crossover highly mutated genomes", |b| {
        b.iter(|| genome_0.cross_in(&genome_1, &mut rng))
    });
}

pub fn mutate_genome_benchmark(c: &mut Criterion) {
    let parameters = Parameters {
        seed: None,
        structure: Default::default(),
        mutations: vec![
            Mutations::AddNode {
                chance: 1.0,
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
            Mutations::AddConnection { chance: 1.0 },
        ],
    };

    let gc = GenomeContext::new(parameters);

    let mut genome = gc.initialized_genome();

    c.bench_function("mutate genome", |b| {
        b.iter(|| genome.mutate_with_context(&gc))
    });
}

pub fn add_node_to_genome_benchmark(c: &mut Criterion) {
    let gc = GenomeContext::default();

    let mut genome = gc.initialized_genome();

    c.bench_function("add node to genome", |b| {
        b.iter(|| genome.add_node_with_context(&gc))
    });
}

criterion_group!(
    benches,
    mutate_genome_benchmark,
    crossover_same_genome_benchmark,
    crossover_highly_mutated_genomes_benchmark,
    add_node_to_genome_benchmark
);
criterion_main!(benches);
