use criterion::{criterion_group, criterion_main, Criterion};
use set_genome::{activations::Activation, Genome, Mutations, Parameters};

pub fn crossover_same_genome_benchmark(c: &mut Criterion) {
    let parameters = Parameters::default();

    let genome_0 = Genome::initialized(&parameters.structure);
    let genome_1 = Genome::initialized(&parameters.structure);

    c.bench_function("crossover same genome", |b| {
        b.iter(|| genome_0.cross_in(&genome_1))
    });
}

pub fn crossover_highly_mutated_genomes_benchmark(c: &mut Criterion) {
    let parameters = Parameters {
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

    let mut genome_0 = Genome::initialized(&parameters.structure);
    let mut genome_1 = Genome::initialized(&parameters.structure);

    for _ in 0..100 {
        genome_0.mutate_with_context(&parameters);
        genome_1.mutate_with_context(&parameters);
    }

    c.bench_function("crossover highly mutated genomes", |b| {
        b.iter(|| genome_0.cross_in(&genome_1))
    });
}

pub fn mutate_genome_benchmark(c: &mut Criterion) {
    let parameters = Parameters {
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

    let mut genome = Genome::initialized(&parameters.structure);

    c.bench_function("mutate genome", |b| {
        b.iter(|| genome.mutate_with_context(&parameters))
    });
}

pub fn add_node_to_genome_benchmark(c: &mut Criterion) {
    let parameters = Parameters::default();

    let mut genome = Genome::initialized(&parameters.structure);

    c.bench_function("add node to genome", |b| {
        b.iter(|| genome.add_node_with_context(&parameters))
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
