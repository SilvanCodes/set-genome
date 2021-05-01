# set_genome

This crate is supposed to act as the representation/reproduction aspect in neuroevolution algorithms and may be combined with arbitrary selection mechanisms.

SET stands for **S**et **E**ncoded **T**opology and this crate implements a genetic data structure, the `Genome`,
using this set encoding to describe artificial neural networks (ANNs).
Further this crate defines operations on this genome, namely `Mutations` and `Crossover`.
Mutations alter a genome by adding or removing genes, crossover recombines two genomes.
To have an intuitive definition of crossover for network structures the [NEAT algorithm] defined a procedure and has to be understood as a mental predecessor to this SET encoding,
which very much is a formalization and progression of the ideas NEAT introduced regarding the genome.
The thesis describing this genome and other ideas can be found [here], a paper focusing just on the SET encoding will follow soon.

[neat algorithm]: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
[here]: https://www.silvan.codes/SET-NEAT_Thesis.pdf

## Usage

```toml
[dependencies]
set_genome = "0.1"
```

See the [documentation] more information.

[documentation]: https://docs.rs/set_genome
