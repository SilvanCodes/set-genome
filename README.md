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

## Problems

Right now with representing weights as a bitvec the weights themselves become normally distributed.
That makes weigths overall tend to zero mean.
I observerved overall low weights and low progress.
Reducing the mutation rate per nucleotide from 0.1 to 0.01 did help.

Before that change only the initial weight was sampled from a such a distribution and then the subsequent update deltas during a weight mutation.
That did not entail the weigths themselves were normally distributed.

The precision with which weights can be expressed is now discrete an given by (number_of_ones - mean) / mean where number_of_ones is on the integer interval [0 - 64 * resolution] and mean is given by (64 * resolution / 2).
The resolution parameter so far is constant.
It could be made dynamic starting from 1 and increase due to "gene duplication" events.

Implementation by BitVec type.
Duplication rate similar to mutatio rate.

## Interpretation

One could use the Chi-Squared test to see if weights in selected individuals still follow a normal distribution.
If they do not this is a strong indicator that we are "better than randomness" and some weights are preferred over other.