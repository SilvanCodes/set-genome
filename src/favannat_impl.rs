use favannat::network::{EdgeLike, NetworkLike, NodeLike, Recurrent};

use crate::{
    genes::{activations, Activation, Connection, Node},
    genome::Genome,
};

impl NodeLike for Node {
    fn id(&self) -> usize {
        self.id.0 as usize
    }
    fn activation(&self) -> fn(f64) -> f64 {
        match self.activation {
            Activation::Linear => activations::LINEAR,
            Activation::Sigmoid => activations::SIGMOID,
            Activation::Gaussian => activations::GAUSSIAN,
            Activation::Tanh => activations::TANH,
            Activation::Step => activations::STEP,
            Activation::Sine => activations::SINE,
            Activation::Cosine => activations::COSINE,
            Activation::Inverse => activations::INVERSE,
            Activation::Absolute => activations::ABSOLUTE,
            Activation::Relu => activations::RELU,
            Activation::Squared => activations::SQUARED,
        }
    }
}

impl EdgeLike for Connection {
    fn start(&self) -> usize {
        self.input.0 as usize
    }
    fn end(&self) -> usize {
        self.output.0 as usize
    }
    fn weight(&self) -> f64 {
        self.weight
    }
}

impl NetworkLike<Node, Connection> for Genome {
    fn nodes(&self) -> Vec<&Node> {
        self.nodes().collect()
    }
    fn edges(&self) -> Vec<&Connection> {
        self.feed_forward.as_sorted_vec()
    }
    fn inputs(&self) -> Vec<&Node> {
        self.inputs.as_sorted_vec()
    }
    fn outputs(&self) -> Vec<&Node> {
        self.outputs.as_sorted_vec()
    }
    fn hidden(&self) -> Vec<&Node> {
        self.hidden.as_sorted_vec()
    }
}

impl Recurrent<Node, Connection> for Genome {
    fn recurrent_edges(&self) -> Vec<&Connection> {
        self.recurrent.as_sorted_vec()
    }
}

#[cfg(test)]
mod tests {
    use favannat::{MatrixRecurrentFabricator, StatefulEvaluator, StatefulFabricator};
    use rand_distr::{Distribution, Uniform};

    use crate::{activations::Activation, Genome, Mutations, Parameters, Structure};

    // This test brakes with favannat version 0.6.1 due to a bug there. Now with favannat 0.6.2 it is fine.
    #[test]
    fn verify_output_does_not_occasionally_leak_internal_state() {
        let parameters = Parameters {
            structure: Structure {
                number_of_inputs: 13,
                number_of_outputs: 3,
                percent_of_connected_inputs: 1.0,
                outputs_activation: Activation::Sigmoid,
                seed: 42,
            },
            mutations: vec![
                Mutations::ChangeWeights {
                    chance: 0.8,
                    percent_perturbed: 0.5,
                    standard_deviation: 0.2,
                },
                Mutations::AddNode {
                    chance: 0.1,
                    activation_pool: vec![
                        Activation::Sigmoid,
                        Activation::Tanh,
                        Activation::Gaussian,
                        Activation::Step,
                        // Activation::Sine,
                        // Activation::Cosine,
                        Activation::Inverse,
                        Activation::Absolute,
                        Activation::Relu,
                    ],
                },
                Mutations::AddConnection { chance: 0.2 },
                Mutations::AddConnection { chance: 0.02 },
                Mutations::AddRecurrentConnection { chance: 0.1 },
                Mutations::RemoveConnection { chance: 0.05 },
                Mutations::RemoveConnection { chance: 0.01 },
                Mutations::RemoveNode { chance: 0.05 },
            ],
        };

        let mut genome = Genome::initialized(&parameters);

        for _ in 0..100 {
            genome.mutate();
        }

        let mut evaluator = MatrixRecurrentFabricator::fabricate(&genome).expect("not okay");

        let between = Uniform::from(-10000.0..10000.0);
        let mut rng = rand::thread_rng();

        for _ in 0..1000 {
            let input = (0..13)
                .map(|_| between.sample(&mut rng))
                .collect::<Vec<_>>();
            let output = evaluator.evaluate(input);
            assert!(
                output[0] <= 1.0,
                "got {} which is bigger than 1.0, genome: {:?}, evaluator: {:?}",
                output[0],
                &genome,
                &evaluator
            );
            assert!(
                output[0] >= 0.0,
                "got {} which is smaller than 0.0, genome: {:?}, evaluator: {:?}",
                output[0],
                &genome,
                &evaluator
            );
            assert!(
                output[1] <= 1.0,
                "got {} which is bigger than 1.0, genome: {:?}, evaluator: {:?}",
                output[1],
                &genome,
                &evaluator
            );
            assert!(
                output[1] >= 0.0,
                "got {} which is smaller than 0.0, genome: {:?}, evaluator: {:?}",
                output[1],
                &genome,
                &evaluator
            );
            assert!(
                output[2] <= 1.0,
                "got {} which is bigger than 1.0, genome: {:?}, evaluator: {:?}",
                output[2],
                &genome,
                &evaluator
            );
            assert!(
                output[2] >= 0.0,
                "got {} which is smaller than 0.0, genome: {:?}, evaluator: {:?}",
                output[2],
                &genome,
                &evaluator
            );
        }
    }
}
