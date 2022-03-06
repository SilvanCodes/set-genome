use favannat::network::{EdgeLike, NetworkLike, NodeLike, Recurrent};

use crate::{
    genes::{activations, Activation, Connection, Node},
    genome::Genome,
};

impl NodeLike for Node {
    fn id(&self) -> usize {
        self.id.0
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
        self.input.0
    }
    fn end(&self) -> usize {
        self.output.0
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
