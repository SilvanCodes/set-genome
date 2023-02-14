//! Lists constant functions matching the [`Activation`] enum variants.
//!
//! The pool of activation functions is the same as in [this paper](https://weightagnostic.github.io/).

use serde::{Deserialize, Serialize};

/// Possible activation functions for ANN nodes.
///
/// See the [actual functions listed here] under **Constants**.
///
/// [actual functions listed here]: ../activations/index.html#constants
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Activation {
    Linear,
    Sigmoid,
    Tanh,
    Gaussian,
    Step,
    Sine,
    Cosine,
    Inverse,
    Absolute,
    Relu,
    Squared,
}

impl Activation {
    pub fn all() -> Vec<Self> {
        vec![
            Self::Linear,
            Self::Sigmoid,
            Self::Tanh,
            Self::Gaussian,
            Self::Step,
            Self::Sine,
            Self::Cosine,
            Self::Inverse,
            Self::Absolute,
            Self::Relu,
            Self::Squared,
        ]
    }
}

/// Returns the argument unchanged.
pub const LINEAR: fn(f64) -> f64 = |val| val;

/// Steepened sigmoid function, the same use in the original NEAT paper.
pub const SIGMOID: fn(f64) -> f64 = |val| 1.0 / (1.0 + (-4.9 * val).exp());

/// It is a [rescaled sigmoid function].
///
/// [rescaled sigmoid function]: https://brenocon.com/blog/2013/10/tanh-is-a-rescaled-logistic-sigmoid-function/
pub const TANH: fn(f64) -> f64 = |val| 2.0 * SIGMOID(2.0 * val) - 1.0;

/// [Gaussian function] with parameters a = 1, b = 0, c = 1 a.k.a. standard normal distribution.
///
/// [Gaussian function]: https://en.wikipedia.org/wiki/Gaussian_function
pub const GAUSSIAN: fn(f64) -> f64 = |val| (val * val / -2.0).exp();

/// Returns one if argument greater than zero, else zero.
pub const STEP: fn(f64) -> f64 = |val| if val > 0.0 { 1.0 } else { 0.0 };

/// Returns sine of argument.
pub const SINE: fn(f64) -> f64 = |val| (val * std::f64::consts::PI).sin();

/// Returns cosine of argument.
pub const COSINE: fn(f64) -> f64 = |val| (val * std::f64::consts::PI).cos();

/// Returns negative argument.
pub const INVERSE: fn(f64) -> f64 = |val| -val;

/// Returns absolute value of argument.
pub const ABSOLUTE: fn(f64) -> f64 = |val| val.abs();

/// Returns argument if it is greater than zero, else zero.
pub const RELU: fn(f64) -> f64 = |val| 0f64.max(val);

/// Returns square of argument.
pub const SQUARED: fn(f64) -> f64 = |val| val * val;
