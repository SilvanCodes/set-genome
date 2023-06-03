use bitvec::prelude::*;
use rand::Rng;
use seahash::SeaHasher;
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};

use super::{Gene, Id};

/// Struct describing a ANN connection.
///
/// A connection is characterised by its input/origin/start, its output/destination/end and its weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub input: Id,
    pub output: Id,
    weight: BitVec,
    pub id_counter: u64,
}

impl Connection {
    pub fn new(input: Id, weight_value: f64, output: Id) -> Self {
        let mut connection = Self {
            input,
            output,
            weight: bitvec!(0; 1),
            id_counter: 0,
        };

        connection.set_weight(weight_value);

        connection
    }

    pub fn new_with_resolution(
        input: Id,
        weight_value: f64,
        weight_resolution: usize,
        output: Id,
    ) -> Self {
        let mut connection = Self {
            input,
            output,
            weight: bitvec!(0; weight_resolution),
            id_counter: 0,
        };

        connection.set_weight(weight_value);

        connection
    }

    pub fn from_u64(input: Id, weight_state: usize, output: Id) -> Self {
        let weight_state = [weight_state];
        let weight = BitSlice::from_slice(&weight_state);

        Self {
            input,
            output,
            weight: BitVec::from_bitslice(&weight[0..2]),
            id_counter: 0,
        }
    }

    pub fn set_weight(&mut self, weight_value: f64) {
        let weight = &mut self.weight;

        let bits_mean = weight.len() as f64 / 2.0;

        let number_of_bits = (weight_value * bits_mean + bits_mean).round() as usize;

        // dbg!(&weight_value);
        // dbg!(&weight);
        // dbg!(&bits_mean);
        // dbg!(&number_of_bits);

        let bit_positions =
            rand::seq::index::sample(&mut rand::thread_rng(), weight.len(), number_of_bits)
                .into_vec();

        for index in 0..weight.len() {
            if bit_positions.contains(&index) {
                weight.set(index, true)
            } else {
                weight.set(index, false)
            }
        }

        // dbg!(&bit_positions);
        // dbg!(&weight);
    }

    pub fn id(&self) -> (Id, Id) {
        (self.input, self.output)
    }

    pub fn next_id(&mut self) -> Id {
        let mut id_hasher = SeaHasher::new();
        self.input.hash(&mut id_hasher);
        self.output.hash(&mut id_hasher);
        self.id_counter.hash(&mut id_hasher);
        self.id_counter += 1;
        Id(id_hasher.finish())
    }

    pub fn mutate_weight(&mut self, mutation_rate: f64, rng: &mut impl Rng) {
        // self.weight = Self::weight_perturbation(self.weight, standard_deviation, rng);

        for mut i in &mut self.weight {
            if rng.gen::<f64>() < mutation_rate {
                *i = !i.clone();
            }
        }
    }

    pub fn mutate_resolution(&mut self, duplication_rate: f64, rng: &mut impl Rng) {
        if rng.gen::<f64>() < duplication_rate {
            let last_bit = self.weight.last().as_deref().cloned().unwrap();
            self.weight.push(last_bit);
        }
        if rng.gen::<f64>() < duplication_rate {
            if self.weight.len() > 1 {
                self.weight.pop();
            }
        }
    }

    pub fn weight(&self) -> f64 {
        let bits_mean = self.weight.len() as f64 / 2.0;

        (self.weight.count_ones() as f64 - bits_mean) / bits_mean
    }

    // pub fn weight_perturbation(weight: f64, standard_deviation: f64, rng: &mut impl Rng) -> f64 {
    //     // approximatly normal distributed sample, see: https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution#Approximating_a_Normal_distribution
    //     let mut perturbation =
    //         ((0..12).map(|_| rng.gen::<f64>()).sum::<f64>() - 6.0) * standard_deviation;

    //     while (weight + perturbation) > 1.0 || (weight + perturbation) < -1.0 {
    //         perturbation = -perturbation / 2.0;
    //     }
    //     weight + perturbation
    // }
}

impl Gene for Connection {
    fn recombine(&self, other: &Self) -> Self {
        Self {
            weight: other.weight.clone(),
            ..*self
        }
    }
}

impl PartialEq for Connection {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for Connection {}

impl Hash for Connection {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

impl PartialOrd for Connection {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Connection {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id().cmp(&other.id())
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use crate::{Connection, Id};

    #[test]
    fn get_weight_in_and_out() {
        let connnection = Connection::new(Id(0), -1.0, Id(1));
        assert_eq!(connnection.weight(), -1.0);

        let connnection = Connection::new(Id(0), 1.0, Id(1));
        assert_eq!(connnection.weight(), 1.0);
    }

    #[test]
    fn mutate_weights() {
        let rng = &mut thread_rng();

        let mut connection = Connection::new(Id(0), rng.gen(), Id(1));

        // for _ in 0..100 {
        //     connections.push(Connection::new(Id(0), rng.gen(), Id(1)))
        // }

        // let mut connnection = Connection::new(Id(0), 0.0, Id(1));

        let mutation_rate = 0.1;

        for _ in 0..1000 {
            //     for connection in &mut connections {}
            connection.mutate_resolution(mutation_rate, rng);
            connection.mutate_weight(mutation_rate, rng);
            dbg!(&connection.weight());
        }

        // for _ in 0..10 {
        //     connnection.recombine(&connnection);
        //     dbg!(&connnection.weight());
        // }

        assert!(connection.weight() - 1.0 < f64::EPSILON)
    }
}
