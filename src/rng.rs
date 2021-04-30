use rand::{prelude::SmallRng, Rng, RngCore, SeedableRng};
use rand_distr::{Distribution, Normal};

/// This struct serves as the randomness source for all operations.
#[derive(Debug)]
pub struct GenomeRng {
    small: SmallRng,
    weight_distribution: Normal<f64>,
    cap: f64,
}

impl GenomeRng {
    pub fn new(seed: u64, std_dev: f64, cap: f64) -> Self {
        Self {
            small: SmallRng::seed_from_u64(seed),
            weight_distribution: Normal::new(0.0, std_dev)
                .expect("could not create weight distribution"),
            cap,
        }
    }

    pub fn gamble(&mut self, chance: f64) -> bool {
        self.gen::<f64>() < chance
    }

    pub fn weight_perturbation(&mut self, weight: f64) -> f64 {
        let mut perturbation = self.weight_distribution.sample(&mut self.small);
        while (weight + perturbation) > self.cap || (weight + perturbation) < -self.cap {
            perturbation = -perturbation / 2.0;
        }
        weight + perturbation
    }
}

impl RngCore for GenomeRng {
    fn next_u32(&mut self) -> u32 {
        self.small.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.small.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.small.fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.small.try_fill_bytes(dest)
    }
}

#[cfg(test)]
mod tests {
    use super::GenomeRng;
    #[test]
    fn respect_weight_cap() {
        let cap = 1.0;
        let mut rng = GenomeRng::new(0, 0.5, cap);
        let mut weight = 0.0;

        for _ in 0..1000 {
            weight = rng.weight_perturbation(weight);
            assert!(weight <= cap && weight >= -cap, "{}", weight);
        }
    }
}
