use rand::{prelude::SmallRng, Rng, RngCore, SeedableRng};
use rand_distr::{Distribution, Normal};

#[derive(Debug)]
pub struct GenomeRng {
    small: SmallRng,
    weight_distribution: Normal<f64>,
}

impl GenomeRng {
    pub fn new(seed: u64, std_dev: f64) -> Self {
        Self {
            small: SmallRng::seed_from_u64(seed),
            weight_distribution: Normal::new(0.0, std_dev)
                .expect("could not create weight distribution"),
        }
    }

    pub fn gamble(&mut self, chance: f64) -> bool {
        self.gen::<f64>() < chance
    }

    pub fn weight_perturbation(&mut self) -> f64 {
        self.weight_distribution.sample(&mut self.small)
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
