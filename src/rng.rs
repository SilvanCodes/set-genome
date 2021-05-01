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
    /// Creates a [`GenomeRng`].
    ///
    /// `seed` is specified for reproducibility of experiments.
    /// `std_dev` configures the standard deviation of the normal distribution from which the weight perturbations are sampled.
    /// `cap` specifies the upper and lower bound of values returned from [`GenomeRng::weight_perturbation`].
    ///
    /// ```
    /// use set_genome::GenomeRng;
    /// let genome_rng = GenomeRng::new(42, 0.1, 1.0);
    /// ```
    pub fn new(seed: u64, std_dev: f64, cap: f64) -> Self {
        Self {
            small: SmallRng::seed_from_u64(seed),
            weight_distribution: Normal::new(0.0, std_dev)
                .expect("could not create weight distribution"),
            cap,
        }
    }

    /// Returns true `chance` percent of the time.
    ///
    /// ```
    /// # use set_genome::GenomeRng;
    /// # let mut genome_rng = GenomeRng::new(42, 0.1, 1.0);
    /// // a coin flip
    /// if genome_rng.gamble(0.5) {
    ///     println!("heads")
    /// } else {
    ///     println!("tails")
    /// }
    ///
    /// // the following always happens, gambling with 100% chance to succeed
    /// assert!(genome_rng.gamble(1.0), "I should always win!");
    /// ```
    pub fn gamble(&mut self, chance: f64) -> bool {
        self.gen::<f64>() < chance
    }

    /// Returns the `weight` altered by some random value.
    ///
    /// ```
    /// # use set_genome::GenomeRng;
    /// let mut genome_rng = GenomeRng::new(42, 0.1, 1.0);
    /// // random_weight will probably be some small value,
    /// // definitely not bigger than 1.0 or smaller than -1.0.
    /// let random_weight = genome_rng.weight_perturbation(0.0);
    /// ```
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
