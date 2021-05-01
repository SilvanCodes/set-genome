use std::{collections::HashMap, ops::RangeFrom};

use super::{id_iter::IdIter, Id};

/// Acts as a generator and cache for ANN node identities.
///
/// For any evolution process you should only ever use exactly one [`IdGenerator`] and pass it around whenever one is required,
/// as several operations and gurantees rely on the cache inside the generator to function as defined.
/// Usually it suffices to use the `<operation>_with_context` functions on the [`crate::Genome`] and let the [`crate::GenomeContext`] handle the identity generation.
///
/// ```
/// use set_genome::IdGenerator;
/// let id_generator = IdGenerator::default();
/// ```
#[derive(Debug)]
pub struct IdGenerator {
    id_gen: RangeFrom<usize>,
    id_cache: HashMap<(Id, Id), Vec<Id>>,
}

impl Default for IdGenerator {
    fn default() -> Self {
        IdGenerator {
            id_gen: 0..,
            id_cache: HashMap::new(),
        }
    }
}

impl IdGenerator {
    /// Returns the next unique identity.
    ///
    /// ```
    /// # use set_genome::{IdGenerator, Id};
    /// let mut id_generator = IdGenerator::default();
    /// assert_eq!(id_generator.next_id(), Id(0));
    /// assert_eq!(id_generator.next_id(), Id(1));
    /// ```
    pub fn next_id(&mut self) -> Id {
        self.id_gen.next().map(Id).unwrap()
    }

    /// Returns an iterator over identities for the given cache key which automatically extends the cache with new entries when iterated beyond the entries cached so far.
    ///
    /// ```
    /// # use set_genome::{IdGenerator, Id};
    /// // cache is initially empty
    /// let mut id_generator = IdGenerator::default();
    /// // ask for any arbitrary cache entry
    /// let mut id_iter = id_generator.cached_id_iter((Id(9), Id(9)));
    /// // we get the next available id
    /// assert_eq!(id_iter.next(), Some(Id(0)));
    /// // the id_iter advanced the generator
    /// assert_eq!(id_generator.next_id(), Id(1));
    /// // asking for the same entry later returns the cached entry and then fresh ones
    /// let mut id_iter = id_generator.cached_id_iter((Id(9), Id(9)));
    /// assert_eq!(id_iter.next(), Some(Id(0)));
    /// assert_eq!(id_iter.next(), Some(Id(2)));
    /// ```
    pub fn cached_id_iter(&mut self, cache_key: (Id, Id)) -> IdIter {
        let cache_entry = self.id_cache.entry(cache_key).or_insert_with(Vec::new);
        IdIter::new(cache_entry, &mut self.id_gen)
    }
}

#[cfg(test)]
mod tests {

    use super::{Id, IdGenerator};

    #[test]
    fn get_new_id() {
        let mut test_id_manager = IdGenerator::default();

        assert_eq!(test_id_manager.next_id(), Id(0));
        assert_eq!(test_id_manager.next_id(), Id(1));
        assert_eq!(test_id_manager.next_id(), Id(2));
    }

    #[test]
    fn iter_cached_ids() {
        let mut test_id_manager = IdGenerator::default();

        let mut test_id_iter_0 = test_id_manager.cached_id_iter((Id(4), Id(2)));

        assert_eq!(test_id_iter_0.next(), Some(Id(0)));
        assert_eq!(test_id_iter_0.next(), Some(Id(1)));

        let mut test_id_iter_1 = test_id_manager.cached_id_iter((Id(4), Id(2)));

        assert_eq!(test_id_iter_1.next(), Some(Id(0))); // cached entry
        assert_eq!(test_id_iter_1.next(), Some(Id(1))); // cached entry
        assert_eq!(test_id_iter_1.next(), Some(Id(2))); // new entry
    }
}
