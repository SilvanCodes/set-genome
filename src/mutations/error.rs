use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum MutationError {
    #[error("No two nodes could be connected by a new feed-forward connection.")]
    CouldNotAddFeedForwardConnection,
    #[error("No two nodes could be connected by a new recurrent connection.")]
    CouldNotAddRecurrentConnection,
    #[error("No removable node present in the genome.")]
    CouldNotRemoveNode,
    #[error("No removable feed-forward connection present in the genome.")]
    CouldNotRemoveFeedForwardConnection,
    #[error("No removable recurrent connection present in the genome.")]
    CouldNotRemoveRecurrentConnection,
}
