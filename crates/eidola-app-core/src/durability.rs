//! The durability barrier: nothing leaves the device ahead of what it
//! depends on being on disk.
//!
//! Every request this crate sends may rest on a commit made just before it: a
//! credential marked spending before its spend is sent, a pending issuance
//! before it is asked for, a prefix-cache key's claim before the body carrying
//! it is written. Recovery after a crash reads those rows. If a request could
//! reach the server while the commit it rests on could still be lost, a
//! restart would find no trace of a spend the server has seen, or of an
//! issuance it has answered.
//!
//! [`barrier`] is the guarantee recovery needs. **Contract: once it returns,
//! every database commit made before the call is durable on the host.** The
//! transport awaits it before every request leaves the device, on every
//! transport (`transport::RequestBuilder::send`, `transport::Client::execute`),
//! and again after a deferred body has been produced and before its bytes are
//! written (`transport::deferred_body`). It never fails: a store that cannot
//! make a commit durable surfaces that at the commit.
//!
//! Where the database writes through to the filesystem before a commit
//! returns (every native build), every commit is already durable when it
//! returns, and the barrier returns at once.

/// Wait until every database commit made before this call is durable.
pub(crate) async fn barrier() {}
