//! The shell's file directory as app-core's [`HostStorage`]: where a mobile
//! profile (its database, write-ahead log, config and update state) lives.
//!
//! One call per method, each applied by the shell before it replies (the
//! contract's one file queue), which is the order app-core's stream relies on.
//! A write larger than one request may carry is split into consecutive writes.

use std::io;

use eidola_app_core::storage::{HostFuture, HostStorage};

use crate::client::{ShellClient, Transport};
use crate::protocol::MAX_REQUEST_BYTES;

/// A profile's files, in the shell's file directory.
pub struct ShellFiles<T> {
    client: ShellClient<T>,
}

impl<T: Transport> ShellFiles<T> {
    pub fn new(client: ShellClient<T>) -> Self {
        Self { client }
    }
}

impl<T: Transport> HostStorage for ShellFiles<T> {
    fn read(&self, name: &str) -> HostFuture<'_, io::Result<Option<Vec<u8>>>> {
        let name = name.to_owned();
        Box::pin(async move { Ok(self.client.files_read(&name).await?) })
    }

    fn write_at(&self, name: &str, offset: u64, bytes: Vec<u8>) -> HostFuture<'_, io::Result<()>> {
        let name = name.to_owned();
        Box::pin(async move {
            if bytes.is_empty() {
                // Creates the file, as an empty write at an offset does.
                self.client.files_write(&name, offset, &[], false).await?;
                return Ok(());
            }
            let mut at = offset;
            for chunk in bytes.chunks(MAX_REQUEST_BYTES) {
                self.client.files_write(&name, at, chunk, false).await?;
                at += chunk.len() as u64;
            }
            Ok(())
        })
    }

    fn truncate(&self, name: &str, len: u64) -> HostFuture<'_, io::Result<()>> {
        let name = name.to_owned();
        Box::pin(async move { Ok(self.client.files_truncate(&name, len).await?) })
    }

    fn sync(&self, name: &str) -> HostFuture<'_, io::Result<()>> {
        let name = name.to_owned();
        Box::pin(async move { Ok(self.client.files_sync(&name).await?) })
    }

    fn rename(&self, from: &str, to: &str) -> HostFuture<'_, io::Result<()>> {
        let (from, to) = (from.to_owned(), to.to_owned());
        Box::pin(async move { Ok(self.client.files_rename(&from, &to).await?) })
    }

    fn remove(&self, name: &str) -> HostFuture<'_, io::Result<()>> {
        let name = name.to_owned();
        Box::pin(async move { Ok(self.client.files_remove(&name).await?) })
    }

    fn list(&self) -> HostFuture<'_, io::Result<Vec<String>>> {
        Box::pin(async move { Ok(self.client.files_list().await?) })
    }
}
