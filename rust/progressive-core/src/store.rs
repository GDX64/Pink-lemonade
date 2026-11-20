//! Static multi-object byte-range storage traits.
//!
//! Strategies emit static artifacts through [`ObjectWriter`] (supporting streaming writes
//! for multi-gigabyte datasets) and read them online through [`ByteRangeStore`]
//! (mirroring HTTP `Range` requests against static hosting or object storage).

use std::future::Future;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use crate::types::StoreError;

/// Metadata for a single static object in a [`ByteRangeStore`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectMeta {
    pub key: String,
    pub len: u64,
}

/// Read-only static byte-range server interface.
///
/// Represents dumb object/file storage capable only of serving byte slices by `(key, offset, len)`.
pub trait ByteRangeStore: Send + Sync {
    /// Read up to `len` bytes starting at `offset` from object `key`.
    ///
    /// Clamps at EOF if `offset <= object_len` and `offset + len > object_len`, matching
    /// HTTP `Range` semantics. Returns [`StoreError::OutOfRange`] if `offset > object_len`.
    fn read_range(
        &self,
        key: &str,
        offset: u64,
        len: usize,
    ) -> impl Future<Output = Result<Vec<u8>, StoreError>> + Send;

    /// Return the total byte length of object `key`.
    fn object_len(&self, key: &str) -> impl Future<Output = Result<u64, StoreError>> + Send;

    /// List all objects and their byte lengths.
    fn list_objects(&self) -> impl Future<Output = Result<Vec<ObjectMeta>, StoreError>> + Send;
}

impl<S: ByteRangeStore + ?Sized> ByteRangeStore for Arc<S> {
    fn read_range(
        &self,
        key: &str,
        offset: u64,
        len: usize,
    ) -> impl Future<Output = Result<Vec<u8>, StoreError>> + Send {
        (**self).read_range(key, offset, len)
    }

    fn object_len(&self, key: &str) -> impl Future<Output = Result<u64, StoreError>> + Send {
        (**self).object_len(key)
    }

    fn list_objects(&self) -> impl Future<Output = Result<Vec<ObjectMeta>, StoreError>> + Send {
        (**self).list_objects()
    }
}

/// Write-side interface used by [`crate::DatasetBuilder`] to emit static objects.
pub trait ObjectWriter {
    type Sink: Write + Send;

    /// Open a streaming writer for object `key`, replacing any existing object at that key.
    ///
    /// Callers should flush and drop the returned sink when finished.
    fn create_object(&mut self, key: &str) -> Result<Self::Sink, StoreError>;

    /// Convenience helper for writing a complete in-memory byte slice as an object.
    fn write_object(&mut self, key: &str, bytes: &[u8]) -> Result<(), StoreError> {
        let mut sink = self.create_object(key)?;
        sink.write_all(bytes)?;
        sink.flush()?;
        Ok(())
    }
}

/// Validate an object key so all storage backends enforce identical safe relative paths.
pub fn validate_key(key: &str) -> Result<PathBuf, StoreError> {
    if key.is_empty() || key.starts_with('/') || key.starts_with('\\') {
        return Err(StoreError::InvalidKey(key.to_owned()));
    }
    let path = Path::new(key);
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            _ => return Err(StoreError::InvalidKey(key.to_owned())),
        }
    }
    if normalized.as_os_str().is_empty() {
        return Err(StoreError::InvalidKey(key.to_owned()));
    }
    Ok(normalized)
}
