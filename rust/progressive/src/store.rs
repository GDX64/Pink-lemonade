//! Dumb static multi-object byte-range storage traits and concrete backends.
//!
//! Strategies build static artifacts through [`ObjectWriter`] (supporting streaming writes
//! for multi-gigabyte datasets) and read them back online through [`ByteRangeStore`]
//! (mirroring HTTP `Range` requests against static hosting or object storage).

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::future::Future;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

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
    /// Callers should flush and drop the returned sink (or call `flush`) when finished.
    fn create_object(&mut self, key: &str) -> Result<Self::Sink, StoreError>;

    /// Convenience helper for writing a complete in-memory byte slice as an object.
    fn write_object(&mut self, key: &str, bytes: &[u8]) -> Result<(), StoreError> {
        let mut sink = self.create_object(key)?;
        sink.write_all(bytes)?;
        sink.flush()?;
        Ok(())
    }
}

/// Validate an object key so both in-memory and disk stores enforce identical safe relative paths.
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

/// Fast in-memory multi-object store for unit tests and small/medium quality benchmarks.
#[derive(Debug, Default, Clone)]
pub struct MemoryStore {
    objects: Arc<RwLock<BTreeMap<String, Vec<u8>>>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total bytes stored across all objects.
    pub fn total_bytes(&self) -> u64 {
        let map = self.objects.read().expect("memory store lock poisoned");
        map.values().map(|v| v.len() as u64).sum()
    }

    /// Number of objects stored.
    pub fn object_count(&self) -> usize {
        let map = self.objects.read().expect("memory store lock poisoned");
        map.len()
    }
}

/// Streaming writer that commits bytes into a [`MemoryStore`] on `flush` and `drop`.
pub struct MemoryObjectSink {
    key: String,
    buf: Vec<u8>,
    objects: Arc<RwLock<BTreeMap<String, Vec<u8>>>>,
}

impl MemoryObjectSink {
    fn commit(&self) {
        let mut map = self.objects.write().expect("memory store lock poisoned");
        map.insert(self.key.clone(), self.buf.clone());
    }
}

impl Write for MemoryObjectSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buf.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.commit();
        Ok(())
    }
}

impl Drop for MemoryObjectSink {
    fn drop(&mut self) {
        let mut map = self.objects.write().expect("memory store lock poisoned");
        map.insert(self.key.clone(), std::mem::take(&mut self.buf));
    }
}

impl ObjectWriter for MemoryStore {
    type Sink = MemoryObjectSink;

    fn create_object(&mut self, key: &str) -> Result<Self::Sink, StoreError> {
        let _ = validate_key(key)?;
        Ok(MemoryObjectSink {
            key: key.to_owned(),
            buf: Vec::new(),
            objects: Arc::clone(&self.objects),
        })
    }

    fn write_object(&mut self, key: &str, bytes: &[u8]) -> Result<(), StoreError> {
        let _ = validate_key(key)?;
        let mut map = self.objects.write().expect("memory store lock poisoned");
        map.insert(key.to_owned(), bytes.to_vec());
        Ok(())
    }
}

impl ByteRangeStore for MemoryStore {
    async fn read_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>, StoreError> {
        let _ = validate_key(key)?;
        let map = self.objects.read().expect("memory store lock poisoned");
        let bytes = map
            .get(key)
            .ok_or_else(|| StoreError::NotFound(key.to_owned()))?;
        let object_len = bytes.len() as u64;
        if offset > object_len {
            return Err(StoreError::OutOfRange {
                key: key.to_owned(),
                offset,
                object_len,
            });
        }
        let start = offset as usize;
        let end = start.saturating_add(len).min(bytes.len());
        Ok(bytes[start..end].to_vec())
    }

    async fn object_len(&self, key: &str) -> Result<u64, StoreError> {
        let _ = validate_key(key)?;
        let map = self.objects.read().expect("memory store lock poisoned");
        map.get(key)
            .map(|b| b.len() as u64)
            .ok_or_else(|| StoreError::NotFound(key.to_owned()))
    }

    async fn list_objects(&self) -> Result<Vec<ObjectMeta>, StoreError> {
        let map = self.objects.read().expect("memory store lock poisoned");
        Ok(map
            .iter()
            .map(|(k, v)| ObjectMeta {
                key: k.clone(),
                len: v.len() as u64,
            })
            .collect())
    }
}

/// Directory-backed multi-object store for multi-million / GB-scale datasets on disk.
///
/// Writes stream directly to files without buffering entire artifacts in RAM; range reads
/// perform positional file I/O on a blocking thread pool so concurrent async fetches never
/// stall the Tokio runtime.
#[derive(Debug, Clone)]
pub struct DiskStore {
    root: PathBuf,
    /// Optional cache of open read-only file handles keyed by object key.
    open_files: Arc<Mutex<BTreeMap<String, Arc<Mutex<File>>>>>,
}

impl DiskStore {
    /// Open or create a directory-backed store at `root`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, StoreError> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            open_files: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn resolve(&self, key: &str) -> Result<PathBuf, StoreError> {
        let rel = validate_key(key)?;
        Ok(self.root.join(rel))
    }

    /// Walk the store directory and return total bytes across all objects.
    pub fn total_bytes(&self) -> Result<u64, StoreError> {
        let mut total = 0u64;
        for meta in collect_disk_objects(&self.root, Path::new(""))? {
            total += meta.len;
        }
        Ok(total)
    }

    /// Remove the entire backing directory and all stored objects.
    pub fn remove_dir_all(self) -> Result<(), StoreError> {
        drop(self.open_files);
        if self.root.exists() {
            fs::remove_dir_all(&self.root)?;
        }
        Ok(())
    }
}

impl ObjectWriter for DiskStore {
    type Sink = BufWriter<File>;

    fn create_object(&mut self, key: &str) -> Result<Self::Sink, StoreError> {
        let path = self.resolve(key)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        // Evict any cached read handle for this key before overwriting.
        self.open_files
            .lock()
            .expect("open_files lock poisoned")
            .remove(key);
        let file = File::create(path)?;
        Ok(BufWriter::new(file))
    }
}

impl ByteRangeStore for DiskStore {
    async fn read_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>, StoreError> {
        let path = self.resolve(key)?;
        let key_owned = key.to_owned();
        let open_files = Arc::clone(&self.open_files);

        tokio::task::spawn_blocking(move || {
            let file_handle = {
                let mut cache = open_files.lock().expect("open_files lock poisoned");
                if let Some(existing) = cache.get(&key_owned) {
                    Arc::clone(existing)
                } else {
                    let file = match File::open(&path) {
                        Ok(f) => f,
                        Err(err) if err.kind() == io::ErrorKind::NotFound => {
                            return Err(StoreError::NotFound(key_owned));
                        }
                        Err(err) => return Err(StoreError::Io(err)),
                    };
                    let handle = Arc::new(Mutex::new(file));
                    cache.insert(key_owned.clone(), Arc::clone(&handle));
                    handle
                }
            };

            let mut file = file_handle.lock().expect("file handle lock poisoned");
            let object_len = file.metadata()?.len();
            if offset > object_len {
                return Err(StoreError::OutOfRange {
                    key: key_owned,
                    offset,
                    object_len,
                });
            }
            let remaining = (object_len - offset) as usize;
            let to_read = len.min(remaining);
            if to_read == 0 {
                return Ok(Vec::new());
            }
            file.seek(SeekFrom::Start(offset))?;
            let mut buf = vec![0u8; to_read];
            file.read_exact(&mut buf)?;
            Ok(buf)
        })
        .await
        .map_err(|err| StoreError::Io(io::Error::other(err)))?
    }

    async fn object_len(&self, key: &str) -> Result<u64, StoreError> {
        let path = self.resolve(key)?;
        let key_owned = key.to_owned();
        tokio::task::spawn_blocking(move || match fs::metadata(&path) {
            Ok(meta) if meta.is_file() => Ok(meta.len()),
            Ok(_) => Err(StoreError::NotFound(key_owned)),
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                Err(StoreError::NotFound(key_owned))
            }
            Err(err) => Err(StoreError::Io(err)),
        })
        .await
        .map_err(|err| StoreError::Io(io::Error::other(err)))?
    }

    async fn list_objects(&self) -> Result<Vec<ObjectMeta>, StoreError> {
        let root = self.root.clone();
        tokio::task::spawn_blocking(move || collect_disk_objects(&root, Path::new("")))
            .await
            .map_err(|err| StoreError::Io(io::Error::other(err)))?
    }
}

fn collect_disk_objects(root: &Path, prefix: &Path) -> Result<Vec<ObjectMeta>, StoreError> {
    let dir = root.join(prefix);
    let mut out = Vec::new();
    if !dir.exists() {
        return Ok(out);
    }
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let rel = prefix.join(entry.file_name());
        if file_type.is_dir() {
            out.extend(collect_disk_objects(root, &rel)?);
        } else if file_type.is_file() {
            let key = rel.to_string_lossy().replace('\\', "/");
            out.push(ObjectMeta {
                key,
                len: entry.metadata()?.len(),
            });
        }
    }
    out.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(out)
}
