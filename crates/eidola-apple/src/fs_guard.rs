use std::fs;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum GuardError {
    #[error("path `{0}` is a symbolic link")]
    Symlink(PathBuf),
    #[error("path `{0}` is not a directory")]
    NotDirectory(PathBuf),
    #[error("path `{0}` does not exist")]
    Missing(PathBuf),
    #[error("path `{0}` is neither a regular file nor a directory")]
    NotRegular(PathBuf),
    #[error("failed to inspect `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub(crate) fn regular_files(root_path: &Path) -> Result<Vec<(PathBuf, PathBuf)>, GuardError> {
    root(root_path)?;
    let mut files = Vec::new();
    collect_regular_files(root_path, Path::new(""), &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(files)
}

fn collect_regular_files(
    root_path: &Path,
    relative_dir: &Path,
    files: &mut Vec<(PathBuf, PathBuf)>,
) -> Result<(), GuardError> {
    let directory = if relative_dir.as_os_str().is_empty() {
        root_path.to_path_buf()
    } else {
        existing(root_path, relative_dir)?
    };
    let entries = fs::read_dir(&directory).map_err(|source| GuardError::Io {
        path: directory.clone(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| GuardError::Io {
            path: directory.clone(),
            source,
        })?;
        let relative = relative_dir.join(entry.file_name());
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(|source| GuardError::Io {
            path: path.clone(),
            source,
        })?;
        if metadata.file_type().is_symlink() {
            return Err(GuardError::Symlink(path));
        }
        if metadata.is_dir() {
            collect_regular_files(root_path, &relative, files)?;
        } else if metadata.is_file() {
            files.push((relative, path));
        } else {
            return Err(GuardError::NotRegular(path));
        }
    }
    Ok(())
}

pub(crate) fn existing(root: &Path, relative: &Path) -> Result<PathBuf, GuardError> {
    walk(root, relative, false)
}

pub(crate) fn for_write(root: &Path, relative: &Path) -> Result<PathBuf, GuardError> {
    walk(root, relative, true)
}

pub(crate) fn root(root: &Path) -> Result<(), GuardError> {
    let metadata = fs::symlink_metadata(root).map_err(|source| GuardError::Io {
        path: root.to_path_buf(),
        source,
    })?;
    if metadata.file_type().is_symlink() {
        return Err(GuardError::Symlink(root.to_path_buf()));
    }
    if !metadata.is_dir() {
        return Err(GuardError::NotDirectory(root.to_path_buf()));
    }
    Ok(())
}

pub(crate) fn optional_existing(
    root: &Path,
    relative: &Path,
) -> Result<Option<PathBuf>, GuardError> {
    root_metadata(root)?;
    match walk_components(root, relative, true)? {
        WalkResult::Present(path) => Ok(Some(path)),
        WalkResult::Missing => Ok(None),
    }
}

fn walk(root_path: &Path, relative: &Path, allow_missing: bool) -> Result<PathBuf, GuardError> {
    root_metadata(root_path)?;
    match walk_components(root_path, relative, allow_missing)? {
        WalkResult::Present(path) => Ok(path),
        WalkResult::Missing => Ok(root_path.join(relative)),
    }
}

fn root_metadata(root_path: &Path) -> Result<(), GuardError> {
    root(root_path)
}

enum WalkResult {
    Present(PathBuf),
    Missing,
}

fn walk_components(
    root_path: &Path,
    relative: &Path,
    allow_missing: bool,
) -> Result<WalkResult, GuardError> {
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(GuardError::Missing(root_path.join(relative)));
    }
    let components: Vec<_> = relative.components().collect();
    let mut current = root_path.to_path_buf();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(name) = component else {
            unreachable!("components were validated above")
        };
        current.push(name);
        let metadata = match fs::symlink_metadata(&current) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && allow_missing => {
                return Ok(WalkResult::Missing);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(GuardError::Missing(current));
            }
            Err(source) => {
                return Err(GuardError::Io {
                    path: current,
                    source,
                });
            }
        };
        if metadata.file_type().is_symlink() {
            return Err(GuardError::Symlink(current));
        }
        if index + 1 < components.len() && !metadata.is_dir() {
            return Err(GuardError::NotDirectory(current));
        }
    }
    Ok(WalkResult::Present(current))
}
