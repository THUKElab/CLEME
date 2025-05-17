import gzip
import os
from pathlib import Path
from typing import Generator, List, Optional


def concat_dirs(dirname: str, basename: str) -> str:
    """Append basename to dirname, avoiding backslashes when running on Windows.

    This function ensures proper path joining by adding a trailing slash to
    dirname if needed, then using os.path.join.

    Args:
        dirname: Directory path
        basename: Filename or subdirectory

    Returns:
        Properly joined path
    """
    dirname += "/" if dirname[-1] != "/" else ""
    return os.path.join(dirname, basename)


def smart_open(file: str, mode: str = "rt", encoding: str = "utf-8"):
    """Open a file for reading or writing, automatically handling gzip compression.

    Args:
        file: Path to the file
        mode: File mode (read, write)
        encoding: File encoding

    Returns:
        File handle for the opened file
    """
    if file.endswith(".gz"):
        return gzip.open(file, mode=mode, encoding=encoding, newline="\n")
    return open(file, mode=mode, encoding=encoding, newline="\n")


def add_files(
    dirpath: str, exclude: Optional[List] = None, recursive: bool = False, num_files_limit: Optional[int] = None
) -> List[Path]:
    """Find files in a directory, with options for exclusion and recursion.

    Args:
        dirpath: Directory path to search
        exclude: List of glob patterns to exclude
        recursive: Whether to search subdirectories
        num_files_limit: Maximum number of files to return

    Returns:
        List of Path objects for the found files

    Raises:
        ValueError: If no files are found
    """
    dirpath = Path(dirpath)
    all_files, rejected_files = set(), set()

    # Process exclusion patterns
    if exclude is not None:
        for excluded_pattern in exclude:
            if recursive:
                # Recursive glob
                for file in dirpath.rglob(excluded_pattern):
                    rejected_files.add(Path(file))
            else:
                # Non-recursive glob
                for file in dirpath.glob(excluded_pattern):
                    rejected_files.add(Path(file))

    # Find all files
    file_refs: Generator[Path, None, None]
    if recursive:
        file_refs = Path(dirpath).rglob("*")
    else:
        file_refs = Path(dirpath).glob("*")

    # Filter files
    for ref in file_refs:
        is_dir = ref.is_dir()
        skip_because_excluded = ref in rejected_files
        if not is_dir and not skip_because_excluded:
            all_files.add(ref)

    new_input_files = sorted(all_files)

    if len(new_input_files) == 0:
        raise ValueError(f"No files found in {dirpath}.")

    # Apply file limit if specified
    if num_files_limit is not None and num_files_limit > 0:
        new_input_files = new_input_files[0:num_files_limit]

    # print total number of files added
    print(f"> Total files added from {dirpath}: {len(new_input_files)}")
    return new_input_files
