import time
import traceback
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Generator, Iterable, List, Optional, Type, Union


def iter_batch(iterable: Union[Iterable, Generator], size: int) -> Iterable:
    """Iterate over an iterable in batches of specified size.

    Args:
        iterable: The input iterable to be batched
        size: Size of each batch

    Returns:
        An iterable of batches (as lists)

    Example:
        >>> list(iter_batch([1,2,3,4,5], 3))
        [[1, 2, 3], [4, 5]]
    """
    source_iter = iter(iterable)
    while source_iter:
        b = list(islice(source_iter, size))
        if len(b) == 0:
            break
        yield b


def get_tqdm_iterable(items: Iterable, show_progress: bool, desc: str) -> Iterable:
    """Optionally wrap an iterable with tqdm for progress tracking.

    Args:
        items: The iterable to potentially wrap with tqdm
        show_progress: Whether to show progress bar
        desc: Description for the progress bar

    Returns:
        The original iterable or a tqdm-wrapped version
    """
    _iterator = items
    if show_progress:
        try:
            from tqdm import tqdm

            return tqdm(items, desc=desc)
        except ImportError:
            pass
    return _iterator
