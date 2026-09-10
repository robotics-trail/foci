"""Filesystem locations that several entry points have to agree on."""

from __future__ import annotations

from pathlib import Path


# src/data, derived from this file's position inside the package.
#
# demos/ and src/benchmark/ used to compute this independently, both with
# `dirname(__file__)/".."` -- but from directories at different depths, so they
# resolved to <repo>/data and <repo>/src/data respectively.  Only one of the two
# could ever be right, and moving the dataset silently broke the other half of
# the scripts.  Deriving it from the package removes the choice.
DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"


def data_path(*parts: str) -> Path:
    """
    Absolute path to a file or directory under src/data.

    Parameters
    ----------
    *parts:
        Path components below the data directory, e.g. data_path("Forest.ply").

    Returns
    -------
    pathlib.Path

    Raises
    ------
    FileNotFoundError
        Naming both the missing path and the data directory it was resolved
        against.  The datasets are gitignored (*.ply), so a fresh clone hits
        this, and the message has to say where to put them rather than failing
        later inside the PLY reader.
    """
    path = DATA_DIR.joinpath(*parts)

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Expected it under the data directory "
            f"{DATA_DIR} (*.ply is gitignored, so datasets are not in the "
            "repository)."
        )

    return path
