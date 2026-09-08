import os
import re
from collections import UserList

# Characters not allowed in a file name / directory name / QE prefix.
_INVALID_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class FileListSnapshot(UserList):
    """A fresh, list-compatible event value for every working-directory scan.

    Gradio hashes ordinary lists by their elements, suppressing State.change
    when a run or edit overwrites existing filenames. UserList is hashed by
    object identity by Gradio, so each new snapshot also signals those writes.
    Keep this a UserList: a list subclass would still be hashed by its contents.
    """


def sort_by_name(names):
    """Sort names case-insensitively (ties broken by the raw name for stability)."""
    return sorted(names, key=lambda n: (n.lower(), n))


def get_files_in_working_directory(working_directory_path):
    # Guard against a missing/None path (e.g. an error path before a working
    # directory is opened): os.listdir(None) would list the server's cwd.
    if not working_directory_path or not os.path.isdir(working_directory_path):
        return FileListSnapshot()
    files = [f for f in os.listdir(working_directory_path) if not f.endswith('Zone.Identifier')]
    # Sorted by name: this list feeds every file dropdown in the app, and
    # os.listdir() order is arbitrary.
    return FileListSnapshot(sort_by_name(files))


def validate_name(name, kind):
    """Return an error string if name is empty/has invalid characters, else None."""
    name = (name or "").strip()
    if not name:
        return f"Please provide a {kind}."
    if _INVALID_NAME.search(name) or name in (".", ".."):
        return (f"The {kind} {name!r} contains invalid characters "
                "(avoid / \\ : * ? \" < > | and control characters).")
    return None
