import os
import re

# Characters not allowed in a file name / directory name / QE prefix.
_INVALID_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sort_by_name(names):
    """Sort names case-insensitively (ties broken by the raw name for stability)."""
    return sorted(names, key=lambda n: (n.lower(), n))


def get_files_in_working_directory(working_directory_path):
    # Guard against a missing/None path (e.g. an error path before a working
    # directory is opened): os.listdir(None) would list the server's cwd.
    if not working_directory_path or not os.path.isdir(working_directory_path):
        return []
    files = [f for f in os.listdir(working_directory_path) if not f.endswith('Zone.Identifier')]
    # Sorted by name: this list feeds every file dropdown in the app, and
    # os.listdir() order is arbitrary.
    return sort_by_name(files)


def validate_name(name, kind):
    """Return an error string if name is empty/has invalid characters, else None."""
    name = (name or "").strip()
    if not name:
        return f"Please provide a {kind}."
    if _INVALID_NAME.search(name) or name in (".", ".."):
        return (f"The {kind} {name!r} contains invalid characters "
                "(avoid / \\ : * ? \" < > | and control characters).")
    return None
