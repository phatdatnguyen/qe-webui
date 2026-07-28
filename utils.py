import os

def get_files_in_working_directory(working_directory_path):
    # Guard against a missing/None path (e.g. an error path before a working
    # directory is opened): os.listdir(None) would list the server's cwd.
    if not working_directory_path or not os.path.isdir(working_directory_path):
        return []
    files = [f for f in os.listdir(working_directory_path) if not f.endswith('Zone.Identifier')]
    return files

