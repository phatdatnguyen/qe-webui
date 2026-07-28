import os

def get_files_in_working_directory(working_directory_path):
    files = [f for f in os.listdir(working_directory_path) if not f.endswith('Zone.Identifier')]
    return files

