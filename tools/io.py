import os


def get_all_files(path: str) -> list[(str, str)]:
    file_paths = []

    for foldername, subfolders, filenames in os.walk(path):
        for filename in filenames:
            # Construct the full path to the file
            file_path = os.path.join(foldername, filename)
            file_paths.append((file_path, filename))

    return file_paths
