def read_labels_from_file(labels_file_path):
    with open(labels_file_path, "r") as f:
        return f.read().splitlines()
