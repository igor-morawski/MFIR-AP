import os
def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def ensure_parent_exists(path):
    ensure_path_exists(os.path.dirname(path))