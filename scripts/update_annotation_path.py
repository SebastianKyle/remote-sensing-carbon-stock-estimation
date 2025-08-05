import os
import json

def update_paths_in_json(file_path, old_base_path, new_base_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Update the paths
    data['rgb_path'] = data['rgb_path'].replace(old_base_path, new_base_path).replace('\\', '/')
    data['chm_path'] = data['chm_path'].replace(old_base_path, new_base_path).replace('\\', '/')

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def update_paths_in_directory(directory, old_base_path, new_base_path):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                update_paths_in_json(file_path, old_base_path, new_base_path)

if __name__ == '__main__':
    old_base_path = 'D:/Self_Practicing/Computer Vision/research/Experiments/src/'

    # Sửa thành đường dẫn tương ứng với máy của bạn
    new_base_path = '/media02/lqngoc22/thesis-tree-delineation/'                                                    # thư mục của source code tree_delineation
    annotations_directory = '/media02/lqngoc22/thesis-tree-delineation/data/preprocessed/test/annotations'          # thư mục chứa các file đánh nhãn tương ứng với tập train và test

    update_paths_in_directory(annotations_directory, old_base_path, new_base_path)
