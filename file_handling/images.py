import glob
import os
from shutil import copyfile
import tqdm

def make_nested_structure_for_files(file_path, glob_search):
    files = glob.glob(os.path.join(file_path, glob_search))
    for file in tqdm.tqdm(files):
        basename_file = os.path.basename(file)
        start_string = basename_file[:2]
        if not os.path.exists(os.path.join(file_path, start_string)):
            os.mkdir(os.path.join(file_path, start_string))
        copyfile(src=file, dst=os.path.join(file_path, start_string, basename_file))


if __name__ == '__main__':
    make_nested_structure_for_files(file_path='data/train', glob_search='*.jpg')