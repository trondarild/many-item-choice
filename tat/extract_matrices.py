import pandas as pd
import numpy as np
import os
import re

''' 
Will produce preference and saliency matrices for all subject-setsize-trial combinations

To run:
cd <project dir>
python tat/extract_matrices.py
'''

def to_absolute_path(relative_path):
    return os.path.abspath(relative_path)

def read_gaze_data(subject, setsize):
    file_path = f'./data/subject_files/sub-{subject}_setsize-{setsize}_desc-gazes.csv'
    file_path = to_absolute_path(file_path)
    return pd.read_csv(file_path)

def process_item_stim_data(df):
    return df[['trial', 'item', 'stimulus', 'item_value']].drop_duplicates(subset=['trial', 'item', 'stimulus'])

def read_saliency_data():
    fname = to_absolute_path("./tat/img_saliency.csv")
    return pd.read_csv(fname)

def merge_with_saliency(item_stim_df, saliency_df):
    return pd.merge(item_stim_df, saliency_df, how='left', on='stimulus')

def create_matrix(group_df, value_column, output_dir, subject, setsize, trial, matrix_type):
    values = group_df.reset_index()[value_column].values
    matrix = np.zeros((1, int(setsize)))
    matrix.flat[:len(values)] = values
    matrix_str = f'OUTPUT/' + str(setsize) + '\n' + '\n'.join([' '.join(map(str, row)) for row in matrix])
    output_file_path = os.path.join(output_dir, f"s_{subject}_setsize_{setsize}_trial_{trial}_{matrix_type}.dat")
    # print(output_file_path)
    with open(output_file_path, 'w') as f:
        f.write(matrix_str)


def find_files_and_extract_info(directory):
    pattern = r"sub-(\d+)_setsize-(\d+)_desc-gazes.csv"
    subjects_setsizes = set()
    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            subject, setsize = match.groups()
            subjects_setsizes.add((subject, setsize))
    return subjects_setsizes
        


def main():
    directory = to_absolute_path("./data/subject_files/") # present working dir is project dir
    subjects_setsizes = find_files_and_extract_info(directory)
    
    # subjects_setsizes = [(1, 36)]  # This should be dynamically determined from filenames in the directory
    saliency_df = read_saliency_data()
    
    for subject, setsize in subjects_setsizes:
        df = read_gaze_data(subject, setsize)
        item_stim_df = process_item_stim_data(df)
        item_stim_saliency_df = merge_with_saliency(item_stim_df, saliency_df)
        
        for matrix_type, value_column, output_dir in [('preference', 'item_value', to_absolute_path('./data/preference_matrices')),
                                                      ('saliency', 'saliency', to_absolute_path('./data/salience_matrices'))]:
            os.makedirs(output_dir, exist_ok=True)
            for trial, group_df in item_stim_saliency_df.groupby('trial'):
                create_matrix(group_df, value_column, output_dir, subject, setsize, trial, matrix_type)
        print(f"Matrices for subject {subject} and setsize {setsize} have been generated.")

if __name__ == "__main__":
    main()
