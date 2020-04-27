"""Taken from: http://slazebni.cs.illinois.edu/fall18/assignment3_part2.html """
import os
import csv
import numpy as np

def write_csv(file_path, y_list):
    solution_rows = [('id', 'category')] + [(i, np.sqrt(1 - y)) for (i, y) in enumerate(y_list)]
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(solution_rows)

def output_submission_csv(output_file_path, y_test):
    write_csv(output_file_path, y_test)
