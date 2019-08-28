import os
from functools import reduce
import pandas as pd
from sklearn.utils import shuffle


def split_validate_data(csv_path, new_name, base_dir, train_part=9/10):
	df = pd.read_csv(csv_path)
	shuffle(df)

	num_train_data = int(len(df) * train_part)

	train_data_df = df[:num_train_data]
	val_data_df = df[num_train_data:]

	split_train_name = '{}_split_train.csv'.format(new_name)
	split_val_name = '{}_split_val.csv'.format(new_name)

	train_data_df.to_csv(os.path.join(base_dir, split_train_name), index=0)
	val_data_df.to_csv(os.path.join(base_dir, split_val_name), index=0)
	return split_train_name, split_val_name


def merge_data_list(csv_files, new_name, base_dir):
	dfs = [pd.read_csv(path) for path in csv_files]
	new_df = reduce(lambda x, y: x.append(y), dfs)
	new_name = '{}.csv'.format(new_name)
	new_df.to_csv(os.path.join(base_dir, new_name), index=0)
	return new_name


if __name__ == '__main__':
	base_dir = r'D:\somethingElse\other_projects\recursion-cellular\dataset\csv_files'
	csv_files = ['U2OS_train.csv', 'RPE_train.csv', 'HUVEC_train.csv', 'HEPG2_train.csv']

	split_train_list = []
	split_val_list = []
	for path in csv_files:
		experiment = path.split('_')[0]
		split_train_name, split_val_name = split_validate_data(os.path.join(base_dir, path), experiment, base_dir)
		split_train_list.append(split_train_name)
		split_val_list.append(split_val_name)

	merge_data_list((os.path.join(base_dir, path) for path in split_train_list), 'split_train', base_dir)
	merge_data_list((os.path.join(base_dir, path) for path in split_val_list), 'split_val', base_dir)
