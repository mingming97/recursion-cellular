import pandas as pd


def split_experiment_data_list(origin_path, suffix='train'):
	df = pd.read_csv(origin_path)
	new_dfs = dict()

	for i in range(len(df)): 
		row = df.loc[i]
		experiment = row['experiment'].split('-')[0]
		experiment_df = new_dfs.get(experiment, pd.DataFrame())
		experiment_df = experiment_df.append(row)
		new_dfs[experiment] = experiment_df

	for experiment, experiment_df in new_dfs.items():
		experiment_df['plate'] = experiment_df['plate'].astype(int)
		experiment_df.to_csv('./{}_{}.csv'.format(experiment, suffix))


if __name__ == '__main__':
	train_csv = 'C:\\Users\\VI\\Desktop\\train.csv'
	test_csv = 'C:\\Users\\VI\\Desktop\\test.csv'

	split_experiment_data_list(train_csv, suffix='train')
	split_experiment_data_list(test_csv, suffix='test')
 