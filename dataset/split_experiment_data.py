import pandas as pd
import glob
import os
from PIL import Image

import torch
import torchvision.transforms.functional as F


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
		experiment_df.to_csv('./{}_{}.csv'.format(experiment, suffix), index=0)


def data_static(data_path, image_size=(512, 512), num_channels=6):
	mean_x = [torch.zeros(*image_size) for _ in range(num_channels)]
	mean_x2 = [torch.zeros(*image_size) for _ in range(num_channels)]
	count = 0

	for i in range(num_channels):
		channel_data_path = os.path.join(data_path, '*_w{}.png'.format(i + 1))
		image_path_list = glob.glob(channel_data_path)
		assert count == 0 or count == len(image_path_list)
		count = len(image_path_list)

		for image_path in image_path_list:
			image = Image.open(image_path)
			image = F.to_tensor(image).squeeze()

			mean_x[i] += image
			mean_x2[i] += torch.pow(image, 2)

	num_pixel = image_size[0] * image_size[1]
	mean_x = [torch.sum(x) / count / num_pixel for x in mean_x]
	mean_x2 = [torch.sum(x) / count / num_pixel for x in mean_x2]
	std_x = [(x2 - torch.pow(x, 2)).sqrt() for x2, x in zip(mean_x2, mean_x)]
	return mean_x, std_x, count


if __name__ == '__main__':
	get_split_data = True
	get_data_static = False
	get_experiment_data_static = False

	if get_split_data:
		train_csv = 'C:\\Users\\VI\\Desktop\\train.csv'
		test_csv = 'C:\\Users\\VI\\Desktop\\test.csv'

		split_experiment_data_list(train_csv, suffix='train')
		split_experiment_data_list(test_csv, suffix='test')

	data = {}
	if get_experiment_data_static:
		experiments = ['HEPG2', 'HUVEC', 'RPE', 'U2OS']
		data_dir = r'F:\xuhuanqiang\kaggle_recursion_cellular\recursion-cellular-dataset\train'

		for experiment in experiments:
			print("static {}: ".format(experiment))
			data_path = os.path.join(data_dir, '{}-*'.format(experiment), '*')
			mean, std, count = data_static(data_path)
			print('count:', count)
			print('mean: ', mean)
			print('std: ', std)
			data[experiment + '_mean'] = mean
			data[experiment + '_std'] = std

	if get_data_static:
		data_dir = r'F:\xuhuanqiang\kaggle_recursion_cellular\recursion-cellular-dataset\train\*\*'
		mean, std, count = data_static(data_dir)
		print('count:', count)
		print('mean: ', mean)
		print('std: ', std)
		data['mean'] = mean
		data['std'] = std

	torch.save(data, './data_static.pth')
