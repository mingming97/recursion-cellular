import pandas as pd
import os


def datalist_from_file(label_file, data_mode='rgb'):
    assert data_mode in ('rgb', 'six_channels')
    data = pd.read_csv(label_file)
    datalist = []
    if data_mode == 'rgb':
        for index, row in data.iterrows():
            img_name = row['filename']
            sirna = row.get('sirna', None)
            if sirna is not None:
                sirna = int(sirna)
            datalist.append((img_name, sirna))
    else:
        for index, row in data.iterrows():
            experiment_index = row['experiment']
            plate_index = row['plate']
            well_index = row['well']
            sirna = row.get('sirna', None)
            if sirna is not None:
                sirna = int(sirna)
            img_name = row['filename']
            site = img_name.split('.')[0][-2:]

            img_path = os.path.join(experiment_index, 'Plate{}'.format(plate_index), '{}_{}_'.format(well_index, site))
            datalist.append((img_path, sirna))
    return datalist
