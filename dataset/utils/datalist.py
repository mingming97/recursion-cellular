import pandas as pd

def datalist_from_file(label_file):
    data = pd.read_csv(label_file)
    datalist = []
    for index, row in data.iterrows():
        img_name = row['filename']
        sirna = row.get('sirna', None)
        datalist.append((img_name, sirna))
    return datalist
