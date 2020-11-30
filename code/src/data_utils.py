import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

## 데이터를 pickle 형태로 저장
def save_pickle(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)

## pickle 형태의 데이터를 불러오기
def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

## 데이터를 불러올 때 index로 불러오기
def make_data_idx(dates, window_size=1):
    input_idx = []
    for idx in range(window_size-1, len(dates)):
        cur_date = dates[idx].to_pydatetime()
        in_date = dates[idx - (window_size-1)].to_pydatetime()

        _in_period = (cur_date - in_date).days * 24 * 60 + (cur_date - in_date).seconds / 60

        if _in_period == (window_size-1):
            input_idx.append(list(range(idx - window_size+1, idx+1)))
    return input_idx

def load_train_valid(file_path, valid_portion=0.2, shuffle=True, window_size=1):
    ## 파일이 있는지 확인.
    assert os.path.isfile(file_path), "[{}] 파일이 없습니다.".format(file_path)

    ## 파일 불러오기
    if '.p' in file_path:
        df = load_pickle(file_path)
    if '.csv' in file_path:
        df = pd.read_csv(file_path)
    if '.xlsx' in file_path:
        df = pd.read_excel(file_path)


    train_df, valid_df = train_test_split(df, test_size=valid_portion, shuffle=shuffle)
    return SVRDataset(train_df, window_size), SVRDataset(valid_df, window_size)

def load_test(file_path, window_size=1):
    ## 파일이 있는지 확인.
    assert os.path.isfile(file_path), "[{}] 파일이 없습니다.".format(file_path)

    ## 파일 불러오기
    if '.p' in file_path:
        df = load_pickle(file_path)
    if '.csv' in file_path:
        df = pd.read_csv(file_path)
    if '.xlsx' in file_path:
        df = pd.read_excel(file_path)
    return SVRDataset(df, window_size)


class SVRDataset():
    def __init__(self, df, window_size=1):
        self.window_size = window_size

        ## 파일 유효성 확인(Nan 확인)
        assert not df.isnull().values.any(), 'Dataframe 내 NaN이 포함되어 있습니다.'

        ## Index 추출
        df['date']=pd.to_datetime(df['date'], format='%Y-%m-%d')
        dates = list(df['date'])
        input_ids = make_data_idx(dates, window_size=window_size)

        ## 선택된 변수 Column Float으로 변경
        selected_column = []
        for var_name in df.columns.tolist():
            if 'date' not in var_name and 'count' not in var_name:
                df[var_name] = pd.to_numeric(df[var_name], errors='coerce')
                selected_column.append(var_name)

        var_data = []
        reference_data = df[selected_column].values
        for ids in input_ids:
            temp_data = reference_data[ids].reshape(-1)
            var_data.append(temp_data)

        label_data = df['count'].values

        ## Summary 용
        self.df = df.iloc[np.array(input_ids)[:, -1]]
        self.selected_column = selected_column

        self.var_data = np.array(var_data)
        self.label_data = label_data

    def get_data(self):
        return self.var_data

    def get_label(self):
        return self.label_data


## 결과를 저장하는 파일
class ResultWriter:
    def __init__(self, directory):
        """ Save training Summary to .csv
        input
            args: training args
            results: training results (dict)
                - results should contain a key name 'val_loss'
        """
        self.dir = directory
        self.hparams = None
        self.load()
        self.writer = dict()

    def update(self, args, **results):
        now = datetime.now()
        date = "%s-%s %s:%s" % (now.month, now.day, now.hour, now.minute)
        self.writer.update({"date": date})
        self.writer.update(results)
        self.writer.update(vars(args))

        if self.hparams is None:
            self.hparams = pd.DataFrame(self.writer, index=[0])
        else:
            self.hparams = self.hparams.append(self.writer, ignore_index=True)
        self.save()

    def save(self):
        assert self.hparams is not None
        self.hparams.to_csv(self.dir, index=False)

    def load(self):
        path = os.path.split(self.dir)[0]
        if not os.path.exists(path):
            os.makedirs(path)
            self.hparams = None
        elif os.path.exists(self.dir):
            self.hparams = pd.read_csv(self.dir)
        else:
            self.hparams = None




