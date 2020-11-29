from sklearn.svm import SVR
import pandas as pd
import os
import numpy as np
import logging
from tqdm import tqdm, trange
from argparse import ArgumentParser
import random
from data_utils import load_train_valid, ResultWriter, load_test
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, output_notebook, show, save
from bokeh.models import BoxAnnotation, LinearAxis, Range1d

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = 'sklearn_model.bin'
CONFIG_NAME = "training_args.bin"


def save_model(save_path, temp_model):
    model_path = os.path.join(save_path, CHECKPOINT_NAME)
    with open(model_path, 'wb') as handler:
        pickle.dump(temp_model, handler)


def load_saved_model(load_path):
    model_path = os.path.join(load_path, CHECKPOINT_NAME)
    with open(model_path, 'rb') as handler:
        return pickle.load(handler)


def load_saved_config(load_path):
    config_path = os.path.join(load_path, CONFIG_NAME)
    with open(config_path, 'rb') as handler:
        return pickle.load(handler)


def save_config(save_path, temp_config):
    config_path = os.path.join(save_path, CONFIG_NAME)
    with open(config_path, 'wb') as handler:
        pickle.dump(temp_config, handler)


def plot_picture_ver2(dates, var_values, confidence_values, save_path, name_dict):
    fig = figure(title=name_dict['title'],
                 x_axis_label='Timeline',
                 x_axis_type='datetime',
                 y_axis_label='score',
                 plot_width=2000,
                 plot_height=500)

    fig.y_range = Range1d(start=min(min(var_values), min(confidence_values)), end=max(max(var_values), max(confidence_values)))
    fig.line(dates, var_values, line_width=2, color=name_dict['var_color'], legend_label=name_dict['var_name'])
    fig.line(dates, confidence_values, line_width=2, color=name_dict['confidence_color'], legend_label=name_dict['confidence_name'])

    fig.legend.click_policy = 'hide'
    output_file(filename=save_path)
    save(fig)


def plot_picture(dates, states, var_values, confidence_values, save_path, name_dict, s_size=0.1):
    ## Fig 생성
    fig = figure(title=name_dict['title'],
                 x_axis_label='Timeline',
                 x_axis_type='datetime',
                 y_axis_label='score',
                 plot_width=2000,
                 plot_height=500)

    fig.y_range = Range1d(start=min(var_values), end=max(var_values))
    fig.line(dates, var_values, line_width=2, color=name_dict['var_color'], legend_label=name_dict['var_name'])

    if states is not None and len(dates) > 0:
        temp_start = dates[0]
        temp_state = states[0]

        temp_date = dates[0]
        for xc, value in zip(dates, states):
            if temp_state != value:
                if temp_state == 'prognosis':
                    fig.add_layout(BoxAnnotation(left=temp_start, right=temp_date, fill_alpha=0.2, fill_color='blue'))
                if temp_state == 'abnormal':
                    fig.add_layout(BoxAnnotation(left=temp_start, right=temp_date, fill_alpha=0.2, fill_color='orange'))
                temp_start = xc
                temp_state = value
            temp_date = xc

        if temp_state == 'prognosis':
            fig.add_layout(BoxAnnotation(left=temp_start, right=xc, fill_alpha=0.2, fill_color='blue'))
        if temp_state == 'abnormal':
            fig.add_layout(BoxAnnotation(left=temp_start, right=xc, fill_alpha=0.2, fill_color='orange'))

    if confidence_values is not None:
        fig.extra_y_ranges = {"var": Range1d(start=-1, end=max(confidence_values)+1)}
        fig.add_layout(LinearAxis(y_range_name="var"), 'right')
        fig.line(dates, confidence_values, legend_label=name_dict['confidence_name'], line_width=2, y_range_name='var',
                 color=name_dict['confidence_color'], line_alpha=.3)

    fig.legend.click_policy = 'hide'
    output_file(filename=save_path)
    save(fig)


## SEED 설정
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)


def build_parser():
    parser = ArgumentParser()

    ## 모델 type
    parser.add_argument("--model_type", default="svm", type=str, help="svm, svdd 등 모델의 이름을 입력하세요.")

    ## 모델 설정
    parser.add_argument("--window_size", default=1, type=int, help="확장을 고려한 변수입니다.(현재 사용중이지 않음)")
    parser.add_argument("--valid_portion", default=0.2, type=float, help="valid portion")

    parser.add_argument("--kernel", default='rbf', type=str,
                        help="kernel function(linear, poly, rbf, sigmoid, precomputed)")
    parser.add_argument("--gamma", default=1, type=float, help="float type")
    parser.add_argument("--c", default=0.5, type=float, help="float type [0~1]")
    parser.add_argument("--epsilon", default=0.5, type=float, help="epsilon")
    parser.add_argument("--degree", default=3, type=int, help="Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.")

    ## 데이터 위치
    parser.add_argument("--train_file", default="data/20190512_20190515/train.p", type=str,
                        help='train에 사용할 pickle, csv, xlsx 데이터를 선택해 주세요.')
    parser.add_argument("--test_file", default="data/20190512_20190515/all.p", type=str,
                        help='test에 사용할 pickle, csv, xlsx 데이터를 선택해 주세요.')

    parser.add_argument("--do_train", action="store_true", help='학습하려면 true로 변경하세요')
    parser.add_argument("--do_eval", action="store_true", help='평가하려면 true로 변경하세요')

    parser.add_argument("--do_summary", action="store_true", help='summary를 만드려면 true로 변경하세요')
    parser.add_argument("--summary_dir", default='summary', type=str, help='summary를 저장 폴더를 지정하세요')
    parser.add_argument("--with_csv", action="store_true", help='summary csv파일을 만드려면 true로 변경하세요')

    ## 평가표
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default='experiments/experiment.csv',
        help="학습 및 평가한 내용에 대해 Summary를 제공하고 있습니다. Summary파일의 이름을 입력하세요.",
    )

    ## 저장폴더 위치
    parser.add_argument("--pre_trained_dir", type=str, help="이전에 학습한 모델이 저장되어 있는 폴더를 입력하세요")

    ## 저장폴더 위치
    parser.add_argument("--output_dir", default="results", type=str, help="학습 후 최종 결과물이 저장될 위치를 입력하세요.")
    ## 저장폴더에 새로운 파일 overwirte 여부
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="학습 후 최종 결과물이 저장될 위치에 파일이 있을 때 덮어쓰기를 사용하시려면 true로 변경하세요")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    config = parser.parse_args()
    return config


def build_model(args):
    model_type = args.model_type.lower()
    model = None

    if model_type == 'svr':
        model = SVR(kernel=args.kernel, gamma=args.gamma, C=args.c, degree=args.degree, epsilon=args.epsilon)

    return model


## MAIN 실행
def main():
    def _print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))

    ## 설정 불러오기.
    args = build_parser()
    _print_config(args)

    ## 유효성 검사
    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "결과를 저장할 폴더에 ({}) 이미 다른 파일이 있습니다. 이 폴더에 저장하려면 overwrite_output_dir 설정을 true로 변경하시기 바랍니다.".format(
                args.output_dir
            )
        )
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if args.do_train:
        if args.train_file is None:
            raise ValueError(
                "학습할 파일을 입력하세요."
            )

    if args.do_eval:
        if args.test_file is None:
            raise ValueError(
                "평가할 파일을 입력하시오."
            )

    if args.do_train is None and args.do_eval:
        if args.pre_trained_dir is None:
            raise ValueError(
                "평가만을 하려면 학습된 모델을 넣으세요."
            )

    if args.do_train is None and args.do_summary:
        if args.pre_trained_dir is None:
            raise ValueError(
                "SUMMARY만을 하려면 학습된 모델을 넣으세요."
            )

    if args.do_summary:
        if args.summary_dir is None or args.test_file is None:
            raise ValueError(
                "SUMMARY를 만드려면 SUMMARY를 저장할 위치와 평가할 파일을 입력하세요."
            )
        if not os.path.isdir(args.summary_dir):
            os.mkdir(args.summary_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set seed
    set_seed(args)

    ########################### Load Model ##########################
    logger.info("Load model %s", args.model_type)
    model = build_model(args)

    results = {}
    writer = ResultWriter(args.experiments_dir)
    ## 학습 설정
    if args.do_train:
        logger.info("train model %s", args.model_type)
        ## 학습용 데이터 불러오기
        train_dataset, valid_dataset = load_train_valid(
            file_path=args.train_file,
            valid_portion=args.valid_portion,
            shuffle=True,
            window_size=args.window_size,
        )

        ## 학습시작
        now = datetime.now()
        model.fit(train_dataset.get_data(), train_dataset.get_label())
        elapsed = datetime.now() - now
        time_taken = str(timedelta(seconds=elapsed.total_seconds()))

        ## 추론 시작
        val_predicted_label = model.predict(valid_dataset.get_data())
        val_true_label = valid_dataset.get_label()
        val_mse = mean_squared_error(val_true_label, val_predicted_label)
        val_mae = mean_absolute_error(val_true_label, val_predicted_label)

        train_predicted_label = model.predict(train_dataset.get_data())
        train_true_label = train_dataset.get_label()
        train_mse = mean_squared_error(train_true_label, train_predicted_label)
        train_mae = mean_absolute_error(train_true_label, train_predicted_label)

        ## Summay 파일에 정보를 저장하기
        results.update(
            {
                'train_mae': train_mae,
                'train_mse': train_mse,
                'val_mae': val_mae,
                'val_mse': val_mse,
                'training time': time_taken,
            }
        )

        ## 모델 저장
        save_model(args.output_dir, model)
        ## Config 저장
        save_config(args.output_dir, args)

        args.pre_trained_dir = args.output_dir

    # 평가 설정
    if args.do_eval:
        ## 최종 학습된 모델 설정 불러오기
        saved_config = load_saved_config(args.pre_trained_dir)
        ## 최종 학습된 모델 설정 불러오기
        model = load_saved_model(args.pre_trained_dir)

        logger.info("test model %s", saved_config.model_type)

        ## 평가용 데이터 불러오기
        test_dataset = load_train_valid(
            file_path=saved_config.test_file,
            window_size=saved_config.window_size,
        )

        ## 추론 시작
        test_predicted_label = model.predict(test_dataset.get_data())
        test_true_label = test_dataset.get_label()
        test_mse = mean_squared_error(test_true_label, test_predicted_label)
        test_mae = mean_absolute_error(test_true_label, test_predicted_label)

        ## Summay 파일에 정보를 저장하기
        results.update(
            {
                'test_mse': test_mse,
                'test_mae': test_mae,
            }
        )
    writer.update(args, **results)

    if args.do_summary:
        ## 최종 학습된 모델 설정 불러오기
        saved_config = load_saved_config(args.pre_trained_dir)

        ## 최종 학습된 모델 설정 불러오기
        model = load_saved_model(args.pre_trained_dir)

        logger.info("make summary of model %s", saved_config.model_type)
        ## 평가용 데이터 불러오기
        test_dataset = load_test(
            file_path=saved_config.test_file,
            window_size=saved_config.window_size,
        )

        ## 추론 시작
        test_predicted_label = model.predict(test_dataset.get_data())
        test_true_label = test_dataset.get_label()
        test_mse = mean_squared_error(test_true_label, test_predicted_label)
        test_mae = mean_absolute_error(test_true_label, test_predicted_label)

        ## SUMMARY 그림 만들기
        summary_df = test_dataset.df
        summary_df['predicted_label'] = test_predicted_label

        name_dict = {
            "var_name": "abnormal_score",
            "confidence_name": "abnormal class",
            'title': "MSE:[{}]  MAE:[{}]".format(test_mse, test_mae),
            "var_color": 'black',
            "confidence_color": 'red',
            "var_plot": 'line',
            "confidence_plot": 'line',
        }

        ## 그림 생성
        save_name = "abnormal_score.html"
        temp_save_path = os.path.join(args.summary_dir, save_name)
        # plot_picture(summary_df['date'].values, None, summary_df['count'].values, summary_df['predicted_label'].values,
        #              temp_save_path, name_dict)
        plot_picture_ver2(summary_df['date'].values, summary_df['count'].values, summary_df['predicted_label'].values, temp_save_path, name_dict)

        if args.with_csv:
            save_name = "summary.csv"
            temp_save_path = os.path.join(args.summary_dir, save_name)
            summary_df.to_csv(temp_save_path, index=False)


if __name__ == "__main__":
    main()