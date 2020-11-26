
import matplotlib.pyplot as plt
import warnings
import numpy as np
from sklearn import datasets, preprocessing, model_selection, svm, metrics
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


#필요한 함수 정의
def filter_column(df, column, border_low, border_high):
    filter_mask = (df[column] >= border_low) & (df[column] < border_high)

    filtered_df = df[filter_mask].reset_index(drop=True)
    return filtered_df

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#데이터
data = pd.read_csv('C:/Users/KIM/Desktop/GraduateSchool/예측모델/팀플/팀플 데이터/in_total.csv',index_col=0)

#train, test 나누기
train_x= data[data.index < '2020-01-01']
test_x = data[data.index >= '2020-01-01']

#scaling
def scaling(x):
    x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    return x
train_x = scaling(train_x)
test_x = scaling(test_x)

#y설정
train_y = np.sin(train_x).values.ravel()
test_y = np.sin(test_x).values.ravel()

# 모델

models = [
    svm.SVR(kernel='linear', C=1, max_iter=1000000),
    svm.SVR(kernel="poly", C=1, degree=3, epsilon=0.1,max_iter=1000000),
    svm.SVR(kernel='rbf', C=1,max_iter=1000000),
]

for model in models:
    # 학습
    print("\nmodel={}".format(model))
    model.fit(train_x, train_y)

    if model == models[0]:
        # 평가
        pred_y_lin = model.predict(test_x)

        r_square = metrics.r2_score(test_y, pred_y_lin)
        mse = mean_squared_error(test_y, pred_y_lin, squared = False)
        #rmse

        print("liner_r_square={:.5f}".format(r_square))
        print("liner_mse={:.5f}".format(mse))


    if model == models[1]:
        # 평가
        pred_y_poly = model.predict(test_x)

        r_square = metrics.r2_score(test_y, pred_y_poly)
        mse = mean_squared_error(test_y, pred_y_poly, squared = False)

        #rmse

        print("poly_r_square={:.5f}".format(r_square))
        print("poly_mse={:.5f}".format(mse))


    if model == models[2]:
        # 평가
        pred_y_rbf = model.predict(test_x)

        r_square = metrics.r2_score(test_y, pred_y_rbf)
        mse = mean_squared_error(test_y, pred_y_rbf, squared=False)
        # rmse

        print("poly_r_square={:.5f}".format(r_square))
        print("poly_mse={:.5f}".format(mse))



#plot
plt.scatter(test_x, test_y, label='test', c='g')

plt.scatter(test_x, pred_y_lin, label='predict_lin', c='r', alpha=0.5)
plt.scatter(test_x, pred_y_rbf, label='predict_rbf', c='b', alpha=0.5)
plt.scatter(test_x, pred_y_poly, label='predict_poly', c='c', alpha=0.5)

plt.legend()
plt.show()

##########################################################################################################################################################
#grid search로도 실험
param = {'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),'C' : [1,5,10, 50, 100],'degree' : [2,3,4],'coef0' : [0.01,10,0.5],'gamma' : ['scale'], 'max_iter': [1000000]}

modelsvr = SVR()

grids = GridSearchCV(modelsvr,param,cv=5)
grids
grids.fit(train_x,train_y)
print('best_params:', grids.best_params_) #best_params: {'C': 5, 'coef0': 0.01, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly', 'max_iter': 1000000}

model = svm.SVR(kernel="poly", C=5, degree=3, epsilon=0.1,max_iter=1000000, coef0=0.01, gamma = 'scale')
model.fit(train_x, train_y)
pred_y_lin = model.predict(test_x)

r_square = metrics.r2_score(test_y, pred_y_lin)
mse = mean_squared_error(test_y, pred_y_lin, squared = False)

print("liner_r_square={:.5f}".format(r_square))
print("liner_mse={:.5f}".format(mse))

plt.scatter(test_x, test_y, label='test', c='g')
plt.scatter(test_x, pred_y_lin, label='predict', c='r', alpha=0.5)

plt.legend()
plt.show()