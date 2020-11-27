"""
#https://datascienceschool.net/03%20machine%20learning/20.01%20%ED%9E%88%EB%93%A0%20%EB%A7%88%EC%BD%94%ED%94%84%20%EB%AA%A8%ED%98%95.html
import numpy as np
from hmmlearn import hmm
np.random.seed(3)
import matplotlib.pyplot as plt

model = hmm.GaussianHMM(n_components=2, covariance_type="diag")
model.startprob_ = np.array([0.9, 0.1])
model.transmat_ = np.array([[0.95, 0.05], [0.15, 0.85]])
model.means_ = np.array([[1.0], [-3.0]])
model.covars_ = np.array([[15.0], [40.0]])
X, Z = model.sample(500)

#decode
model2 = hmm.GaussianHMM(n_components=2, n_iter=len(X)).fit(X)
Z_hat = model2.decode(X)[1]
X_cum = (1 + 0.01*X).cumprod()
X_cum_hat = X_cum.copy()
X_cum_hat[Z_hat == 0] = np.nan

plt.subplot(211)
plt.plot(X_cum, lw=5)
plt.plot(X_cum_hat, 'r-', lw=5)
plt.title("X cumulated")
plt.subplot(212)
plt.plot(Z, 'bo-')
plt.plot(Z_hat, 'ro-')
plt.title("discrete state")
plt.tight_layout()
plt.show()
"""

#https://stackoverflow.com/questions/45538826/decoding-sequences-in-a-gaussianhmm
import numpy as np
import pandas as pd

df = pd.read_csv('./팀플 데이터/in_total.csv', index_col='date')
n_feat = 3
df.head()
df.min() #9
df.max() #110377
df.mean() #29779

#train, test 나누기
train_x= df[df.index < '2020-01-01']
test_x = df[df.index >= '2020-01-01']

#fit 3-state HMM on training data
# hmm throws a lot of deprecation warnings, we'll suppress them.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    # in the most recent hmmlearn we can't import GaussianHMM directly anymore.
    from hmmlearn import hmm

mdl = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=1000)
mdl.fit(train_x)

#gird-search for optimal 6th (t+1)개의 observation
# length of O_t
span = 5

# final store of optimal configurations per O_t+1 sequence
best_per_span = []

# update these to demonstrate heterogenous outcomes
current_abc = None
current_pred = None

for start in range(len(test_x)-span):
    flag = False
    end = start + span
    first_five = test_x.iloc[start:end].values
    output = []
    for a in np.arange(10000,100000,10):
        sixth = np.array([a])[:, np.newaxis].T
        all_six = np.append(first_five, sixth, axis=0)
        output.append((mdl.decode(all_six), (a)))

    best = max(output, key=lambda x: x[0][0])

    best_dict = {"start":start,
                 "end":end,
                 "sixth":best[1],
                 "preds":best[0][1],
                 "lik":best[0][0]}
    best_per_span.append(best_dict)

    # below here is all reporting
    if best_dict["sixth"] != current_abc:
        current_abc = best_dict["sixth"]
        flag = True
        print("New abc for range {}:{} = {}".format(start, end, current_abc))

    if best_dict["preds"][-1] != current_pred:
        current_pred = best_dict["preds"][-1]
        flag = True
        print("New pred for 6th position: {}".format(current_pred))

    if flag:
        print("Test sequence StartIx: {}, EndIx: {}".format(start, end))
        print("Best 6th value: {}".format(best_dict["sixth"]))
        print("Predicted hidden state sequence: {}".format(best_dict["preds"]))
        print("Likelihood: {}\n".format(best_dict["lik"]))
 
#predict 된 값들
final_sixth =[]
for i in range(len(best_per_span)):
    final_sixth.append(best_per_span[i]["sixth"])

final_sixth