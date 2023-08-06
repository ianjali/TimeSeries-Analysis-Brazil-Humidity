import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from numpy import linalg as LA
from numpy import log
import seaborn as sns
from scipy import signal
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
#sys.append('/Users/anjalimudgal/Desktop/GWU/TimeSeries/FinalToolbox/toolbox.py')

import matplotlib.pyplot as plt
from scipy.stats import chi2
np.random.seed(6313)

# pd.set_option('float_format', '{:.2f}'.format)
#
# np.set_printoptions(precision=2,  # limit to two decimal places in printing numpy
#                     suppress=True,  # suppress scientific notation in printing numpy
#                    )
# pd.option_context('display.precision', 3)

# seasonal_order = (NA, D, NB, S)
#
# model = ARIMA(data, order=(1,1,1), trend='n')
# model_fit = model.fit()
#
# # Make predictions
# predictions = model_fit.predict(start='2022-01-01', end='2022-12-31')
#
# # Print predictions
# print(predictions)

# from statsmodels.tsa.statespace.sarimax import SARIMAX
#
# # load the data
# data = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')
#
# # fit the SARIMA model
# model = SARIMAX(data, order=(1,1,1), seasonal_order=(1,1,1,12))
# model_fit = model.fit()
#
# # make predictions
# predictions = model_fit.predict(start='2023-01-01', end='2023-12-01')
#
# # print the predictions
# print(predictions)
# ===============================
# from sktime.forecasting.arima import AutoARIMA
# forecaster = AutoARIMA(start_p=0, # na min
#                        max_p=40,    # na max
#                        start_q=0,   # nb min
#                        max_q=40,    # nb max
#                        start_P=0,  # NA min
#                        max_P=40,    # NA max
#                        start_Q=0,   # NB min
#                        max_Q=40,    # NB max
#                        seasonal=True,
#                        max_d=5,     # diff
#                        max_D=5,      # diff
#                        stationary=True,
#                        n_fits=20,
#                        stepwise=False)
# forecaster = forecaster.fit(y_train)
# print(forecaster.summary())
# This will not give satisfactory result for seasonal data
# ===============================

# from scipy.stats import chi2_contingency
#
# # Create a contingency table
# observed = np.array([[10, 20, 30],
#                      [6,  9,  17],
#                      [12, 16, 28]])
#
# # Perform chi-square test

# chi2, p_value, dof, expected = chi2_contingency(observed)
#
# # Print results
# print("Chi-square statistic:", chi2)
# print("P-value:", p_value)
# print("Degrees of freedom:", dof)
# print("Expected frequencies:\n", expected)

# from lifelines import KaplanMeierFitter
# import matplotlib.pyplot as plt
#
# ax = plt.subplot(111)
# durations = [5,6,6,2.5,4,4]
# event = [1,0,0,1,1,1]
# kmf = KaplanMeierFitter()
# kmf.fit(durations,event, label = 'Number of users stay on a website')
# kmf.plot_survival_function(ax = ax)
# plt.show()

#sarima na,d,nb,s
#q < q* -> white
#linearRegressionTests(x_train,y_train)
#holtWinterTemp(s,train,test,title="Holt Winter Prediction")
#sse(e) : e.T @ e
#padNumDen(num,den)
#generate_e_theta(y,theta,na,nb)
#lm(y,e_theta,theta,n_a,n_b)
#plotSSE(itr,sse_all)
#ACF_PACF_Plot(y,lags)
#calcGPAC(ry,j=7,k=7 )
#stationarityCheck(y_list,title="")
#generateWhiteNoise(mean = 0,std=1,num_sample=1000)
#Cal_rolling_mean_var(data, name)
#differencing(series,seasonality,order)
#readDateData(url)
#flatForecast(train,test,method="Naive",title="dummy variables",axes=None)
#plotAllForecast(train,test,title="dummy variables")
#simpleExponential(l0, train,test,alpha=0.5,title="Dummy",axes=None)
#computeError(train_df,test_df,orignal_col,title ="method")

#checkConditionNumber(df) - svd + condition

#train_df['Predicted E'] = np.array([train_df['log_1order'] - train_df['oneStep'] for i in range()])
#computeError(train_df,test_df,orignal_col,title ="method") - residual error
#calculateQ(acf,T)
#splitData(df
#findAverageForMA(series,m)
#movingAverageClassical(m,series,m_f=0)
#classicalDecomposition(m,series,m_f)
#decomposition(series,period=7)
#adjustTrend(series,trend,title="Detrended")
#adjustSeasonal(series,seasonal,title=" ")
#autoCorrelationFunction(series,lags,title="dummy variable",axes=None)
#removeNone(y_list)
#getData(url,y_label = ' ',title =' ',x_label='Date')
#plotOriginalData(df,y_label = ' ',title =' ',x_label = 'Date')
##plotHistogram(series)
def confidenceInterval(theta_new,cov_theta,n_a,n_b):
    conf_int = []
    estimated_ar = [1]
    estimated_ma = [1]
    for i in range(0, n_a):
        temp_minus = theta_new[i][0] - 2*np.sqrt(cov_theta[i][i])
        temp_plus = theta_new[i][0] + 2 * np.sqrt(cov_theta[i][i])
        estimated_ar.append(theta_new[i][0])
        print(f"{temp_minus} < a{i+1} < {temp_plus}")
    for j in range(n_a, n_a + n_b):
        temp_minus = theta_new[j][0] - 2 * np.sqrt(cov_theta[j][j])
        temp_plus = theta_new[j][0] + 2 * np.sqrt(cov_theta[j][j])
        estimated_ma.append(theta_new[j][0])
        print(f"{temp_minus} < b{j - n_a + 1 } < {temp_plus}")
    # for i in range(0,len(theta_new)):
    #     temp_minus = theta_new[i][0] - 2*np.sqrt(cov_theta[i][i])
    #     temp_plus = theta_new[i][0] + 2*np.sqrt(cov_theta[i][i])
    #     conf_int.append([temp_minus, temp_plus])
        #print(f"{temp_minus} < ")
    #df = pd.DataFrame(columns=[theta_new,conf_int])
    #print(f"Dataframe is {df}")
    return conf_int,estimated_ar, estimated_ma


def singleACF(doubleacf):
    mid_ry = int(len(doubleacf)/2)
    return doubleacf[mid_ry,:]
def lm_predict(y,na,nb,lags=20):
    """

    :param na: order of ar exclude 1
    :param nb: order of ma exclude 1
    :return: model_hat,e,Q,model
    """
    np.random.seed(6313)
    #Hypothesis
    # H0: Null hypothesis: The residuals are uncorrelated, hence it is white
    # This can be shown when Q < Qc or when the p-value is higher than
    # the threshold
    # HA Alternative hypothesis: The residuals are correlated, hence
    # it is not white.This can be shown when Q > Qc or when
    # the p - value is lower than the threshold.
    lags = 20
    N = len(y)  # Number of samples
    # Generate ARMA Process
    #arma_process = sm.tsa.ArmaProcess(ar, ma)
    #print("Is this a stationary process : ", arma_process.isstationary)
    # Generate ARMA Process dataset
    #y = arma_process.generate_sample(N)
    y_var = np.var(y)
    print("Variance of y:", y_var)
    #change to your autocorrelation function
    ry2 = autoCorrelationFunction(y, lags, title="dummy data", axes=None)
    # Theoretical ACF
    #ry = arma_process.acf(lags=20)
    #ry1 = ry[:: -1]
    #ry2 = np.concatenate((np.reshape(ry1, lags), ry[1:]))
    # ====
    # GPAC Table
    # calcGPAC(ry2, 7, 7)
    # ====
    # ARMA parameter Estimation
    #model = sm.tsa.ARIMA(y, (na, nb)).fit(trend='nc', disp=0) # no trend

    model = sm.tsa.arima.ARIMA(y, order=(na, 0, nb), trend='n').fit()
    #model = sm.tsa.statespace.SARIMAX(y, order=(1, 0, 0), seasonal_order=(0, 0, 1, 365), trend='n')

    # for i in range(na):
    #     print("The AR coefficient af)".format(i), "is:", model.params[i])
    #     for i in range(nb):
    #         print("The MA coefficient b)".format(i), "is:", model.params[i + na])
    print(model.summary())
    # ===
    # Prediction
    model_hat = model.predict(start=0, end=N - 1)

    #forecast = model.predict(start=len(y), end=len(y) + n_periods - 1, dynamic=False)

    #y_hat = model.forecast(steps=len(test))
    # ==
    # Residuals Testing and Chi-Square test
    e = y - model_hat
    re = autoCorrelationFunction(pd.Series(e), lags, 'ACF of residuals')
    Q = len(y) * np.sum(np.square(re[lags+1:]))
    DOF = lags - na - nb
    alfa = 0.01
    chi_critical = chi2.ppf(1 - alfa, DOF)
    print(f"Q is {Q} and chi critical is {chi_critical}")
    if Q < chi_critical:
        print("The residual is white ")
    else:
        print("The residual is NOT white ")
    lbvalue, pvalue = sm.stats.acorr_ljungbox(e, lags=[lags])
    # print(lbvalue)
    # print(pvalue)
    plt.figure()
    plt.plot(y, 'r', label="True data")
    plt.plot(model_hat, 'b', label="Fitted data")
    plt.xlabel("Samples")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title(" Train versus One Step Prediction")
    plt.tight_layout()
    plt.show()
    return model_hat,e,Q,model


def scaleOriginal(data):
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    scaled_data = scaler.transform(data)

def linearRegressionTests(x_train,y_train):
    model_final = sm.OLS(y_train, x_train)
    res_final = model_final.fit()
    # Get the F-statistic and p-value for the model
    print(f"HO: The fit of the intercept-only model and your model are equal.")
    print(f"H1: The fit of the intercept-only model is significantly reduced compared to your model.")
    f_statistic = res_final.fvalue
    p_value = res_final.f_pvalue
    print(f"H0: Coefficient is 0 \nH1: Coefficient is not 0")
    print(f"F-statistic: {f_statistic}, p-value: {p_value}")
    print(f"\n{'=' * 10} T Test {'=' * 10}")
    t_test = res_final.tvalues
    t_test_col = []
    t_test_values = []
    t_test_pvalues = []
    print(f"H0: Coefficient is 0 \nH1: Coefficient is not 0")
    for i in t_test.index:
        t_test_col.append(i)
        t_test_values.append(t_test[i])
        t_test_pvalues.append(res_final.pvalues[i])

    t_test_diction = {'Coef': t_test_col, 'T-Test': t_test_values, 'pvalues': t_test_pvalues}
    t_test_df = pd.DataFrame(t_test_diction)
    print(t_test_df)
    return res_final
def holtWinterTemp(s,train,test,title="Holt Winter Prediction"):
    seasonal_period = s
    model = ExponentialSmoothing(train, seasonal_periods=seasonal_period,
                                 seasonal='add')
                                 #trend='add', seasonal='add')  # , damped=True)
    # Fit the model
    model_fit = model.fit()

    # Generate forecasts for a certain number of periods
    num_periods = len(test)
    forecast = model_fit.forecast(num_periods)
    # Plot the training data and the forecast
    plt.plot(train.index, train.values, label='Training Data')
    plt.plot(test.index, test.values, label='Actual Test Data')
    plt.plot(forecast.index, forecast.values, label='Forecast Using Mul')
    plt.title(title)
    plt.legend()
    plt.show()
def sse(e):
    return e.T @ e



# theta = np.zeros(n_a+n_b)
#
# #num_e = np.pad(num_e, (0, len(den_e) - len(num_e)), 'constant')
# num_e,den_e = padNumDen(num_e,den_e)
# system_e = (num_e,den_e,1)
#
# t,e_theta = signal.dlsim(system_e,y)

#
def padNumDen(num,den):
    if len(den) > len(num):
        num = np.pad(num, (0, len(den) - len(num)), 'constant')
    else:
        den = np.pad(den, (0, len(num) - len(den)), 'constant')
    return num,den

def generate_e_theta(y,theta,na,nb):
    num_e = [1]
    den_e = [1]
    theta = theta.ravel()
    for j in range(0, na):
        num_e.append(theta[j])
    for j in range(na, na+nb):
        den_e.append(theta[j])
    # Pad the numerator with zeros
    #num_e = np.pad(num_e, (0, len(den_e) - len(num_e)), 'constant')
    num_e, den_e = padNumDen(num_e, den_e)
    system_e = (num_e, den_e, 1)
    t,e_theta_i = signal.dlsim(system_e, y)
    return e_theta_i
def get_A_g_x_for_theta(y,e, theta,na, nb,  delta=1e-6):
    param = na+nb
    X =np .zeros((len(e),param))

    for i in range(0,param):
        temp = theta.copy()
        temp[i] = temp[i] + delta
        # num_e = [1]
        # den_e =[1]
        # for j in range(0,na):
        #     num_e.append(theta[j])
        # for j in range(0, nb):
        #     den_e.append(theta[j+na])
        # system_e = (num_e, den_e, 1)
        # e_theta_i = signal.dlsim(system_e, y)
        e_theta_i = generate_e_theta(y,temp,na,nb)
        x_i = (e-e_theta_i)/delta
        X[:,i]= x_i[:,0]

    A = X.T @ X
    g = X.T @ e
    return A,g,X
def updateTheta(y,A,grad,u_,na,nb,theta):
    #step 2
    identity_matrix = [[u_ if i == j else 0 for j in range(na+nb)] for i in range(na+nb)]
    delta_theta = LA.inv(A + identity_matrix) @ grad
    theta_new = theta.reshape(-1,1) + delta_theta
    e_theta_new = generate_e_theta(y,theta_new,na,nb)
    return theta_new,e_theta_new,delta_theta


# Hess, grad, X = get_A_g_x_for_theta(y,e_theta,theta, n_a, n_b)
# theta_new, e_theta_new, delta_theta = updateTheta(y,Hess,grad,u_, n_a,n_b, theta)
def plotSSE(itr,sse_all):
    #print(f"itr is {itr}")
    #print(f"sse is {sse_all}")
    plt.plot(itr,sse_all)
    plt.title("SSE versus iteration")
    plt.xlabel('Iteration')
    plt.ylabel('SSE')
    plt.xticks(np.arange(itr[0],itr[-1]+1,1))
    plt.tight_layout()
    plt.grid()
    plt.show()

def lm(y,e_theta,theta,n_a,n_b):
    samples = len(y)
    cov_theta = 0
    sigma_sq = 0
    u_ = 0.01
    umax = 10e+2
    max_iter = 50
    iteration = 0
    itr = []
    sse_loop = []
    theta_new = 0
    while iteration < max_iter:
        itr.append(iteration)
        sse_loop.append(sse(e_theta)[0][0])

        Hess, grad, X = get_A_g_x_for_theta(y, e_theta, theta, n_a, n_b)
        theta_new, e_theta_new, delta_theta = updateTheta(y, Hess, grad, u_, n_a, n_b, theta)



        if sse(e_theta_new) < sse(e_theta):
            if LA.norm(delta_theta) < 1e-3:  # ?
                theta = theta_new.copy()
                sigma_sq = sse(e_theta_new) / (samples - (n_a + n_b))
                cov_theta = sigma_sq * LA.inv(Hess)
                itr.append(iteration+1)
                sse_loop.append(sse(e_theta_new)[0][0])
                break
            else:
                # theta = theta_new.copy()
                u_ = u_ / 10

        while sse(e_theta_new) >= sse(e_theta):
            u_ = u_ * 10
            if u_ > umax:
                print(f"Error message: umax")
                break
            # Hess, grad, X = get_A_g_x_for_theta(y, e_theta, n_a, n_b)
            theta_new, e_theta_new, delta_theta = updateTheta(y, Hess, grad, u_, n_a, n_b, theta)
            # continue
        iteration = iteration + 1
        if iteration > max_iter:
            print(f"error message : iteration exceeded")
            break
        theta = theta_new.copy()
        e_theta = e_theta_new.copy()
        # Hess, grad, X = get_A_g_x_for_theta(y, e_theta_new, theta, n_a, n_b)
        # theta_new, e_theta_new, delta_theta = updateTheta(y, Hess, grad, u_, n_a, n_b, theta)

    plotSSE(itr,sse_loop)
    return theta_new, cov_theta, sigma_sq


# def generate_e_theta(theta,na,nb):
#     num_e = [1]
#     den_e = [1]
#     for j in range(0, na):
#         num_e.append(theta[j])
#     for j in range(0, nb):
#         den_e.append(theta[j + na])
#     system_e = (num_e, den_e, 1)
#     e_theta_i = signal.dlsim(system_e, y)
# def update_theta(e,param, na, nb, u_ = 0.01, delta=1e-6):
#     X =np.zeros((len(e),param))
#
#     for i in range(0,len(param)):
#         theta[i] = theta[i] + delta
#         # num_e = [1]
#         # den_e =[1]
#         # for j in range(0,na):
#         #     num_e.append(theta[j])
#         # for j in range(0, nb):
#         #     den_e.append(theta[j+na])
#         # system_e = (num_e, den_e, 1)
#         # e_theta_i = signal.dlsim(system_e, y)
#         e_theta_i = generate_e_theta(theta,na,nb)
#         x_i = (e-e_theta_i)/delta
#         X[:,i]= x_i
#     A = X.T @ X
#     g = X.T @ e
#
#     #step 2
#     identity_matrix = [[u_ if i == j else 0 for j in range(na+nb)] for i in range(na+nb)]
#     delta_theta = LA.inv(A + u_)
#     theta_new = theta + delta_theta
#     e_theta_new = generate_e_theta(theta_new,na,nb)
#     return  theta_new,e_theta_new


def ACF_PACF_Plot(y,lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()
def getPhHI(j,k,ry):
    """
    calculate
    :param j:
    :param k:
    :return:
    """
    # global numerator
    # global denom
    mid_ry = int(len(ry)/2)
    #for denominator
    den = []
    for row in range(0,k):
        temp_row = []
        index_col = row
        for col in range(0,k):
            temp_row.append(ry[((mid_ry + j )+index_col)-(col)])
            #print(f"col {(index_col)-(col)}")
        den.append(temp_row)
        #print(f"row = {row}")
    den = np.array(den)
    den = den.reshape(k,k)
    #print(f"denominator for (j={j}, k={k}) is \n{den}")

    #for numerator
    num = den.copy()
    for row in range(0,k):
        num[row][k-1]= ry[(mid_ry + j)+row+1]
    #print(f"numerator is \n{num}")
    if (np.linalg.det(den) == 0 and np.linalg.det(den) ==0):
        return np.nan
    if np.linalg.det(den) == 0:
        return np.inf
    #if (j==4 and k == 3):
    #    numerator = num
    #    denom = den
    #    print(f"num is \n{num}\n num det is {np.linalg.det(num)} \nden det is {np.linalg.det(den)}\nden is \n{den}\n and phi is :\n{np.linalg.det(num)/np.linalg.det(den)}")
    return np.linalg.det(num)/np.linalg.det(den)
def calcGPAC(ry,j=7,k=7 ):
    phi_mat = np.zeros((j,k))
    for j_index in range(0,j):
        for k_index in range(1,k+1):
            phi_mat[j_index][k_index-1] = getPhHI(j_index,k_index,ry)

    #print(phi_mat)
    row_index = list(range(1,k+1))
    index_col = list(range(0,j))

    df = pd.DataFrame(phi_mat, index= index_col, columns=row_index)
    #df = df.round(decimals = 3)
    plt.figure(figsize=(16, 8))
    sns.heatmap(df, annot=True,linewidths=0.5,annot_kws={"weight": "bold"},fmt='0.3f')
    plt.title('Generalized Partial Autocorrelation(GPAC) Table',fontdict={'fontsize':12})
    plt.tight_layout()
    plt.show()
    print(df)


def stationarityCheck(y_list,title=""):
    """
    Do the following for checking stationarity
    Plot the rolling mean, rolling variance
    Do ADF Test
    Do KPSS Test
    :param y_list: list on which we need to perform
    :param title: title to the plots
    :return:
    """
    y_list = pd.Series(y_list)
    Cal_rolling_mean_var(pd.Series(y_list), title)
    print(f"\n  Applying \033[1m ADF test \033[0m \n")
    print(f"{'=' * 10}ADF{'=' * 10}")

    index = 0
    y_list = y_list.fillna(0)
    while(y_list[index] == 0):
        index+=1
    y_list = y_list[index:,]
    ADF_Cal(y_list)
    print(f"{'=' * 10}KPSS{'=' * 10}")
    print(f"\nApplying KPSS test \n")
    kpss_test(y_list)

#stationarityCheck(y_MA_1000,'MA(1000)')

def generateWhiteNoise(mean = 0,std=1,num_sample=1000):
    # white noise
    """

    :param mean: mean of random noise
    :param std: standard deviation of white noise
    :param num_sample: number of samples of white noise required
    :return: white noise array
    """
    np.random.seed(6313)
    e = np.random.normal(mean, std, num_sample)
    return e

def generateARData(num_samples, order, coef):
    """
    :param num_samples: total number of samples
    :param order: order of AR model
    :param coef: coefficient of AR equation : should be array with [a1,a2], y(t)+ a1*y(t-1) +a2*y(t-2) + ... = e(t) so
    :return: y(t) array
    """
    e = generateWhiteNoise(num_sample=num_samples)
    y_for = np.zeros(num_samples)
    y_for[0] = e[0]
    for initial_data_i in range(1,order): # i = 1
        y_for[initial_data_i] = e[initial_data_i] #y[1] = e[1]
        #print(f"for i in {initial_data_i}")
        for order_j in range(0,initial_data_i): #j = 0,
            #print(f"for j in {order_j}")
            y_for[initial_data_i] = y_for[initial_data_i]-(y_for[initial_data_i-order_j-1] * coef[order_j])
    for i in range(order, num_samples):
        y_for[i] = e[i]
        for j in range(0, order):
            y_for[i] = y_for[i] - (y_for[i-j-1] * coef[j])
    return y_for

# def generateARData(num_samples, order, coef):
#     e = generateWhiteNoise(num_sample=num_samples)
#     y_for = np.zeros(num_samples)
#     y_for[0] = e[0]
#     for initial_data_i in range(1,order): # i = 1
#         y_for[initial_data_i] = e[initial_data_i] #y[1] = e[1]
#         #print(f"for i in {initial_data_i}")
#         for order_j in range(0,initial_data_i): #j = 0,
#             #print(f"for j in {order_j}")
#             y_for[initial_data_i] = y_for[initial_data_i]-(y_for[initial_data_i-order_j-1] * coef[order_j])
#     for i in range(order, num_samples):
#         y_for[i] = e[i]
#         for j in range(0, order):
#             y_for[i] = y_for[i] - (y_for[i-j-1] * coef[j])
#     return y_for
def coefficientAR(num_samples,order,y_reg):
    """
    For AR process of order = order this function will return the coefficients
    :param num_samples: total number of values in sample
    :param order: order of AR model
    :param y_reg: time series
    :return: beta (coefficients)
    """
    T=num_samples - order - 1
    X = np.zeros([T + 1, order])
    Y = np.zeros([T + 1, 1])
    for i in range(0, T + 1):
        for j in range(1, order + 1):
            X[i][j - 1] = -y_reg[order - j + i]
        Y[i] = y_reg[order + i]
    beta = linearRegression(X, Y)
    print(f"Coefficient estimation for {num_samples} samples using normal equation are\n{beta}")
    return beta
def forEquation(num_samples, a1=-0.5,a2=-0.2):
    """
    Calculates y(t)-0.5*y(t-1)-0.2*y(t-2) = e(t)
    :param num_samples: total number of samples we need values for
    :param a1: coefficient of y(t-1)
    :param a2: coefficient of y(t-2)
    :return: y array
    """
    e = generateWhiteNoise(num_sample=num_samples)
    y_for = np.zeros(num_samples)
    for i in range(num_samples):
        if i == 0:
            y_for[0] = e[0]
        elif i == 1:
            y_for[i] = (-1*a1) * y_for[i - 1] + e[i]
        else:
            y_for[i] = ((-1*a1) * y_for[i - 1]) + ((-1*a2) * y_for[i - 2]) + e[i]
    return y_for
def readData(url):
    df= pd.read_csv(url)
    return df

def checkConditionNumber(df):
    A= df.to_numpy()
    H = A.T @ A
    U, D, VT = np.linalg.svd(H)
    print(f"\n{'=' * 5}Singular Values{'=' * 5}\n")
    print(f"Sigular Values = {D}\n")
    # We have values close to 0 in SVD so we will remove features
    print(f"\n{'=' * 5}Condition Number{'=' * 5}\n")
    condition_number = LA.cond(A)
    print(f"Condition Number  {condition_number}\n")
def linearRegression(X, Y):
    beta = ((LA.inv(X.T @ X)) @ X.T) @ Y
    return beta
# X_train = sm.add_constant(X)
# X_matrix_train = X_train.to_numpy()
# Y_train = st_train[['price']].to_numpy()
#beta = linearRegression(X_matrix_train, Y_train)
# def movingAverage(m,series,m_f=0):
#     #k=0
#     moving_avg = []
#     folding_avg = []
#     #k = (m-1)/2
#     moving_avg = findAverage(series,m)
#     print(f"Moving average is {moving_avg}")
#     if m%2 !=0:
#         return moving_avg
#     else:
#         folding_avg = findAverage(moving_avg,m_f)
#         return folding_avg
# #df['3 MA']=
# #new_df = df.iloc[:50]
# def findAverage(series,m):
#     #moving_avg =[]
#     k = int((m-1)/2)
#     lower_k =k
#     upper_k =k
#     length = series.shape[0]
#     moving_avg = np.empty(length)
#     moving_avg[:] = np.NaN
#     if m %2 ==0:
#         lower_k= k+1
#     st_index = 0
#     end_index = length - 1
#     while np.isnan(series[st_index]) == True:
#         st_index = st_index + 1
#     while np.isnan(series[end_index])  == True:
#         end_index = end_index - 1
#     for i in range(st_index+lower_k,end_index-upper_k+1):
#         sum = 0
#         for j in range(i-lower_k,i+upper_k+1):
#             sum = sum + series[j]
#         moving_avg[i]=sum/m
#     return moving_avg
def autoCorrelation(series,lags,title="dummy variable",show_plot=True):
    acf = []
    deno = 0
    mean = np.mean(series)
    #print(f"Mean of series is {mean} and series is \n{series}")
    for i in range(0,len(series)):
        deno = deno + ((series[i]- mean)** 2)
    #print(deno)
    for lag in range(0,lags+1):
        acf.append(0)
        for j in range(lag,len(series)):
            acf[lag] = acf[lag]+((series[j]-mean)*(series[j-lag]-mean))
        acf[lag] = acf[lag]/deno
    double_sided_acf = acf[1:].copy()
    double_sided_acf.reverse()
    double_sided_acf = double_sided_acf+acf

    # x = np.linspace(-lag,lag,2*lag+1)
    # insignificant_band = 1.96/(len(series)**0.5)
    # print(f"Insginificant band is from {insignificant_band*-1} to {insignificant_band}")
    # #plotting the points
    # markers,_,_=plt.stem(x,double_sided_acf)
    # plt.xlim(-lag-1, lag+1)
    # plt.xlabel("Lags")
    # plt.ylim(-1.2,1.2)
    # plt.ylabel("Autocorrelation")
    # plt.setp(markers,color='red',marker='o')
    # plt.axhspan(-1*insignificant_band,insignificant_band,alpha=0.2,color='blue')
    # plt1=plt
    # plt.tight_layout()
    # plt.title("Autocorrelation of "+title)
    if show_plot == True:
        plotAutocorrealtion(double_sided_acf,series,lag,title)
        #plt.show()

    return double_sided_acf
def plotAutocorrealtion(acf,series,lag,title):
    x = np.linspace(-lag, lag, 2 * lag + 1)
    insignificant_band = 1.96 / (len(series) ** 0.5)
    print(f"Insginificant band is from {insignificant_band * -1} to {insignificant_band}")
    # plotting the points
    markers, _, _ = plt.stem(x, acf)
    plt.xlim(-lag - 1, lag + 1)
    plt.xlabel("Lags")
    #plt.ylim(-1.2, 1.2)
    plt.ylabel("Autocorrelation")
    plt.setp(markers, color='red', marker='o')
    plt.axhspan(-1 * insignificant_band, insignificant_band, alpha=0.2, color='blue')
    plt1 = plt
    plt.title("Autocorrelation of " + title)
    plt.tight_layout()
    plt.show()

def Cal_rolling_mean_var(data, name):
    """
    This function plots the rolling mean and variance of the given series
    :param data: series
    :param name: the parameter we want to mention in the title
    :return: None
    """
    length = len(data)
    rolling_mean = []
    rolling_var = []
    samples = []
    for i in range(1,length):
        temp_data = data.head(i)
        samples.append(i)
        rolling_mean.append(temp_data.mean())
        rolling_var.append(temp_data.var())
        # print(temp_data)
    rolling_var[0]=0

    #print(f"Variance is {rolling_var}")
    plt.subplot(2, 1, 1)
    plt.plot(samples, rolling_mean, label='Varying Mean')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Mean - ' + name)
    plt.legend(loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(samples, rolling_var, label='Varying Variance')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Variance - ' + name)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def ADF_Cal(x):
    # Null Hypothesis(HO): Series is non - stationary or series has a unit root.
    # Alternate Hypothesis(HA): Series is stationary or series has no unit root.

    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    if result[1] < 0.05:
        print(f"The tested series is \033[1m stationary \033[0m since pvalue :{result[1]} < 0.05 ")
    else :
        print(f"The tested series is \033[1m non stationary \033[0m since pvalue :{result[1]} > 0.05 ")
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
def kpss_test(timeseries):
    # Null Hypothesis (HO): Series is trend stationary or series has no unit root.
    # Alternate Hypothesis(HA): Series is non-stationary or series has a unit root.
    print ('Results of \033[1m KPSS Test \033[0m :')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
    if(kpss_output['p-value'] < 0.05):
        print(f"p value is {kpss_output['p-value']} < 0.05 ; we can reject the null hypothesis")
        print("The series is \033[1m non stationary \033[0m")
    else :
        print(f"p value is {kpss_output['p-value']} > 0.05 ; we cannot reject the null hypothesis")
        print("The series is \033[1m stationary \033[0m")

def differencing(series,seasonality,order):
  #series = pd.Series(series)
  s=seasonality
  diff =[]
  diff.append(None)
  for i in range(1,len(series)):
      if(series[i]==None or (i-seasonality < 0 ) or series[i-seasonality] ==None):
          diff.append(None)
      else:
          diff.append((series[i]-series[i-seasonality]))
  if order == 1:
      diff_series = pd.Series(diff)
      #print(f"diff series is : {diff_series}")
      return np.array(diff_series)
  else:
      #print(f"diff is \n{diff}")
      return differencing(pd.Series(diff), 1, order-1)

# a) 1st order differencing
#first_diff=differencing(air_df['#Passengers'],1,1)
#third_diff=differencing(air_df['#Passengers'],1,3)

def logTransform(series):
    return log(series)

##reading data with date as index
# url="https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv"
# df=pd.read_csv(url,header=0,
#                    parse_dates=[0],
#                    index_col=0)
def readDateData(url):
    df = pd.read_csv(url, header=0,
                      parse_dates=[0],
                        index_col=0)
    return df


def standarize(train_all, test_all):
    for col in train_all.columns:
        mean = train_all[col].mean()
        deviation = train_all[col].std()
        train_all[col] = (train_all[col] - mean) / deviation
        test_all[col] = (test_all[col] - mean) / deviation
    # mean = np.std(train["hmdy(%)"])
    # std = np.std(train["hmdy(%)"])
    return train_all, test_all
#flat forecast
def plotAllForecast(train,test,title="dummy variables"):
    rows = 2
    cols = 2
    fig, axs = plt.subplots(nrows=rows, ncols=cols,figsize=(16, 8))
    methods = ["Naive","Average","Drift","SSE"]
    for row in range(rows):
        for col in range(cols):
            index = cols*row+col
            method = methods[index]
            if method == "SSE":
                l0 = train[0]
                simpleExponential(l0, train, test, alpha=0.5, title=method, axes = axs[row,col])
            else:
                flatForecast(train, test, method=method, title=method,axes = axs[row,col])
    fig.suptitle(f'Forecasting', fontsize=10)
    plt.tight_layout()
    plt.show()
# rows = 3
# cols = 2
# fig, axs = plt.subplots(nrows=rows, ncols=cols,figsize=(16, 8))
# for row in range(rows):
#     for col in range(cols):
#         index = cols*row+col
#         lag = 50
#         #acf = autoCorrelationFunction(df['#Passengers'], lag, axes=axs[row,col])
# fig.suptitle(f'Autocorrelation closing price stocks between ',fontsize = 10)
# plt.tight_layout()
# plt.show()

# n_one_step,n_h_step=flatForecast(train_set,test_set,method="Naive",title="dummy variables")
# naive_error,n_one_hat_e=computeError(train_set,n_one_step,test_set,n_h_step,title = "Naive")
# acf_naive=autoCorrelation(n_one_hat_e,5,"Naive Residual",False)
# q_value_naive = calculateQ(acf_naive,len(n_one_step))
def flatForecast(train,test,method="Naive",title="dummy variables",axes=None):
    """

    :param train:
    :param test:
    :param method:
    :param title:
    :return:
    """
    T = len(train)
    one_step = []
    #x_train =np.arange(1,T+1,1)
    h_step = []
    x_test =np.arange(T+1,T+len(test)+1,1)
    h = 0

    if method == "Average":
        one_step.append(None)
        for i in range(1,T):
            one_step.append(np.mean(train[0:i]))
        avg = np.mean(train[0:T])
        h_step = [avg] * len(test)

    elif method == "Naive":
        one_step.append(None)
        for i in range(1,T):
            one_step.append(train[i-1])
        h_step = [train[T-1]]*len(test)

    elif method == "Drift":
        one_step.append(None)
        one_step.append(None)
        for i in range(2,T):
            change = (train[i-1]-train[0])/(i-1)
            pred = train[i-1] + change
            one_step.append(pred)
        pred = (train[T-1] - train[0])/(T-1)
        for i in range(0,len(test)):
            h = x_test[i]-T
            change = h*pred
            h_step.append(train[T-1]+change)
    col_name = train.name
    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)
    train_df['oneStep'] = np.array(one_step)
    test_df['hStep'] = np.array(h_step)
    #x_onestep = x_train[2:]
    #fig, ax = plt.subplots()
    # print(method)
    # print(train_df)
    # print(h_step)
    computeError(train_df, test_df, col_name, title=method)
    if axes == None:
        fig, ax = plt.subplots()
    else :
        ax= axes
    ax.plot(train_df[col_name], label='Train Set')#,linestyle='--',dashes=(5, 10))
    #ax.plot(train_df['oneStep'], label='One Step')
    ax.plot(test_df[col_name], label='Test Set')
    ax.plot(test_df['hStep'], label='h step prediction')
    ax.set_title('Forecast '+str(method))
    ax.set_ylabel('Time series value')
    ax.set_xlabel('Dates')
    ax.grid()
    ax.legend()
    plt.tight_layout()
    # ax.plot(x_train, train, label='Train Set')
    # ax.plot(x_test, test, label='Test Set')
    # ax.plot(x_test,h_step,label = 'h step prediction')
    # ax.set_title('Forecast for '+title+" using "+method+" Prediction")
    # ax.set_ylabel('Time series value')
    # ax.set_xlabel('Sample')
    # ax.grid()
    #     # Add a legend to the plot
    # ax.legend()
        # Show the plot
    #plt.show()
    if axes == None:
        plt.show()
    return train_df,test_df

def simpleExponential(l0, train,test,alpha=0.5,title="SES",axes=None):
    T = len(train)
    one_step = []
    # #h_step =[]
    one_step.append(l0)
    for i in range(1,T):
         y_pred = (alpha*train[i-1]) + ((1-alpha)*one_step[i-1])
         one_step.append(y_pred)
    future_pred = (alpha*train[T-1]) + ((1-alpha)*one_step[T-1])
    h_step = [future_pred]*len(test)

    col_name = train.name
    train_df= pd.DataFrame(train)
    test_df = pd.DataFrame(test)

    one_step[0] = None
    one_step[1] = None
    train_df["oneStep"] = np.array(one_step)
    test_df["hStep"] = np.array(h_step)
    #print(f"train is\n{train_df}")
    #print(f"Test is \n{test_df}")
    computeError(train_df, test_df, col_name, title=title)
    if axes == None:
          fig, ax = plt.subplots()
    else:
          ax=axes
    # # Plot the first array
    # x_train =np.arange(1,T+1,1)
    # x_test =np.arange(T+1,T+len(test)+1,1)
    #x_onestep = x_train
    ax.plot(train_df[col_name], label='Train Set')
    #ax.plot(train_df['oneStep'], label='One Step')
    ax.plot(test_df[col_name], label='Test Set')
    ax.plot(test_df['hStep'], label='h step prediction')
    ax.set_title('Forecast SES:' + str(alpha))
    ax.set_ylabel('Time series value')
    ax.set_xlabel('Dates')
    ax.grid()
    ax.legend()
    plt.tight_layout()
    # ax.plot(x_train, train, label='Train Set')
    # ax.plot(x_test, test, label='Test Set')
    # ax.plot(x_test, h_step, label='h step prediction')
    # ax.set_title('Forecast for ' + title + str(alpha) +" using SES Prediction")
    # ax.set_ylabel('Time series value')
    # ax.set_xlabel('Sample')
    # ax.grid()
    # ax.legend()
    # plt.tight_layout()
    # Show the plot
    if axes == None:
         plt.show()
    #return train_df,test_df
    # return one_step[2:], h_step

def computeError(train_df,test_df,orignal_col,title ="method"):
    """
    :param train: train dataframe containing original col and one step
    :param test: test step dataframe containing original col and h step
    :param orignal_col: original col name
    :param title:
    :return:
    """
    #predicted/residual
    train_df['Predicted E'] = [train_df[orignal_col][i] - train_df['oneStep'][i] if train_df['oneStep'][i] != None
                               else None for i in range(0, len(train_df))]
    train_df['Predicted E2'] = [train_df['Predicted E'][i] **2 if train_df['Predicted E'][i]!= None
                               else None for i in range(0, len(train_df))]
    test_df['ForecastedE'] = [test_df[orignal_col][i] - test_df['hStep'][i] if test_df['hStep'][i] != None
                               else None for i in range(0, len(test_df))]
    test_df['ForecastedE2'] = [test_df['ForecastedE'] **2 if test_df['ForecastedE'][i]!= None
                               else None for i in range(0, len(test_df))]
    #print(train_df)
    print(f"For {title}")
    title_acf = "ACF of Residual for "+str(title)
    acf = autoCorrelationFunction(removeNone(train_df['Predicted E']),50,title=title_acf,axes=None)
    Q = calculateQ(acf, len(train_df))
    residual_error_2 = np.sum(train_df['Predicted E2'])
    mse_res = residual_error_2 / len(train_df['Predicted E2'])

    #mse_res = mean_squared_error(train_df['Predicted E2'], valid['Predictions'])
    print(f"MSE for residual error is {mse_res:.2f}")
    print(f"RMSE for residual error is {np.sqrt(mse_res):.2f}")
    print(f"Q value is : {Q}")
    #values = test_df['ForecastedE2'].values
    #mse_fore = np.mean(values)

    #mean_e2 = np.mean(test_df['ForecastedE2'])
    #mse_fore = forecast_error_2 / len(test_df['ForecastedE2'])
    #print(f"MSE for forecasted error is {mean_e2}\n")
    #print(f"RMSE for forecasted error is {np.sqrt(mean_e2)}\n")

# def computeError(train, one_step, test, h_step,title="method"):
#     #predicted/residual
#     predicted_e = [train[2:][i] - one_step[i] for i in range (len(one_step))]
#     predicted_e2 = [predicted_e[i] ** 2 for i in range(len(predicted_e))]
#     forecast_e = [test[i] - h_step[i] for i in range (len(h_step))]
#     forecast_e2= [forecast_e[i] ** 2 for i in range (len(forecast_e))]
#     print(f"Calculation of error and MSE for {title} method")
#     residual_error = pd.DataFrame(list(zip(train[2:],one_step, predicted_e,predicted_e2)),
#                       columns=['Train','Prediction', 'ResidualError','ErrorSquare'])
#     forecast_error= pd.DataFrame(list(zip(test,h_step, forecast_e,forecast_e2)),
#                       columns=['Test','Forecast', 'ForecastError','ErrorSquare'])
#
#     print(residual_error)
#     residual_error_2 = np.sum(predicted_e2)
#     mse_res = residual_error_2/len(predicted_e2)
#     print(f"MSE for predicted error is {mse_res}")
    #print(f"Sum of error square is {residual_error_2}")

def calculateQ(acf,T):
    sum = 0
    one_side = int(len(acf)/2)
    actual_acf = acf[one_side: ]
    #print(actual_acf)
    #print(acf[0:])
    for i in actual_acf[1:]:
        #print(f" i is {i} and i**2 is {i**2}")
        sum= sum+(i**2)
    Q = sum * T
    return Q
def splitData(df):
    training_data, testing_data = train_test_split(df, test_size=0.2, random_state=6313,shuffle=False)
    return training_data,testing_data

def findAverageForMA(series,m):
    #moving_avg =[]
    k = int((m-1)/2)
    lower_k =k
    upper_k =k
    length = series.shape[0]
    moving_avg = np.empty(length)
    moving_avg[:] = np.NaN
    if m %2 ==0:
        lower_k= k
        upper_k = k+1

    st_index = 0
    end_index = length - 1

    while np.isnan(series[st_index]) == True:
        st_index = st_index + 1
    while np.isnan(series[end_index])  == True:
        end_index = end_index - 1
    if m==2:
        lower_k = 1
        upper_k =0
    #print(f"Start index is {st_index} and end index is {end_index}")
    #print(f"Upper k is {upper_k} and lower is {lower_k}")
    #print(f"looping {st_index+lower_k} , {end_index-upper_k+1}")
    for i in range(st_index+lower_k,end_index-upper_k+1):
        sum = 0
        for j in range(i-lower_k,i+upper_k+1):
            sum = sum + series[j]
        moving_avg[i]=sum/m
    return moving_avg

def movingAverageClassical(m,series,m_f=0):
    moving_avg = findAverageForMA(series,m)
    #print(moving_avg)
    if m%2 !=0:
        return moving_avg
    else:
        folding_avg = findAverageForMA(moving_avg,m_f)
        return folding_avg
def classicalDecomposition(m,series,m_f):
    MA = movingAverageClassical(m, series, m_f)
    input_df = series.assign(Input=MA)
    #MA3 = movingAverage(3, new_df['Temp'], m_f)
    return input_df


def decomposition(series,period=7):
    # apply STL decomposition
    stl = STL(series, period=period)
    result = stl.fit()
    # extract the components
    seasonal = result.seasonal
    trend = result.trend
    residual = result.resid
    #strength
    # The strength of trend for this data set is
    seasonaly_adjusted = series - seasonal
    st = 1 - (np.var(residual) / np.var(seasonaly_adjusted))
    #print(f"{st}")
    strength_trend = np.max([0.0, st])
    print(f"{'='*5}Strength of seasonality and Trend{'='*5}")
    print(f"The strength of trend for this data set is {strength_trend}")
    trend_adjusted = seasonal + residual
    st_s = 1 - (np.var(residual) / np.var(trend_adjusted))
    #print(f"{st_s}")
    strength_seasonality = np.max([0.0, st_s])
    print(f"The strength of seasonality for this data set is {strength_seasonality}")

    # plot the components
    result.plot()
    plt.suptitle('Original Data - Decompostion')
    plt.tight_layout()
    plt.show()
    return seasonal,trend,residual

def adjustTrend(series,trend,title="Detrended"):
    detrended = series - trend  # - trend

    plt.plot(series, label='Original Data')
    plt.plot(detrended, label='Detrended Data')
    plt.legend(loc='lower right')
    plt.title('Detrended along with Original Data')
    plt.xlabel('Time')
    plt.ylabel(title)
    plt.grid()
    plt.show()

def adjustSeasonal(series,seasonal,title=" "):
    seasonaly_adjusted = series - seasonal  # - trend

    plt.plot(series, label='Original Data')
    plt.plot(seasonaly_adjusted, label='Seasonaly Adjusted Data')
    plt.legend(loc='lower right')
    plt.title('Seasonally Adjusted data along with Original Data')
    plt.xlabel('Time')
    plt.ylabel(title)
    plt.grid()
    plt.show()
    return seasonaly_adjusted
# rows = 3
# cols = 2
# fig, axs = plt.subplots(nrows=rows, ncols=cols,figsize=(16, 8))
# for row in range(rows):
#     for col in range(cols):
#         index = cols*row+col
#         lag = 50
#         #acf = autoCorrelationFunction(df['#Passengers'], lag, axes=axs[row,col])
# fig.suptitle(f'Autocorrelation closing price stocks between ',fontsize = 10)
# plt.tight_layout()
# plt.show()
#
# acf = autoCorrelationFunction(df['#Passengers'], lag)

def autoCorrelationFunction(series,lags,title="dummy variable",axes=None):
    acf = []
    deno = 0
    series = series.reset_index(drop=True)
    mean = np.mean(series)
    #print(f"Mean of series is {mean} and series is \n{series}")
    for i in range(0,len(series)):
        deno = deno + ((series[i]- mean)** 2)
    #print(deno)
    for lag in range(0,lags+1):
        acf.append(0)
        for j in range(lag,len(series)):
            acf[lag] = acf[lag]+((series[j]-mean)*(series[j-lag]-mean))
        acf[lag] = acf[lag]/deno
    double_sided_acf = acf[1:].copy()
    double_sided_acf.reverse()
    double_sided_acf = double_sided_acf+acf

    x = np.linspace(-lag,lag,2*lag+1)
    insignificant_band = 1.96/(len(series)**0.5)
    #print(f"Insginificant band is from {insignificant_band*-1} to {insignificant_band}")

    if axes == None:
        #fig, axes = plt.subplots(nrows=1, ncols=cols, figsize=(16, 8))
        markers, _, _ = plt.stem(x, double_sided_acf)
        plt.xlim(-lag - 1, lag + 1)
        plt.xlabel("Lags")
        # plt.ylim(-1.2, 1.2)
        plt.ylabel("Autocorrelation")
        plt.setp(markers, color='red', marker='o')
        plt.axhspan(-1 * insignificant_band, insignificant_band, alpha=0.2, color='blue')
        plt1 = plt
        plt.title("Autocorrelation of " + title)
        plt.tight_layout()
        plt.show()
        return double_sided_acf
    # #plotting the points
    markers,_,_=axes.stem(x,double_sided_acf)
    axes.set_xlim(-lag - 1, lag + 1)
    axes.set_xlabel("Lags")
    axes.set_ylim(0, 1.2)
    axes.set_ylabel("Autocorrelation")
    plt.setp(markers,color='red',marker='o')
    axes.axhspan(-1*insignificant_band,insignificant_band,alpha=0.2,color='blue')
    axes.set_title("Autocorrelation of "+title)
    return double_sided_acf
def logTransform(series):
    return np.log(series)

def removeNone(y_list):
    #index = 0
    y_list = pd.Series(y_list)
    y_list = y_list.dropna()
    # y_list = y_list.fillna(0)
    # while (y_list[index] == 0):
    #     index += 1
    # y_list = y_list[index:, ]
    return pd.Series(y_list)



def getData(url,y_label = ' ',title =' ',x_label='Date'):
    """
    get the data from url by parsing 0 as dates column and making it as index
    :param url: csv file url
    :param y_label: label on y axis
    :param title: title of the plot
    :return:
    """
    df = pd.read_csv(url, header=0,
                     parse_dates=[0],
                     index_col=0)
    # plotting a time series
    df.plot()
    plt.legend(loc='upper left')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.show()
    return df

def plotOriginalData(df,y_label = ' ',title =' ',x_label = 'Date'):
    df.plot()
    plt.legend(loc='upper left')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.show()

def plotHistogram(series):
    plt.hist(series, bins=30, density=True, alpha=0.5)
    # Set the axis labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Data')
    plt.tight_layout()
    # Show the plot
    plt.show()
def featureRemovalVIF(train):
    train = train.loc[:, train.columns != 'hmdy(%)']
    vif_data = pd.DataFrame()

    vif_data["feature"] = train.columns
    vif_data["VIF"] = [variance_inflation_factor(train.values, i)
                       for i in range(len(train.columns))]

    vif = vif_data.sort_values(by=['VIF'])

    while(vif['VIF'].iloc[-1] > 5):
        print(f"\nremoving {vif['VIF'].iloc[-1]}")
        train = train.loc[:, train.columns != vif['feature'].iloc[-1]]
        vif_data = pd.DataFrame()
        vif_data["feature"] = train.columns
        vif_data["VIF"] = [variance_inflation_factor(train.values, i)
                           for i in range(len(train.columns))]
        vif = vif_data.sort_values(by=['VIF'])
        print(vif)
    return vif['feature']
