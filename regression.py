import matplotlib.pyplot as plt
import pandas as pd

from dataVisualize import *
from toolbox import *
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

def linearRegressionFinal(x_train,y_train):
    x_train = sm.add_constant(x_train)
    # y_train = sm.add_constant()

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
# def backwardRegression(df):
#     #Y = df
#     Y = df[['hmdy(%)']]#.to_numpy()
#     X = df.drop(['hmdy(%)','hmax(%)','hmin'], axis=1)
#
#     X = sm.add_constant(X)
#     model_temp = sm.OLS(Y, X)
#     res = model_temp.fit()
#     print(res.summary())
#
#     #based on this we can drop wdir(deg)
#     model_temp = sm.OLS(Y, X)
#     res = model_temp.fit()
#     print(res.summary())
def featureRemovalVIF(train):
    train = train.loc[:, train.columns != 'hmdy(%)']
    vif_data = pd.DataFrame()

    vif_data["feature"] = train.columns
    vif_data["VIF"] = [variance_inflation_factor(train.values, i)
                       for i in range(len(train.columns))]

    vif = vif_data.sort_values(by=['VIF'])
    print(vif)
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


    #print(vif)
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
def featureSelection(train):
    #correlation
    plotCorrelation(train)


    #collinearity

    A = train.loc[:, train.columns != 'hmdy(%)'].to_numpy()
    H = A.T @ A
    # 3.a Performing SVD
    print(f"\n{'='*5}SVD{'='*5}\n")
    U, D, VT = np.linalg.svd(H)
    print(f"Sigular Values = {D}\n")
    # We have values close to 0 in SVD so we will remove features
    print(f"\n{'=' * 5}Condition Number{'=' * 5}\n")
    condition_number = LA.cond(A)
    print(f"Condition Number before feature removal is {condition_number}\n")

    #print(f"Since Condtion number > 1000, there is severe Collinearity in our data \n")

    print(f"\n{'=' * 5} VIF {'=' * 5}\n")
    #vif_features =
    # calculating VIF for each feature
    # VIF dataframe
    vif_features = featureRemovalVIF(train)
    print(vif_features)
    checkConditionNumber(train[vif_features])

    #backward Regression
    #backwardRegression(train)


def linearRegressionModel(train_all,test_all):

    train_st, test_st = standarize(train_all,test_all)



    featureSelection(train_st)
    features_vif = ['prcp(mm)','radi(KJ/m2)','wdir(deg)','dmax','tmin','atmmax','wdsp(m/s)','wgust(m/s)']
    y_train = train_st['hmdy(%)']
    x_train = train_st.drop(['hmdy(%)','hmax(%)','hmin'],axis = 1)
    #x_train = train_st[features_vif]
    x_train = x_train[features_vif]
    featureSelection(x_train)
    linear_model = linearRegressionFinal(x_train,y_train)
    # temp_x_train = x_train.copy()
    # y_train_temp = y_train.copy()
    x_test = test_st[features_vif]
    x_test = sm.add_constant(x_test)
    y_test = test_st['hmdy(%)']
    y_pred= linear_model.predict(x_test)

    plt.plot(y_train.index, y_train.values, label='Training Data')
    plt.plot(y_test.index, y_test.values, label='Actual Test Data')
    plt.plot(y_pred.index, y_pred.values, label='Forecast Using Linear Regression')
    plt.title("Linear Regression")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    #print(f"Model residual is {linear_model.resid}")
    residual = linear_model.resid

    mse = np.mean(residual ** 2)
    rmse = np.sqrt(mse)
    print(f"Linear Regression : \nMSE :{mse}\nRMSE :{rmse}")
    lags = 100
    acf_lr = autoCorrelationFunction(residual, lags=lags, title="Linear Regression Residual", axes=None)
    Q = len(y_train) * np.sum(np.square(acf_lr[lags + 1:]))
    print(f"Q value :{Q:.2f}")
    print(f"Variance of residual : {np.var(residual):.2f}\nMean of residual : {np.mean(residual):.2f}")