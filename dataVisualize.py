import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
def visualizeColumns(df):
    """
    help in visualizing the columns and there values.
    Humidity depends on temp and pressure.
    Same amount of water vapours results in higher relative humidity in cool air and warm air.
    #higher temp at the equator and the lowest at the poles, the temp decreases with the increase of latitude
    #temp decrease with height
    #various type of soil
    :return:
    """
    sns.violinplot(df[['atmp(mb)']])
    plt.show()
    sns.violinplot(df[['radi(KJ/m2)']])
    plt.show()
    sns.violinplot(df[['prcp(mm)','temp(C)','wgust(m/s)']])
    plt.show()
    sns.violinplot(df[['hmdy(%)']])
    plt.title('Relative Humidity(%')
    plt.show()
    sns.violinplot(df[['wdir(deg)']])
    plt.title('Wind Direction (0-360)')
    plt.show()
def plotCorrelation(df):
    plt.figure(figsize=(16,8))
    heatmap = sns.heatmap(df.corr(),vmin=-1,vmax=1,annot=True,cmap='BrBG')
    heatmap.set_title('Correlation Heatmap',fontdict={'fontsize':12})
    plt.show()

def plotMontly(df,var,title="Dummy"):
    monthly_df = df[var].resample('M').mean()
    monthly_df.plot()
    plt.title(title)
    plt.ylabel(var)
    plt.xlabel("Years")
    plt.tight_layout()
    plt.grid()
    plt.show()
def plotDaily(df,var,title='Dummy'):
    daily_df = df[var].resample('D').mean()
    daily_df.plot()
    plt.title(title)
    plt.ylabel(var)
    plt.xlabel("Years")
    plt.tight_layout()
    plt.grid()
    plt.show()