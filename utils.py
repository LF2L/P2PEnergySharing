import sys
import pandas as pd


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def calcCost(x, **kwargs):
    """
    :param x: one solution vector
    :param kwargs1:
    :return: real: total cost induced by this solution vector
    """
    # print(x)
    s = 0
    for i in range(len(x)):

        if x[i] >= 0:
            s = s + x[i] * kwargs["gridPrices"][i]
        else:
            s = s + x[i] * kwargs["FIT"][i]

    return (s)

def calcREgeneration(GHIfilepath, SParea, SPefficiency):
    """
    :param SPefficiency: efficiency of solar panel
    :param SParea: area of solar panel
    :param GHIfilepath: file path to GHI data: pd.DataFrame [PeriodEnd,PeriodStart,GHI]
    :return: list
    """
    df = pd.read_csv(GHIfilepath)
    GHIlist = df["GHI"].tolist()
    return [ghi*SParea*SPefficiency for ghi in GHIlist]