import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sas7bdat import SAS7BDAT
pd.options.mode.chained_assignment = None 
import seaborn as sns
import pyodbc

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100



def try_int(value):
    try:
        return int(value)
    except:
        return value
    
def read_from_sas(filename, data_directory):
    """Return a dataframe from given SAS table."""
    with SAS7BDAT(data_directory + filename + ".sas7bdat") as f:
        df = f.to_data_frame()
    return df

def round_correct(value, decimal = 0):
    """
    Round a pandas series the same way that SAS and R do
    Use with pandas: some_series.apply(round_correct, args =(number_of_decimals_to_round_to,))
    """
    reg_round = round(value, decimal)
    rounding_error = .5 * (1/10**decimal)
    val_to_add = 1 * (1/10**decimal)
    if value >= 0:
        if round(value - reg_round, decimal + 3) == rounding_error:
            return reg_round + val_to_add
        else:
            return reg_round
    else:
        if round(value - reg_round, decimal + 3) == -rounding_error:
            return reg_round - val_to_add
        else:
            return reg_round

def round_all_columns(df, n_decimals):
    for col in df.columns:
        try:
            df[col]= df[col].apply(round_correct, args = (n_decimals,))
        except:
            pass    
    return df


def get_district(dbn):
	"""Get leading two numbers of DBN as district"""
	return dbn[0:2]

def get_bn(dbn):
	"""Get trailing 4 numbers of DBN as bn"""
	return dbn[2:]

def get_borough(dbn):
	"""Get borough from dbn"""
	return dbn[2]

def create_z_score(list_of_vals):
    """Calculate z-scores for a given list"""
    group_mean = np.nanmean(list_of_vals)
    group_std = np.nanstd(list_of_vals)
    zs = [abs((x-group_mean)/group_std) for x in list_of_vals]
    return zs

def fix_ascii_col_names(df):
    """Columns from SQL sometimes start with weird characters - this fixes them"""
    df.columns = [x.encode('ascii', 'ignore').decode("UTF-8") for x in df.columns]
    
def create_bn_col(df, dbn_col = "dbn"):
    """Create bn column from given dbn column"""
    df['bn'] = df[dbn_col].apply(lambda x: x[2:])


def percentile_rank_groups_10000(df, cols_to_groupby, col_to_rank, method_to_rank = "average"):
    """
    Convert numerical column to percentile rank in the same method that SAS uses
    See http://support.sas.com/documentation/cdl/en/proc/61895/HTML/default/viewer.htm#a000146840.htm
    """
    df['raw_rank'] = df[[col_to_rank] + cols_to_groupby].groupby(cols_to_groupby).rank(method = method_to_rank)
    df['groupby_size'] = df.groupby(cols_to_groupby)[col_to_rank].transform('size')
    df['per_rank'] = ((df.raw_rank*100)/(df.groupby_size + 1)).apply(round_correct, args = (1,))


def autolabel(rects, ax, decimal = 1, percentage = 1, additional_character = "%"):
    """Use to label bar plots
    rects = ax.patches"""
    for rect in rects:
        height = rect.get_height()
        height_val = round_correct(rect.get_height()*percentage,decimal)
        s = '%1.0f' % float(height_val)
        ax.text(rect.get_x() + (rect.get_width()/2), height,
                s = str(s) + additional_character,
                ha='center', va='bottom', fontsize = 10)

def to_percent(y, position):
    """
    Usage:
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    """
    s = str(100 * y)
    s = '%1.0f' % float(s)
    return str(s) + '%'

def return_dups(dataframe, variable):
    return dataframe[dataframe[variable].isin(dataframe[dataframe[variable].duplicated()][variable])]

def get_first_non_null(x):
    """
    Get first non null value from row
    Usage: df[cols].apply(get_first_non_null, axis = 1)
    """
    if x.first_valid_index() is None:
        return None
    else:
        return x[x.first_valid_index()]
    
def get_last_non_null(x):
    """
    Get last non null value from row
    Usage: df[cols].apply(get_last_non_null, axis = 1)
    """
    if x.last_valid_index() is None:
        return None
    else:
        return x[x.last_valid_index()]

# Use to display multiple output lines in one cell
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

def print_sprint_ids(ids):
    for id in ids:
        print("'" + str(id) + "',")

#RPSG public spreadsheet columns names
city_columns = ['grade', 'year', 'category', 'n_tested', 'scale_score', 'level_1_n', 'level_1_per', 'level_2_n', 'level_2_per',
                       'level_3_n', 'level_3_per','level_4_n', 'level_4_per', 'level_34_n', 'level_34_per']

# 2016 SQR public workbook summary pages
hs_2016_sqr = pd.read_excel("http://schools.nyc.gov/NR/rdonlyres/32595FE4-15E0-4DFE-A4F4-2D9AD32ED6D1/0/2015_2016_HS_SQR_Results_2016_11_15.xlsx", sheetname = None)
ems_2016_sqr = pd.read_excel("http://schools.nyc.gov/NR/rdonlyres/B3F6B2AC-DE2D-4F5E-8F62-A11EBF3090EC/0/2015_2016_EMS_SQR_Results_2016_11_15.xlsx", sheetname = None)
hst_2016_sqr = pd.read_excel("http://schools.nyc.gov/NR/rdonlyres/2E919AAB-D033-45C4-8CDB-F0D3EF9FACD1/0/2015_2016_HST_SQR_Results_2016_11_16.xlsx", sheetname = None)
yabc_2016_sqr = pd.read_excel("http://schools.nyc.gov/NR/rdonlyres/9857774C-1EFB-4519-9A5D-D7869298DCBD/0/2015_2016_YABC_SQR_Results_2016_11_16.xlsx", sheetname = None)
ec_2016_sqr = pd.read_excel("http://schools.nyc.gov/NR/rdonlyres/EB047F9D-E0B3-48EF-BB85-1C6B03844BFB/0/2015_2016_EC_SQR_Results_2016_11_16.xlsx", sheetname = None)
d75_2016_sqr = pd.read_excel("http://schools.nyc.gov/NR/rdonlyres/25C17A3B-93B8-40FD-9F88-A4E7A81AA545/0/2015_2016_D75_SQR_Results_2016_11_16.xlsx", sheetname = None)

def grouped_weighted_avg(values, weights, by):
    "Usage: grouped_weighted_avg(values=df[values_col], weights=df[weight_col], by=df[grouped_col])"
    return (values * weights).groupby(by).sum() / weights.groupby(by).sum()

def kappa(cm):
    """
    Compares your classifier with a random classifier that predicts the classes as
    often as your classifier does.
    Its values range from -1 to 1.
    If its value is positive, your classifier is doing better than chance.
    If its value is negative, then your classifier is doing worse than chance.
    """
    num_classes = len(cm)
    sum_all = 0
    sum_diag = 0
    sum_rands = 0
    for i in range(0, num_classes):
        sum_diag = sum_diag + cm[i, i]
        sum_col = 0
        sum_row = 0
        for j in range(0, num_classes):
            sum_col = sum_col + cm[j, i]
            sum_row = sum_row + cm[i, j]
            sum_all = sum_all + cm[i, j]
        sum_rands = sum_rands + sum_row * sum_col
    acc = sum_diag * 1.0 / sum_all
    rand = sum_rands * 1.0 / (sum_all * sum_all)
    return (acc - rand) / (1 - rand)

# SQL server connections
conn_27 = pyodbc.connect(r'DRIVER={SQL Server};SERVER=MTSQLVS27\MTSQLINS27;Trusted_Connection=yes;')
conn_ats = pyodbc.connect(r'DRIVER={SQL Server};SERVER=ES11vSINFAG02,4433;Trusted_Connection=yes;')