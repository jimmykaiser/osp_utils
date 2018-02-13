# Libraries for data analysis
import pandas as pd
import numpy as np

# Libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Library for reading data from SAS
from sas7bdat import SAS7BDAT

# Library for connecting to SQL server
import pyodbc

# TO DO: Add utility for installing needed packages


######################################
### Working with Jupyter Notebooks ###
######################################

# Control warnings and display in notebooks
pd.options.mode.chained_assignment = None 
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

# Use to display multiple output lines in one cell
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

def hide_code_on_export():
    """Use before any code cells to hide code on export to HTML
    There will be a toggle button to show or hide the code
    Markdown cells will still be shown"""
    import IPython.core.display as di

    # This line will hide code by default when the notebook is exported as HTML
    di.display_html('<script>jQuery(function() {if (jQuery("body.notebook_app").length == 0) { jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});</script>', raw=True)

    # This line will add a button to toggle visibility of code blocks, for use with the HTML export version
    di.display_html('''<button onclick="jQuery('.input_area').toggle(); jQuery('.prompt').toggle();">Toggle code</button>''', raw=True)


#################################
### Reading data from SAS/SQL ###
#################################

# SQL server connections
conn_sprint = pyodbc.connect(r'DRIVER={SQL Server};SERVER=ES00VADOSQL001;Trusted_Connection=yes;')
conn_oadmint = pyodbc.connect(r'DRIVER={SQL Server};SERVER=ES00VADOSQL001;Trusted_Connection=yes;')
conn_ordint = pyodbc.connect(r'DRIVER={SQL Server};SERVER=ES11vADOSQL006,4433;Trusted_Connection=yes;')
conn_ats = pyodbc.connect(r'DRIVER={SQL Server};SERVER=ES11vSINFAG02,4433;Trusted_Connection=yes;')

def read_from_sas(filename):
    """Return a dataframe from given SAS table."""
    with SAS7BDAT(filename + ".sas7bdat") as f:
        df = f.to_data_frame()
    return df


def format_list_for_sql_query(list):
    return ", ".join("'{0}'".format(x) for x in list)


def get_school_names(year, db_connection = conn_oadmint):
    supertable_query = """SELECT [Location_Code]
              ,[System_Code]
              ,[Location_Name]
          FROM [OADM_INT].[dbo].[Location_Supertable1]
          where Fiscal_Year = """ + year + """
          and System_ID = 'ATS'"""
    school_names_df = pd.read_sql(supertable_query, con = db_connection)
    school_names_df.columns = ['bn', 'dbn', 'school_name']
    school_names_df.dbn = school_names_df.dbn.str.rstrip()
    school_names_df.bn = school_names_df.bn.str.rstrip()
    return school_names_df


def print_sprint_ids(ids):
    """Use if you want to quickly grab a few ids to look at in SQL"""
    for id in ids:
        print("'" + str(id) + "',")


###########################
### Working with Pandas ###
###########################

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


def flatten_column_names(df):
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    return df


#######################
### Formatting data ###
#######################

def round_correct(number, places=0):
    '''
    round_correct(number, places)

    example:

        >>> round_correct(2.55, 1) == 2.6
        True

    uses standard functions with no import to give "normal" behavior to 
    rounding so that trueround(2.5) == 3, trueround(3.5) == 4, 
    trueround(4.5) == 5, etc. Use with caution, however. This still has 
    the same problem with floating point math. The return object will 
    be type int if places=0 or a float if places=>1.

    number is the floating point number needed rounding

    places is the number of decimal places to round to with '0' as the
        default which will actually return our interger. Otherwise, a
        floating point will be returned to the given decimal place.

    Note:   Use trueround_precision() if true precision with
            floats is needed

    GPL 2.0
    copywrite by Narnie Harshoe <signupnarnie@gmail.com>
    '''
    place = 10**(places)
    rounded = (int(number*place + 0.5if number>=0 else -0.5))/place
    if rounded == int(rounded):
        rounded = int(rounded)
    return rounded


def round_all_columns(df, n_decimals):
    for col in df.columns:
        try:
            df[col]= df[col].apply(round_correct, args = (n_decimals,))
        except:
            pass    
    return df


def fix_ascii_col_names(df):
    """Columns from SQL sometimes start with weird characters - this fixes them"""
    df.columns = [x.encode('ascii', 'ignore').decode("UTF-8") for x in df.columns]
    

##########################
### Working with DBN's ###
##########################

def get_district(dbn):
	"""Get leading two numbers of DBN as district"""
	return dbn[0:2]


def get_bn(dbn):
	"""Get trailing 4 numbers of DBN as bn"""
	return dbn[2:]


def get_borough(dbn):
	"""Get borough from dbn"""
	return dbn[2]


def create_bn_col(df, dbn_col = "dbn"):
    """Create bn column from given dbn column"""
    df['bn'] = df[dbn_col].apply(lambda x: x[2:])


##################################
### Math/statistical functions ###
##################################

def create_z_score(list_of_vals):
    """Calculate z-scores for a given list"""
    group_mean = np.nanmean(list_of_vals)
    group_std = np.nanstd(list_of_vals)
    zs = [abs((x-group_mean)/group_std) for x in list_of_vals]
    return zs


def percentile_rank_groups_10000(df, cols_to_groupby, col_to_rank, method_to_rank = "average"):
    """
    Convert numerical column to percentile rank in the same method that SAS uses
    See http://support.sas.com/documentation/cdl/en/proc/61895/HTML/default/viewer.htm#a000146840.htm
    """
    df['raw_rank'] = df[[col_to_rank] + cols_to_groupby].groupby(cols_to_groupby).rank(method = method_to_rank)
    df['groupby_size'] = df.groupby(cols_to_groupby)[col_to_rank].transform('size')
    df['per_rank'] = ((df.raw_rank*100)/(df.groupby_size + 1)).apply(round_correct, args = (1,))


def process_raw_data_for_equal_percentile_conversion(df,
                                                     suffix,
                                                     ranking_method = 'average',
                                                     ascending_method = True,
                                                     pct_method = True):
    """df should be one column with just all the scores to be matched"""
    raw_data_col_name = 'raw_data_' + suffix
    df.columns = [raw_data_col_name]
    # Percentile rank data
    df['per_rank'] = df[raw_data_col_name].rank(method = ranking_method, 
                                                ascending = ascending_method, 
                                                pct = pct_method)
    df = df.sort_values('per_rank').drop_duplicates()
    return df, raw_data_col_name


def get_closest(per_rank_1, data_df_2, raw_data_col_name_2):
    return data_df_2.iloc[(data_df_2.per_rank - per_rank_1).abs().argsort()[:1]][raw_data_col_name_2].values[0]


def create_equal_percentile_conversion_chart(data_df_1, suffix_1, 
                                             data_df_2, suffix_2):
    ranked_df_1, raw_data_col_name_1 = process_raw_data_for_equal_percentile_conversion(data_df_1, suffix_1)
    ranked_df_2, raw_data_col_name_2 = process_raw_data_for_equal_percentile_conversion(data_df_2, suffix_2)
    ranked_df_1[raw_data_col_name_2] = ranked_df_1.per_rank.apply(get_closest, args = (ranked_df_2, raw_data_col_name_2))
    return ranked_df_1.sort_values('per_rank')


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


########################## 
### Data visualization ###
##########################

def autolabel(ax, 
              decimal = 1,
              percentage = 1,
              additional_character = "",
              exclude_zero_vals = False, 
              fontsize = 10):
    """Use to label bar plots"""
    rects = ax.patches
    if percentage == 100:
        decimal_format = decimal - 2
    else:
        decimal_format = decimal
    for rect in rects:
        height = rect.get_height()
        height_val = round_correct((round_correct(rect.get_height(),decimal) * percentage), decimal)
        height_val = "{:.{decimal_format}f}".format(height_val, decimal_format = decimal_format)
        s = str(height_val)
        if exclude_zero_vals:
            if height != 0:
                ax.text(rect.get_x() + (rect.get_width()/2), height,
                        s = s + additional_character,
                        ha='center', va='bottom', fontsize = fontsize)
        else:
            ax.text(rect.get_x() + (rect.get_width()/2), height,
                        s = s + additional_character,
                        ha='center', va='bottom', fontsize = fontsize)


def to_percent(y, position):
    """
    Usage:
    formatter = plt.FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    """
    s = str(100 * y)
    s = '%1.0f' % float(s)
    return str(s) + '%'


def change_axis_to_percent(ax, axis = 'y'):
    formatter = plt.FuncFormatter(to_percent)
    if axis == 'x':
        ax.xaxis.set_major_formatter(formatter)
    if axis == 'y':
        ax.yaxis.set_major_formatter(formatter)


###################################
### Working with public reports ###
###################################

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


# TO DO: Add 2017 public workbooks


###########################
### Predictive modeling ###
###########################



#############
### Other ###
#############

def write_to_excel_template(worksheet, data, cell_range=None, named_range=None):

    """
    Updates an excel worksheet with the given data using openpyxl
    :param worksheet: an excel worksheet
    :param data: data used to update the worksheet cell range (list, tuple, np.ndarray, pd.Dataframe)
    :param cell_range: a string representing the cell range, e.g. 'AB12:XX23'
    :param named_range: a string representing an excel named range
    """
    def clean_data(data):
        if not isinstance(data, (list, tuple, np.ndarray, pd.DataFrame)):
            raise TypeError('Invalid data, data should be an array type iterable.')
 
        if not len(data):
            raise ValueError('You need to provide data to update the cells')
 
        if isinstance(data, pd.DataFrame):
            data = data.values
 
        elif isinstance(data, (list, tuple)):
            data = np.array(data)
 
        return np.hstack(data)
 
    def clean_cells(worksheet, cell_range, named_range):
        # check that we can access a cell range
        if not any((cell_range, named_range) or all((cell_range, named_range))):
            raise ValueError('`cell_range` or `named_range` should be provided.')
 
        # get the cell range
        if cell_range:
            try:                 
                cells = np.hstack(worksheet[cell_range])
            except ( AttributeError):
                raise ValueError('The cell range provided is invalid, cell range must be in the form XX--[:YY--]')
 
        else:
            try:
                cells = worksheet.get_named_range(named_range)
            except (NamedRangeException, TypeError):
                raise ValueError('The current worksheet {} does not contain any named range {}.'.format(
                     worksheet.title,
                     named_range))
         # checking that we have cells to update, and data
        if not len(cells):
            raise ValueError('You need to provide cells to update.')
        return cells
    
    cells = clean_cells(worksheet, cell_range, named_range)
    data = clean_data(data)
    # check that the data has the same dimension as cells
    if len(cells) != data.size:
        raise ValueError('Cells({}) should have the same dimension as the data({}).'.format(len(cells), data.size))

    for i, cell in enumerate(cells):
        if data[i] == 'nan':
            cell.value = u""
        else:
            cell.value = data[i]    




