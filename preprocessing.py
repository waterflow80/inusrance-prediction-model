import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import RobustScaler

import warnings
warnings.filterwarnings('ignore')


# load data and drop id
df_insurance = pd.read_csv("./test_Insurance.csv")
df_insurance.drop(axis=1, labels=['Customer Id'], inplace=True)

# nan values function
def handle_nan_values(df:pd.DataFrame, column:str, strategy:str):
  mf_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
  garden_arr = mf_imputer.fit_transform(df_insurance.loc[:, [column]])
  df_insurance[column] = pd.DataFrame(data=garden_arr, columns=[column])

# normalization funtion
def normalize_with_robust_scaler(df:pd.DataFrame, column:str):
  standard_sclaer = RobustScaler()
  bd_norm_arr = standard_sclaer.fit_transform(df.loc[:, [column]])
  df[column] = bd_norm_arr

# handle outliers function
def handle_outliers_with_box_plot(df:pd.DataFrame, column:str):
  Q1, Q3 = np.percentile(df[column], [25, 75])
  IQR = Q3 - Q1
  upper_limit = Q3+1.5*IQR
  lower_limit = Q1-1.5*IQR
  df[column] = np.where(df[column] > upper_limit, upper_limit,
                                                np.where(df[column] < lower_limit, lower_limit, df[column]))
  
# remove column function
def remove_column(df:pd.DataFrame, column:str):
  df.drop(axis=1, labels=[column], inplace=True)

# transform the num_windows values from 'without' --> 0, '>=10' --> 10 
# and cast the values to int
def transform_num_windows(df:pd.DataFrame)->None:
  df['NumberOfWindows'] = np.where(df['NumberOfWindows'] == 'without', '0', np.where(df['NumberOfWindows'] == '>=10', '10', df['NumberOfWindows']))
  ## casting to int
  df['NumberOfWindows'] = pd.to_numeric(df['NumberOfWindows'])
  
def encode_true_false(df:pd.DataFrame, feature:str,categories:list, result_cols:list):
  # this encoder will transform N to 0 and V to 1 in the given feature
  # save the result into new columns of which the names are given in result_cols
  enc = OneHotEncoder(sparse_output=False, categories=[categories])
  df[result_cols] = enc.fit_transform(df.loc[:, [feature]]) # will be added at the end
  df.drop(axis=1, labels=[feature], inplace=True)

# sepearate data
def seperate_data(df:pd.DataFrame, class_name:str):
  df_cp = df.copy(deep=True)
  Y_train = df_cp.loc[:, [class_name]].values
  df_cp.drop(axis=1, inplace=True, labels=[class_name])
  X_train = df_cp.values
  return X_train, Y_train

def transform_insured_period(df:pd.DataFrame):
  df['Insured_Period'] = df['Insured_Period'].apply(lambda x: 0 if x == 0.5 else x)
  
def encode_Building_type(df):
  enc = OrdinalEncoder(categories=[['Wood-framed', 'Ordinary', 'Non-combustible', 'Fire-resistive']])
  df['Building_Type'] = enc.fit_transform(df.loc[:, ['Building_Type']])


# nan values
handle_nan_values(df_insurance, 'Geo_Code', 'most_frequent')
handle_nan_values(df_insurance, 'Building Dimension', 'median')
handle_nan_values(df_insurance, 'Garden', 'most_frequent')

# Normalize 'Building Dimension'
normalize_with_robust_scaler(df_insurance, 'Building Dimension')
'Building Dimension'

# Handling outliers
handle_outliers_with_box_plot(df_insurance, 'Building Dimension')

# Remove unmeaningful columns
remove_column(df_insurance, 'Geo_Code')
remove_column(df_insurance, 'YearOfObservation')

# data transformation
transform_num_windows(df_insurance)
transform_insured_period(df_insurance)

# Encoding data
encode_true_false(df_insurance, 'Building_Fenced',['V', 'N'], ['Fenced=T', 'Fenced=F'])
encode_true_false(df_insurance, 'Building_Painted',['V', 'N'], ['Painted=T', 'Painted=F'])
encode_Building_type(df_insurance)
encode_true_false(df_insurance, 'Garden', ['V','O'], ['Garden=T', 'Garden=F'])
encode_true_false(df_insurance, 'Settlement', ['R', 'U'], ['Settle=R', 'Settle=U'])

# seperate data
X_test, Y_test = seperate_data(df_insurance, 'Claim')
print(type(X_test), X_test.shape)
print(type(Y_test), Y_test.shape)

def get_test_dataframe()->pd.DataFrame:
  return df_insurance