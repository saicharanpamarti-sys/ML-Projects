import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

file_path=r"C:\Users\saich\Downloads\ML\PythonProject12\placement_predict_50k Dataset (3) 1(in).csv"
df=pd.read_csv(file_path)

# numericColumns=["CGPA", "SoftSkillsRating","AttendancePercent","AptitudeTestScore"]
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["PlacementStatus"])
# print(train_df.shape)
# print(test_df.shape)
# print("\nTraining Data before scaling:")
# print(train_df[numericColumns].head())
# print("\nTesting Data before scaling:")
# print(test_df[numericColumns].head())

scaler=MinMaxScaler()

# train_df[numericColumns]=scaler.fit_transform(train_df[numericColumns])
#
# test_df[numericColumns]=scaler.transform(test_df[numericColumns]) #in testing we use only transform instead of fit_trasnfom as in fit_tarsnform we calculate the min and max values of the entire data and this is also used in test data
#
# print("\nTraining Data after scaling:")
# print(train_df[numericColumns].head())
#
# print("\nTesting Data after scaling:")
# print(test_df[numericColumns].head())

feature=["CodingTestScore"]
print("\nTraining Data before scaling")
print(train_df[feature].head())
print("\nTesting Data before scaling")
print(test_df[feature].head())

train_df[feature]=scaler.fit_transform(train_df[feature])
test_df[feature]=scaler.transform(test_df[feature])

print("\nTraining Data after scaling:")
print(train_df[feature].head())
print("\nTesting Data after scaling:")
print(test_df[feature].head())