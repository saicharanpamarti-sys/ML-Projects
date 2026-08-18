import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


df=pd.read_csv(r"C:\Users\saich\Downloads\ML\PythonProject12\placement_predict_50k Dataset (3) 1(in).csv")
num_cols = [
    "AttendancePercent",
    "CGPA",
    "CodingTestScore",
    "AptitudeTestScore",
    "MockInterviewScore"
]

train_df,test_df=train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    #stratify=df["CGPA"]optional, if we didnt include it, it will randomly divide data on every run
)

print("Train DataFrame rows:", train_df.shape)
print("Train DataFrame columns:", train_df.shape)
print("Test DataFrame rows:", test_df.shape)
print("Test DataFrame columns:", test_df.shape)


print("\nTraining Data before scaling:")
print(train_df[num_cols].head())#top 5

print("\nTesting Data before scaling:")
print(test_df[num_cols].head())

scaler = MinMaxScaler()#scalar = StandardScaler() - to print morre columns

train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

print("\n Scaling Data after scaling:")
print(train_df[num_cols].head())

print("\nTesting Data after scaling:")
print(test_df[num_cols].head())

