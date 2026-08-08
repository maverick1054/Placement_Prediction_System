
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split




df = pd.read_excel("data/placement_predict_50k_Dataset.xlsx")
#
# print(df.head())
# print(df.tail())
# print(df.shape)
# print(df.info)
# print(df.describe())
# print(df.isnull().sum())
# miss = df.isnull().sum()
# print(miss>0)
# print(df.columns)

# plt.figure(figsize = (6,4))
# plt.hist(df["CGPA"], bins=10, edgecolor="black")
# plt.title("Histogram of CGPA")
# plt.xlabel("CGPA")
# plt.ylabel("Frequency")
# plt.show()

#
# plt.figure(figsize = (6,4))
# plt.hist(df["Salary Package"], bins=10, edgecolor="black")
# plt.title("Histogram of salary distribution")
# plt.xlabel("Salary")
# plt.ylabel("Frequency")
# plt.show()


# plt.figure(figsize=(8,8))
# sns.boxplot(x="PlacementStatus", y="CGPA", data=df)
# plt.title("CGPA vs Placement Status")
# plt.xlabel("Placement Status")
# plt.ylabel("CGPA")
# plt.show()
#
# plt.figure(figsize=(8,8))
# sns.countplot(x="Gender", y ="PlacementStatus", data=df)
# plt.title("Gender vs Placement Status")
#
# plt.show()


df = pd.read_csv("dataset/placement_predict_50k_Dataset.xlsx")
num_cols = ["CGPA","AttendancePercent","AptitudeTestScore","CodingTestScore","Internships"]
train_df, test_df = train_test_split(df,test_size=0.2,random_state=42, stratify=df["PlacementStatus"])
scaler = MinMaxScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])
print(test_df["CGPA"])

