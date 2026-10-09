
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("Unemployment in India.csv")

# 2. Clean column names and text values
df.columns = df.columns.str.strip()

df["Region"] = df["Region"].astype(str).str.strip()
df["Area"] = df["Area"].astype(str).str.strip()

rate_col = "Estimated Unemployment Rate (%)"

# 3. Convert columns to correct data types
df["Date"] = pd.to_datetime(
    df["Date"].astype(str).str.strip(),
    dayfirst=True,
    errors="coerce"
)

df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")

df["Estimated Employed"] = pd.to_numeric(
    df["Estimated Employed"], errors="coerce"
)

df["Estimated Labour Participation Rate (%)"] = pd.to_numeric(
    df["Estimated Labour Participation Rate (%)"], errors="coerce"
)

# 4. Remove missing values and duplicate rows
df = df.dropna(subset=["Date", rate_col])
df = df.drop_duplicates()

# 5. Display dataset information
print("First five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df[rate_col].describe())

# 6. Sort records by date
df = df.sort_values("Date")

# 7. Calculate monthly average unemployment rate
monthly = df.groupby(
    df["Date"].dt.to_period("M")
)[rate_col].mean()

monthly.index = monthly.index.to_timestamp()

# 8. Plot unemployment trend
plt.figure(figsize=(12, 5))
plt.plot(monthly.index, monthly.values, marker="o")
plt.title("Unemployment Rate in India Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Compare pre-COVID and COVID periods
pre_covid = df[
    (df["Date"] >= "2019-01-01") &
    (df["Date"] < "2020-03-01")
]

covid_period = df[
    (df["Date"] >= "2020-03-01") &
    (df["Date"] < "2022-01-01")
]

print("\nPre-COVID Average Unemployment Rate:")
print(round(pre_covid[rate_col].mean(), 2), "%")

print("\nCOVID-Period Average Unemployment Rate:")
print(round(covid_period[rate_col].mean(), 2), "%")

# 10. Find the month with the highest average rate
if not monthly.empty:
    peak_date = monthly.idxmax()
    peak_rate = monthly.max()

    print("\nHighest Monthly Average Unemployment:")
    print("Month:", peak_date.strftime("%B %Y"))
    print("Rate:", round(peak_rate, 2), "%")

# 11. Analyze calendar-month patterns
df["Month"] = df["Date"].dt.month

monthly_seasonal = df.groupby("Month")[rate_col].mean()

month_names = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

plt.figure(figsize=(10, 5))
sns.barplot(
    x=monthly_seasonal.index,
    y=monthly_seasonal.values
)
plt.title("Average Unemployment by Calendar Month")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(range(12), month_names)
plt.tight_layout()
plt.show()

# 12. Compare pre-COVID and COVID averages
comparison = pd.DataFrame({
    "Period": ["Pre-COVID", "COVID"],
    "Average Rate": [
        pre_covid[rate_col].mean(),
        covid_period[rate_col].mean()
    ]
}).dropna()

plt.figure(figsize=(7, 5))
sns.barplot(
    data=comparison,
    x="Period",
    y="Average Rate"
)
plt.title("Pre-COVID vs COVID Unemployment Rate")
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# 13. Compare rural and urban unemployment
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="Area",
    y=rate_col
)
plt.title("Average Unemployment: Rural vs Urban")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# 14. Find the regions with the highest average unemployment
region_rates = df.groupby("Region")[rate_col].mean()
region_rates = region_rates.sort_values(ascending=False)

print("\nTop 10 Regions by Average Unemployment Rate:")
print(region_rates.head(10).round(2))
