"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""
from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """
    Generates a scatter plot of MonthlyCharges vs. TotalCharges,
    colored by customer churn status.
    """
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x=df["MonthlyCharges"],
        y=df["TotalCharges"],
        hue=df["Churn"],
        alpha=0.6
    )
    plt.title("Monthly Charges vs. Total Charges (Churned vs. Retained)")
    plt.xlabel("Monthly Charges ($)")
    plt.ylabel("Total Charges ($)")
    plt.legend(title="Churn")
    plt.savefig("relational_plot.png")
    plt.show()
    plt.close()


def plot_categorical_plot(df):
    """
    Generates a bar chart showing the count of churned and retained
    customers by contract type.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Contract", hue="Churn", data=df)
    plt.title("Churn Count by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Customer Count")
    plt.legend(title="Churn")
    plt.savefig("categorical_plot.png")
    plt.show()
    plt.close()


def plot_statistical_plot(df):
    """
    Generates a boxplot for MonthlyCharges, grouped by Churn status.
    This helps visualize the distribution of charges for churned vs.
    retained customers.
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
    plt.title("Monthly Charges Distribution by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges ($)")
    plt.savefig("statistical_plot.png")
    plt.show()
    plt.close()


def statistical_analysis(df, col: str):
    """
    Computes statistical moments: mean, standard deviation, skewness, 
    and excess kurtosis for a given column.
    
    Parameters:
        df (DataFrame): The dataset.
        col (str): The column for analysis.
    
    Returns:
        tuple: (mean, standard deviation, skewness, excess kurtosis)
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])

    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Preprocesses the dataset by converting appropriate columns to numeric,
    handling missing values, and providing basic data insights.

    Parameters:
        df (DataFrame): The dataset.
    
    Returns:
        DataFrame: Cleaned dataset.
    """
    # Convert TotalCharges to numeric (it might contain blank spaces)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values in TotalCharges with median
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    # Drop rows with missing values
    df.dropna(inplace=True)

    # Display dataset insights
    print(df.describe())
    print(df.head())
    

    return df


def writing(moments, col):
    """
    Prints the computed statistical moments in a human-readable format.

    Parameters:
        moments (tuple): (mean, stddev, skew, kurtosis)
        col (str): Column analyzed.
    """
    print(f"For the attribute '{col}':")
    print(f"Mean = {moments[0]:.2f}, Standard Deviation = {moments[1]:.2f}, "
          f"Skewness = {moments[2]:.2f}, and Excess Kurtosis = "
          f"{moments[3]:.2f}.")
    
    # Interpret skewness
    skewness_type = "not skewed"
    if moments[2] > 2:
        skewness_type = "right-skewed"
    elif moments[2] < -2:
        skewness_type = "left-skewed"

    # Interpret kurtosis
    kurtosis_type = "mesokurtic"
    if moments[3] > 2:
        kurtosis_type = "leptokurtic"
    elif moments[3] < -2:
        kurtosis_type = "platykurtic"

    print(f"The data is {skewness_type} and {kurtosis_type}.")


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'MonthlyCharges'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()
