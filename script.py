import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def plot_boxplot(df, output_path):
    df_boxplot = df.drop(["p_id", "diabetes"], axis=1)

    plt.figure(figsize=(12, 10))
    df_boxplot.boxplot(rot=30, fontsize=10)
    plt.title("Boxplot of all features", fontsize=16)

    # plt.show()
    plt.savefig(output_path)


def plot_correlation(df, output_path):
    correlation_matrix = df.drop(["p_id"], axis=1).corr()
    correlation_with_diabetes = correlation_matrix["diabetes"]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        correlation_with_diabetes.index,
        correlation_with_diabetes.values,
        color="skyblue",
        width=0.5,
    )

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            round(yval, 2),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xlabel("Attributes")
    plt.ylabel("Correlation with Diabetes")
    plt.title("Correlation of Features with Diabetes")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(output_path)


def plot_coefficients(df, output_path):
    # Determine the color based on the sign of coefficients
    colors = np.where(df["Coefficient"] > 0, "skyblue", "lightcoral")

    # Plot the bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.barh(df["Feature"], df["Coefficient"], color=colors)
    plt.xlabel("Coefficient Value")
    plt.title("Logistic Regression Coefficients")

    # Add values on the bars
    for bar, coef_val in zip(bars, df["Coefficient"]):
        xval = bar.get_width() if coef_val > 0 else 0

        plt.text(
            xval + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{round(coef_val, 4)}",
            ha="center",
            va="center",
            color="black",
        )

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()


def statistics(df, output_path):
    df = df.drop(["p_id", "diabetes"], axis=1)
    statistics = df.describe(percentiles=[0.25, 0.50, 0.75])

    statistics.to_csv(output_path)


def fill_outliers(df_train, df_test):
    columns_to_process = [
        "glucose_concentration",
        "blood_pressure",
        "skin_fold_thickness",
        "serum_insulin",
        "bmi",
    ]

    # Step 1: Replace 0 values in the specified columns with NaN
    df_train[columns_to_process] = df_train[columns_to_process].replace(0, np.nan)
    df_test[columns_to_process] = df_test[columns_to_process].replace(0, np.nan)

    # Step 2: Group by 'diabetes' and calculate the mean values for each group, and also calculate the mean values for the whole dataset
    means = df_train[columns_to_process].mean()
    grouped = df_train.groupby("diabetes")
    group_means = grouped[columns_to_process].mean()

    # Step 3: Fill missing values with the mean values for each corresponding group
    # For test data, fill  with the mean values for the whole dataset
    for col in columns_to_process:
        df_train[col].fillna(df_train["diabetes"].map(group_means[col]), inplace=True)
        df_test[col].fillna(means[col], inplace=True)

    return df_train, df_test


if __name__ == "__main__":
    df_train = pd.read_csv("dataset/train.csv")
    df_test = pd.read_csv("dataset/test.csv")

    # -- Check null value -- #
    # print("Null values:")
    # print(df_train.isnull().sum())

    # -- Outlier analysis -- #
    plot_boxplot(df_train, "img/boxplot_ori.png")
    statistics(df_train, "output/stat_ori.csv")

    # -- Fill outliers -- #
    df_train, df_test = fill_outliers(df_train, df_test)

    plot_boxplot(df_train, "img/boxplot_filled.png")
    statistics(df_train, "output/stat_filled.csv")

    # -- Correlation analysis -- #
    plot_correlation(df_train, "img/correlation.png")

    # -- Standardization -- #
    scaler = StandardScaler()
    features_to_normalize = [
        "no_times_pregnant",
        "glucose_concentration",
        "blood_pressure",
        "skin_fold_thickness",
        "serum_insulin",
        "bmi",
        "diabetes pedigree",
        "age",
    ]

    df_train[features_to_normalize] = scaler.fit_transform(
        df_train[features_to_normalize]
    )
    df_test[features_to_normalize] = scaler.transform(df_test[features_to_normalize])

    df_train.to_csv("output/normalized_data.csv", index=False)

    # -- Build the training model -- #
    # Spliting data into training and testing sets (8:2)
    random_seed = 24
    X = df_train.drop(["p_id", "diabetes"], axis=1)
    y = df_train["diabetes"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )

    # Training the model
    logreg_model = LogisticRegression(random_state=random_seed)
    logreg_model.fit(X_train, y_train)

    # -- Evaluating the model -- #
    y_pred = logreg_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # -- Print the evaluation results -- #
    print(f"Accuracy: {accuracy:.2f}\n")
    print("Confusion Matrix:")
    print(f"{conf_matrix}\n")
    print("Classification Report:")
    print(classification_rep)

    # -- Coefficients analysis -- #
    coefficients = logreg_model.coef_[0]

    coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": coefficients})
    # coef_df = coef_df.sort_values(by="Coefficient")

    plot_coefficients(coef_df, "img/coefficients.png")

    # -- Predict the test data --#
    test_pred = logreg_model.predict(df_test.drop(["p_id"], axis=1))

    output_df = pd.DataFrame({"p_id": df_test["p_id"], "diabetes": test_pred})
    output_df.to_csv("output/test_predict.csv", index=False)
