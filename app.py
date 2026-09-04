import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE


# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Prediction System")
st.write(
    "A Machine Learning application comparing "
    "Logistic Regression and Decision Tree algorithms."
)

st.info(
    "⚠️ This application is created for educational purposes only "
    "and should not be used as a medical diagnosis."
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("diabetes.csv")

st.sidebar.header("Dataset Information")
st.sidebar.write(f"Total Records: {df.shape[0]}")
st.sidebar.write(f"Total Features: {df.shape[1] - 1}")


# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

df_clean = df.copy()

invalid_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in invalid_columns:
    df_clean[column] = df_clean[column].replace(0, np.nan)
    df_clean[column] = df_clean[column].fillna(
        df_clean[column].median()
    )


# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------

X = df_clean.drop("Outcome", axis=1)
y = df_clean["Outcome"]


# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------

selector = SelectKBest(
    score_func=f_classif,
    k=6
)

X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X_train.columns[
    selector.get_support()
].tolist()


# --------------------------------------------------
# DATA BALANCING USING SMOTE
# --------------------------------------------------

smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_selected,
    y_train
)


# --------------------------------------------------
# SCALING FOR LOGISTIC REGRESSION
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_balanced
)

X_test_scaled = scaler.transform(
    X_test_selected
)


# --------------------------------------------------
# MODEL 1 - LOGISTIC REGRESSION
# --------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train_balanced
)

logistic_predictions = logistic_model.predict(
    X_test_scaled
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)


# --------------------------------------------------
# MODEL 2 - DECISION TREE
# --------------------------------------------------

decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

decision_tree.fit(
    X_train_balanced,
    y_train_balanced
)

tree_predictions = decision_tree.predict(
    X_test_selected
)

tree_accuracy = accuracy_score(
    y_test,
    tree_predictions
)


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📁 Dataset",
    "🧹 Data Preparation",
    "⚖️ Data Balancing",
    "🎯 Feature Selection",
    "📊 Visualization",
    "🤖 Model Comparison",
    "🩺 Prediction"
])


# ==================================================
# TAB 1 - DATASET
# ==================================================

with tab1:

    st.header("📁 Raw Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Records",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Features",
            df.shape[1] - 1
        )

    with col3:
        st.metric(
            "Target",
            "Outcome"
        )

    st.subheader("First 10 Records")

    st.dataframe(
        df.head(10),
        width="stretch"
    )

    st.subheader("Dataset Shape")

    st.write(
        f"The dataset contains **{df.shape[0]} rows** "
        f"and **{df.shape[1]} columns**."
    )


# ==================================================
# TAB 2 - DATA PREPARATION
# ==================================================

with tab2:

    st.header("🧹 Data Preparation")

    st.write(
        "The dataset contains some zero values that are not "
        "meaningful for medical measurements."
    )

    st.subheader("Columns Treated as Missing")

    for column in invalid_columns:
        st.write(f"• {column}")

    st.subheader("Cleaning Method")

    st.write(
        "Zero values are replaced with missing values (NaN) "
        "and then replaced using the median value of each column."
    )

    st.subheader("Before Cleaning")

    st.dataframe(
        df.head(10),
        width="stretch"
    )

    st.subheader("After Cleaning")

    st.dataframe(
        df_clean.head(10),
        width="stretch"
    )


# ==================================================
# TAB 3 - DATA BALANCING
# ==================================================

with tab3:

    st.header("⚖️ Data Balancing Using SMOTE")

    before_counts = y_train.value_counts().sort_index()

    after_counts = pd.Series(
        y_train_balanced
    ).value_counts().sort_index()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Before SMOTE")

        st.write(
            f"Non-Diabetic (0): {before_counts[0]}"
        )

        st.write(
            f"Diabetic (1): {before_counts[1]}"
        )

    with col2:

        st.subheader("After SMOTE")

        st.write(
            f"Non-Diabetic (0): {after_counts[0]}"
        )

        st.write(
            f"Diabetic (1): {after_counts[1]}"
        )

    st.subheader("Class Distribution")

    comparison = pd.DataFrame({
        "Before SMOTE": before_counts,
        "After SMOTE": after_counts
    })

    st.bar_chart(comparison)

    st.success(
        "SMOTE was applied only to the training data "
        "to balance the two classes."
    )


# ==================================================
# TAB 4 - FEATURE SELECTION
# ==================================================

with tab4:

    st.header("🎯 Feature Selection")

    st.write(
        "SelectKBest with ANOVA F-score was used to select "
        "the 6 most relevant features."
    )

    st.subheader("Selected Features")

    for feature in selected_features:
        st.write(f"✅ {feature}")

    feature_scores = pd.DataFrame({
        "Feature": X_train.columns,
        "Score": selector.scores_
    })

    feature_scores = feature_scores.sort_values(
        by="Score",
        ascending=False
    )

    st.subheader("Feature Importance Scores")

    st.dataframe(
        feature_scores,
        width="stretch"
    )

    st.bar_chart(
        feature_scores.set_index("Feature")
    )


# ==================================================
# TAB 5 - VISUALIZATION
# ==================================================

with tab5:

    st.header("📊 Data Visualization")

    st.subheader("Diabetes Outcome Distribution")

    outcome_counts = df_clean["Outcome"].value_counts()

    st.bar_chart(outcome_counts)

    st.write(
        "0 = No Diabetes | 1 = Diabetes"
    )

    st.subheader("Glucose Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        df_clean["Glucose"],
        bins=20
    )

    ax.set_xlabel("Glucose")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Glucose Distribution")

    st.pyplot(fig)

    st.subheader("BMI Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        df_clean["BMI"],
        bins=20
    )

    ax.set_xlabel("BMI")
    ax.set_ylabel("Number of Patients")
    ax.set_title("BMI Distribution")

    st.pyplot(fig)

    st.subheader("Age Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        df_clean["Age"],
        bins=20
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Age Distribution")

    st.pyplot(fig)


# ==================================================
# TAB 6 - MODEL COMPARISON
# ==================================================

with tab6:

    st.header("🤖 Model Comparison")

    results = pd.DataFrame({
        "Algorithm": [
            "Logistic Regression",
            "Decision Tree"
        ],
        "Accuracy": [
            logistic_accuracy,
            tree_accuracy
        ]
    })

    results["Accuracy (%)"] = (
        results["Accuracy"] * 100
    ).round(2)

    st.subheader("Accuracy Results")

    st.dataframe(
        results,
        width="stretch"
    )

    st.subheader("Visual Comparison")

    chart_data = results.set_index(
        "Algorithm"
    )["Accuracy (%)"]

    st.bar_chart(chart_data)

    if logistic_accuracy > tree_accuracy:

        st.success(
            f"🏆 Logistic Regression performs better "
            f"with an accuracy of "
            f"{logistic_accuracy * 100:.2f}%."
        )

    elif tree_accuracy > logistic_accuracy:

        st.success(
            f"🏆 Decision Tree performs better "
            f"with an accuracy of "
            f"{tree_accuracy * 100:.2f}%."
        )

    else:

        st.info(
            "Both algorithms have the same accuracy."
        )


# ==================================================
# TAB 7 - PREDICTION
# ==================================================

with tab7:

    st.header("🩺 Diabetes Prediction")

    st.write(
        "Enter the patient's information below. "
        "The exact same input will be given to both "
        "Machine Learning algorithms."
    )

    st.divider()

    # ----------------------------------------------
    # INPUT FIELDS
    # ----------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose Level",
            min_value=0,
            max_value=300,
            value=120
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0,
            max_value=200,
            value=70
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0,
            max_value=100,
            value=20
        )

    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0,
            max_value=900,
            value=80
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    st.divider()

    predict_button = st.button(
        "🔍 Predict Diabetes",
        type="primary"
    )

    # ----------------------------------------------
    # PREDICTION
    # ----------------------------------------------

    if predict_button:

        input_data = pd.DataFrame({
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [blood_pressure],
            "SkinThickness": [skin_thickness],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [diabetes_pedigree],
            "Age": [age]
        })

        # Apply same preprocessing
        input_selected = selector.transform(
            input_data
        )

        # Logistic Regression requires scaling
        input_scaled = scaler.transform(
            input_selected
        )

        logistic_result = logistic_model.predict(
            input_scaled
        )[0]

        logistic_probability = logistic_model.predict_proba(
            input_scaled
        )[0][1]

        # Decision Tree uses selected features
        tree_result = decision_tree.predict(
            input_selected
        )[0]

        tree_probability = decision_tree.predict_proba(
            input_selected
        )[0][1]

        st.subheader("🔬 Prediction Results")

        result_col1, result_col2 = st.columns(2)

        # ------------------------------------------
        # LOGISTIC REGRESSION RESULT
        # ------------------------------------------

        with result_col1:

            st.markdown(
                "### 🟦 Logistic Regression"
            )

            if logistic_result == 1:

                st.error(
                    "Prediction: Diabetes Detected"
                )

            else:

                st.success(
                    "Prediction: No Diabetes"
                )

            st.write(
                f"Probability of Diabetes: "
                f"**{logistic_probability * 100:.2f}%**"
            )

        # ------------------------------------------
        # DECISION TREE RESULT
        # ------------------------------------------

        with result_col2:

            st.markdown(
                "### 🌳 Decision Tree"
            )

            if tree_result == 1:

                st.error(
                    "Prediction: Diabetes Detected"
                )

            else:

                st.success(
                    "Prediction: No Diabetes"
                )

            st.write(
                f"Probability of Diabetes: "
                f"**{tree_probability * 100:.2f}%**"
            )

        # ------------------------------------------
        # COMPARISON
        # ------------------------------------------

        st.divider()

        st.subheader("📊 Prediction Comparison")

        if logistic_result == tree_result:

            if logistic_result == 1:

                st.warning(
                    "⚠️ Both algorithms predict: Diabetes"
                )

            else:

                st.success(
                    "✅ Both algorithms predict: No Diabetes"
                )

        else:

            st.warning(
                "⚠️ The two algorithms produced different predictions."
            )

        comparison_df = pd.DataFrame({
            "Algorithm": [
                "Logistic Regression",
                "Decision Tree"
            ],
            "Prediction": [
                "Diabetes" if logistic_result == 1
                else "No Diabetes",
                "Diabetes" if tree_result == 1
                else "No Diabetes"
            ],
            "Diabetes Probability (%)": [
                round(logistic_probability * 100, 2),
                round(tree_probability * 100, 2)
            ]
        })

        st.dataframe(
            comparison_df,
            width="stretch"
        )

        st.info(
            "The same patient information was provided "
            "to both algorithms for comparison."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Diabetes Prediction System | "
    "Machine Learning Assignment | "
    "For Educational Purposes Only"
)