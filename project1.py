from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import model_selection
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def about_model():
    st.title("🔧 Predictive Maintenance Classifier")
    st.subheader("🔧 FleetGuard AI — Your Equipment’s Early-Warning System")
    st.markdown("""Stay ahead of breakdowns, not behind them.In heavy‑duty industrial environments, unplanned downtime impacts both productivity and profitability.
    FleetGuard AI continuously monitors vital parameters—temperature, rotational speed, torque—to detect abnormalities before they become failures.
    Powered by an ensemble of Random Forest classifiers, the system captures subtle deviations across sensor streams and votes on risk levels,
    offering a reliable, real‑time prediction of equipment health""")
    st.divider()
    st.subheader("🎯 Proactive, Data-Driven Maintenance")
    st.markdown("By relying on condition‑based insights—rather than purely scheduled servicing—your maintenance becomes smarter." \
    "You’ll save costs, lengthen machinery lifespan, and boost safety." \
    "Studies show this approach enhances:")
    st.markdown("""| Outcome              | Benefit                                           |
    | -------------------- | ------------------------------------------------- |
    |  |  ([Reddit][1]) |
    | Cost-efficiency      |             |
    | Safety & reliability | Minimizes accidents and improves quality          |""")
    st.subheader("Outcome:")
    st.markdown("1.Operational lifespan")
    st.markdown("2.Cost Efficiency")
    st.markdown("3.Safety & reliability")
    st.subheader("Benefits:")
    st.markdown("1.Machines run longer without failure")
    st.markdown("2.Reduces labor, parts, and energy costs")
    st.markdown("3.Minimizes accidents and improves quality")
    st.divider()
    st.title("🛠️How does it works??")
    st.subheader("By analyzing sensor readings such as:" )
    st.markdown("🌡️🔥Temperature")
    st.markdown("⚙️🔃Rotational Speed")
    st.markdown("🔧⚙️Torque")
    st.markdown("Machine Learning models can detect subtle patterns that signal impending faults.")

    st.divider()
    st.subheader("Checkout the sidebar for more➡️")

df = pd.read_csv("predictive_maintenance.csv")
feature_cols = [c for c in df.columns if c not in [ "Failure Type",'Product ID','Type','Target']]
X = df[feature_cols]
y = df["Target"]

def dataset_visualisation():
    st.header("⚙️📈Machine maintenance dataset visualizations")
    st.code(X.head())
    st.code((X.shape, y.shape))

    st.subheader("🔍📅Dataset NULL counts and dtypes")
    null_col, dtype_col = st.columns(2)
    with null_col:
        st.code(X.isna().sum())
    with dtype_col:
        st.code(X.dtypes)
        
    st.divider()
    st.title("Visualisation")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, y='UDI', x='Failure Type', ax=ax)
    ax.set_title("Count of Failure Types vs UDI")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

def prediction_model():
    st.header("🔍Try Prediction Model")
    st.subheader("Enter details of  the Machine:")
    with st.form("input_form"):
        cols = feature_cols
        user_input = {}
        for col in cols:
            mn, mx = float(X[col].min()), float(X[col].max())
            default = float(X[col].median())
            # allow text input, with fallback to slider
            val = st.text_input(f"{col}", value=str(default))
            try:
                num = float(val)
            except ValueError:
                st.warning(f"Invalid value for {col}. Using median {default}.")
                num = default
            user_input[col] = num
        submit = st.form_submit_button("🔍 Predict Failure")

    input_df = pd.DataFrame([user_input])

    st.subheader("User Input Features")
    st.dataframe(input_df)

    import numpy as np

    clf_temp = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    clf_temp.fit(X_train, y_train)
    imp = pd.Series(clf_temp.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    top = imp.head(10).index.tolist()
    X_train_sel, X_test_sel = X_train[top], X_test[top]

    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    pipeline.fit(X_train, y_train)

    if submit:
            pred = pipeline.predict(input_df)[0]
            proba = pipeline.predict_proba(input_df)[0][1]
            st.write(f"**Prediction:** {'🛑 Failure' if pred else '✅ No Failure'}")
            st.write(f"**Failure Probability:** {proba*100:.2f}%")

            st.subheader("📊 Model Evaluation")
            y_pred = pipeline.predict(X_test)
            st.text(classification_report(y_test, y_pred))
            RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
            ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)

pg = st.navigation([
  st.Page(about_model, title="About The Model"),
  st.Page(dataset_visualisation, title="Dataset Visualization"),
  st.Page(prediction_model, title="Let's Predict Machine Failure"),
])
pg.run()
