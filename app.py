import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Market Value Analysis", layout="wide")

# Smaller chart size for dashboard
FIGSIZE = (4.5, 3.2)

# ---------- TITLE ----------
st.markdown(
    "<h1 style='text-align:center;'>📊 Market Value Analysis Dashboard</h1>",
    unsafe_allow_html=True
)
st.write("")

# ---------- LOAD DATA ----------
df = pd.read_csv("companies.csv")

# ---------- FEATURE ENGINEERING ----------
df["ProfitMargin"] = (df["Profit"] / df["Revenue"]) * 100
df["RevenueToMarketValue"] = df["Revenue"] / df["MarketValue"]
df["EmployeesPerBillion"] = df["Employees"] / df["MarketValue"]

# ===================== MACHINE LEARNING MODEL =====================
feature_cols = ["Employees", "Revenue", "Profit"]
X = df[feature_cols]
y = df["MarketValue"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(n_estimators=250, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

r2 = rf.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
residuals = y_test - y_pred

# Sorted Real vs Predicted
y_test_array = y_test.values
order = np.argsort(y_test_array)
y_test_sorted = y_test_array[order]
y_pred_sorted = y_pred[order]

# ===================== TABS =====================
tab_data, tab_eda, tab_ml, tab_predict = st.tabs(
    ["📁 Dataset", "📊 EDA", "🤖 ML Model", "🔮 Predict"]
)

# ===================== DATA TAB =====================
with tab_data:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Unique Companies", df["Company"].nunique())
    st.dataframe(df.head(50), use_container_width=True)

# ===================== EDA TAB =====================
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    # -------- Row 1 --------
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax, cbar=False)
        st.pyplot(fig)

    with c2:
        st.markdown("**Company Frequency**")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.countplot(x=df["Company"], order=df["Company"].value_counts().index, ax=ax)
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    # -------- Row 2 --------
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Revenue Distribution**")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.histplot(df["Revenue"], bins=25, kde=True, ax=ax)
        st.pyplot(fig)

    with c4:
        st.markdown("**Profit Distribution**")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.histplot(df["Profit"], bins=25, kde=True, ax=ax)
        st.pyplot(fig)

    # -------- Row 3 --------
    c5, c6 = st.columns(2)

    with c5:
        st.markdown("**Market Value Distribution**")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.histplot(df["MarketValue"], bins=25, kde=True, ax=ax)
        st.pyplot(fig)

    with c6:
        st.markdown("**Revenue vs Market Value**")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.scatterplot(x="Revenue", y="MarketValue", data=df, s=18, ax=ax)
        st.pyplot(fig)

    # -------- Row 4 --------
    c7, c8 = st.columns(2)

    with c7:
        st.markdown("**Employees vs Market Value**")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.scatterplot(x="Employees", y="MarketValue", data=df, s=18, ax=ax)
        st.pyplot(fig)

    with c8:
        st.markdown("**Boxplots**")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.boxplot(data=df[["Revenue","Profit","MarketValue"]], ax=ax)
        st.pyplot(fig)

# ===================== ML MODEL TAB =====================
with tab_ml:
    st.markdown("<h3>Random Forest Regression</h3>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{r2:.4f}")
    m2.metric("MAE", f"{mae:.4f}")
    m3.metric("RMSE", f"{rmse:.4f}")

    st.write("")

    # -------- Row 1 --------
    ml1, ml2 = st.columns(2)

    with ml1:
        st.markdown("**Feature Importance**")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.barplot(x=rf.feature_importances_, y=feature_cols, ax=ax)
        st.pyplot(fig)

    with ml2:
        st.markdown("**Predicted vs Actual**")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.scatter(y_test, y_pred, s=18)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
                color='red', linestyle='--')
        st.pyplot(fig)

    # -------- Row 2 --------
    ml3, ml4 = st.columns(2)

    with ml3:
        st.markdown("**Residual Distribution**")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.histplot(residuals, bins=25, kde=True, ax=ax)
        st.pyplot(fig)

    with ml4:
        st.markdown("**Actual vs Predicted (Sorted)**")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(y_test_sorted, label="Actual", linewidth=1)
        ax.plot(y_pred_sorted, label="Predicted", linewidth=1)
        ax.legend()
        st.pyplot(fig)

# ===================== PREDICT TAB =====================
with tab_predict:
    st.subheader("Predict Market Value")

    col1, col2, col3 = st.columns(3)
    emp = col1.number_input("Employees", 1000, 300000, 50000)
    rev = col2.number_input("Revenue (Billion USD)", 1.0, 500.0, 50.0)
    prof = col3.number_input("Profit (Billion USD)", 0.1, 100.0, 10.0)

    if st.button("Predict"):
        x_in = scaler.transform([[emp, rev, prof]])
        pred = rf.predict(x_in)[0]
        st.success(f"Estimated Market Value: **${pred:.2f} Billion**")

# ---------- FOOTER ----------
st.markdown(
    "<hr><p style='text-align:center;color:grey;'>Arun Kumar C</p>",
    unsafe_allow_html=True
)
