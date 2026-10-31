import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Prediction System")
st.write("Machine Learning Based Car Price Prediction")


# =========================================================
# LOAD DATASET
# =========================================================

DATA_FILE = "car_price_2025_2026_final.csv"

try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"Dataset load korte problem hoyeche: {e}")
    st.stop()


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

df = df.dropna(axis=1, how="all")


st.success(
    f"Dataset loaded successfully! {len(df)} records found."
)


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "Brand",
    "Model",
    "Year",
    "Present_Price_Lakh",
    "Kms_Driven",
    "Fuel_Type",
    "Transmission",
    "Owner",
    "Mileage_kmpl",
    "Engine_cc"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"CSV-te ei column gulo pawa jayni: {missing_columns}"
    )

    st.write("CSV columns:")
    st.write(df.columns.tolist())

    st.stop()


# =========================================================
# CLEAN TEXT COLUMNS
# =========================================================

text_columns = [
    "Brand",
    "Model",
    "Fuel_Type",
    "Transmission"
]

for col in text_columns:

    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
    )


# =========================================================
# TARGET
# =========================================================

target = "Present_Price_Lakh"

df[target] = pd.to_numeric(
    df[target],
    errors="coerce"
)

df = df.dropna(
    subset=[target]
)


# =========================================================
# FEATURES
# =========================================================

feature_columns = [
    "Brand",
    "Model",
    "Year",
    "Kms_Driven",
    "Fuel_Type",
    "Transmission",
    "Owner",
    "Mileage_kmpl",
    "Engine_cc"
]

X = df[feature_columns].copy()

y = df[target].copy()


# =========================================================
# NUMERIC COLUMNS
# =========================================================

numeric_columns = [
    "Year",
    "Kms_Driven",
    "Owner",
    "Mileage_kmpl",
    "Engine_cc"
]

for col in numeric_columns:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

    X[col] = X[col].fillna(
        X[col].median()
    )


# =========================================================
# CATEGORICAL COLUMNS
# =========================================================

categorical_columns = [
    "Brand",
    "Model",
    "Fuel_Type",
    "Transmission"
]

for col in categorical_columns:

    X[col] = (
        X[col]
        .astype(str)
        .str.strip()
        .replace("nan", "Unknown")
    )


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================================================
# PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),

            categorical_columns
        ),

        (
            "numeric",

            "passthrough",

            numeric_columns
        )
    ]
)


# =========================================================
# RANDOM FOREST MODEL
# =========================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# =========================================================
# PIPELINE
# =========================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ]
)


# =========================================================
# TRAIN MODEL
# =========================================================

with st.spinner(
    "Training Machine Learning model..."
):

    pipeline.fit(
        X_train,
        y_train
    )


# =========================================================
# MODEL EVALUATION
# =========================================================

predictions = pipeline.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "📊 Model Information"
)

st.sidebar.write(
    f"Dataset Size: {len(df)}"
)

st.sidebar.write(
    f"Features Used: {len(feature_columns)}"
)

st.sidebar.write(
    f"Training Data: {len(X_train)}"
)

st.sidebar.write(
    f"Testing Data: {len(X_test)}"
)

st.sidebar.markdown("---")

st.sidebar.write("### Model")
st.sidebar.write(
    "Random Forest Regressor"
)

st.sidebar.write("### Target")
st.sidebar.write(
    "Present Price (Lakh)"
)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader(
    "📈 Model Performance"
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "R² Score",
        f"{r2:.3f}"
    )

with col2:

    st.metric(
        "Mean Absolute Error",
        f"{mae:.3f} Lakh"
    )


# =========================================================
# PREDICTION SECTION
# =========================================================

st.markdown("---")

st.header(
    "🚘 Predict Car Price"
)


# =========================================================
# BRAND AND MODEL
# =========================================================

st.subheader(
    "1️⃣ Select Car"
)

col1, col2 = st.columns(2)


# =========================================================
# BRAND
# =========================================================

with col1:

    brand_list = sorted(
        X["Brand"]
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    selected_brand = st.selectbox(
        "🏷️ Brand",
        brand_list
    )


# =========================================================
# MODEL BASED ON BRAND
# =========================================================

with col2:

    model_list = sorted(
        X.loc[
            X["Brand"]
            .astype(str)
            .str.strip()
            == selected_brand,

            "Model"
        ]
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    selected_model = st.selectbox(
        "🚗 Model",
        model_list
    )


# =========================================================
# FIND SELECTED CAR
# =========================================================

selected_car = X[
    (
        X["Brand"]
        .astype(str)
        .str.strip()
        == selected_brand
    )
    &
    (
        X["Model"]
        .astype(str)
        .str.strip()
        == selected_model
    )
]


# =========================================================
# CAR INFORMATION
# =========================================================

st.markdown("---")

st.subheader(
    "2️⃣ Car Information"
)

st.caption(
    "Model select korar por values automatically ashbe. "
    "Proyojon hole tumi values change korte parbe."
)


if len(selected_car) > 0:

    car = selected_car.iloc[0]


    # =====================================================
    # YEAR + ENGINE
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        year_value = st.number_input(
            "📅 Year",

            min_value=1990,

            max_value=2030,

            value=int(car["Year"]),

            step=1,

            key="year_input"
        )


    with col2:

        engine_value = st.number_input(
    "⚙️ Engine (cc)",
    min_value=0.0,
    max_value=10000.0,
    value=float(car["Engine_cc"]),
    step=50.0,
    key=f"engine_{selected_brand}_{selected_model}"
)
        


    # =====================================================
    # MILEAGE + KMS
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        mileage_value = st.number_input(
            "⛽ Mileage (km/l)",

            min_value=1.0,

            max_value=100.0,

            value=max(
    1.0,
    float(car["Mileage_kmpl"])
    if pd.notna(car["Mileage_kmpl"]) and car["Mileage_kmpl"] > 0
    else float(X["Mileage_kmpl"].median())
),

            step=0.1,

            key=f"mileage_{selected_brand}_{selected_model}"
        )


    with col2:

        kms_value = st.number_input(
            "🛣️ Kms Driven",

            min_value=0.0,

            max_value=1000000.0,

            value=float(car["Kms_Driven"]),

            step=100.0,

            key="kms_input"
        )


    # =====================================================
    # FUEL TYPE
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        fuel_options = sorted(
            X["Fuel_Type"]
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        default_fuel = str(
            car["Fuel_Type"]
        )

        if default_fuel in fuel_options:

            fuel_index = fuel_options.index(
                default_fuel
            )

        else:

            fuel_index = 0


        fuel_value = st.selectbox(
            "⛽ Fuel Type",

            fuel_options,

            index=fuel_index,
key=f"fuel_{selected_brand}_{selected_model}"
        )


    # =====================================================
    # TRANSMISSION
    # =====================================================

    with col2:

        transmission_options = sorted(
            X["Transmission"]
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        default_transmission = str(
            car["Transmission"]
        )

        if (
            default_transmission
            in transmission_options
        ):

            transmission_index = (
                transmission_options.index(
                    default_transmission
                )
            )

        else:

            transmission_index = 0


        transmission_value = st.selectbox(
            "⚙️ Transmission",

            transmission_options,

            index=transmission_index,

            key="transmission_input"
        )


    # =====================================================
    # OWNER
    # =====================================================

    owner_value = st.number_input(
        "👤 Owner",

        min_value=0,

        max_value=10,

        value=int(car["Owner"]),

        step=1,

        key="owner_input"
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

st.markdown("---")

if st.button(
    "🔮 Predict Car Price",
    use_container_width=True
):

    try:

        # -------------------------------------------------
        # CREATE USER INPUT
        # -------------------------------------------------

        input_data = {

            "Brand": selected_brand,

            "Model": selected_model,

            "Year": year_value,

            "Kms_Driven": kms_value,

            "Fuel_Type": fuel_value,

            "Transmission": transmission_value,

            "Owner": owner_value,

            "Mileage_kmpl": mileage_value,

            "Engine_cc": engine_value
        }


        # -------------------------------------------------
        # DATAFRAME
        # -------------------------------------------------

        input_df = pd.DataFrame(
            [input_data]
        )


        # -------------------------------------------------
        # SAME COLUMN ORDER AS TRAINING
        # -------------------------------------------------

        input_df = input_df[
            feature_columns
        ]


        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        predicted_price = pipeline.predict(
            input_df
        )[0]


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.success(
            f"### 💰 Estimated Car Price: ₹ {predicted_price:.2f} Lakh"
        )

        st.info(
            "Price predicted using Random Forest Machine Learning model."
        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )



# =================================================
# DATASET INFORMATION
# =================================================

st.markdown("---")
st.subheader("📊 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Cars",
        len(df)
    )

with col2:
    st.metric(
        "Total Features",
        len(X.columns)
    )

with col3:
    st.metric(
        "Training Data",
        len(X_train)
    )



# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

st.markdown("---")

st.subheader(
    "📊 Actual vs Predicted Price"
)

comparison = pd.DataFrame(
    {
        "Actual Price": y_test.values,

        "Predicted Price": predictions
    }
)

st.line_chart(
    comparison
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Car Price Prediction | Machine Learning Project | Random Forest Regression"
)