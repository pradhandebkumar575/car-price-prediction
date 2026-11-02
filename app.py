import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import sqlite3
from datetime import datetime

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
# CAR IMAGE FUNCTION
# =========================================================

def show_car_image(brand, model, year):

    st.subheader(f"🚘 {brand} {model}")

    search_query = f'"{brand} {model}"'

    try:
        response = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": f'intitle:{search_query}',
                "srnamespace": 6,
                "srlimit": 20,
                "format": "json"
            },
            headers={
                "User-Agent": "CarPricePredictionApp/1.0"
            },
            timeout=10
        )

        response.raise_for_status()

        results = response.json().get(
            "query", {}
        ).get("search", [])

        image_url = None

        for result in results:

            title = result.get("title", "").lower()

            if brand.lower() not in title or model.lower() not in title:
                continue

            image_response = requests.get(
                "https://commons.wikimedia.org/w/api.php",
                params={
                    "action": "query",
                    "titles": result["title"],
                    "prop": "imageinfo",
                    "iiprop": "url",
                    "iiurlwidth": 700,
                    "format": "json"
                },
                headers={
                    "User-Agent": "CarPricePredictionApp/1.0"
                },
                timeout=10
            )

            image_response.raise_for_status()

            pages = image_response.json().get(
                "query", {}
            ).get("pages", {})

            for page in pages.values():

                image_info = page.get("imageinfo", [])

                if image_info:
                    image_url = image_info[0].get("thumburl")
                    break

            if image_url:
                break

        if image_url:

            st.image(
                image_url,
                caption=f"{brand} {model}",
                use_container_width=True
            )

        else:
            st.info(
                f"No matching image found for {brand} {model}."
            )

    except requests.RequestException:

        st.warning(
            "Unable to load car image. Please check your internet connection."
        )
# =========================================================
# LOAD DATASET
# =========================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(APP_DIR, "car_price_dataset_1500.csv")

try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
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
    "Engine_cc",
    "City",
    "Accident_History"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"Error: The following required columns are missing from the CSV file: {missing_columns}"
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
    "Engine_cc",
    "City",
    "Accident_History"
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
    "Transmission",
    "City",
    "Accident_History"
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
    "Some fields are pre-filled based on the dataset. You can modify them if needed."
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

    # CITY
    city_options = sorted(X["City"].astype(str).str.strip().unique().tolist())
    city_default = str(car["City"]).strip()
    city_index = city_options.index(city_default) if city_default in city_options else 0
    city_value = st.selectbox(
        "📍 City",
        city_options,
        index=city_index,
        key="city_input"
    )

    # ACCIDENT HISTORY
    accident_options = sorted(
        X["Accident_History"].astype(str).str.strip().unique().tolist()
    )
    accident_default = str(car["Accident_History"]).strip()
    accident_index = (
        accident_options.index(accident_default)
        if accident_default in accident_options else 0
    )
    accident_value = st.selectbox(
        "⚠️ Accident History",
        accident_options,
        index=accident_index,
        key="accident_history_input"
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

        input_data = {
            "Brand": selected_brand,
            "Model": selected_model,
            "Year": year_value,
            "Kms_Driven": kms_value,
            "Fuel_Type": fuel_value,
            "Transmission": transmission_value,
            "Owner": owner_value,
            "Mileage_kmpl": mileage_value,
            "Engine_cc": engine_value,
            "City": city_value,
            "Accident_History": accident_value
        }

        input_df = pd.DataFrame([input_data])

        input_df = input_df[feature_columns]

        predicted_price = pipeline.predict(input_df)[0]

        st.session_state["predicted_car_price"] = float(predicted_price)

        # Save each successful prediction permanently in a local SQLite database
        history_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_history.db")
        with sqlite3.connect(history_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    predicted_at TEXT NOT NULL,
                    brand TEXT, model TEXT, year INTEGER,
                    kms_driven REAL, fuel_type TEXT, transmission TEXT,
                    owner INTEGER, mileage_kmpl REAL, engine_cc REAL,
                    city TEXT, accident_history TEXT,
                    predicted_price_lakh REAL
                )
            """)
            # Migrate older databases without deleting existing prediction history.
            existing_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(predictions)")
            }
            if "city" not in existing_cols:
                conn.execute("ALTER TABLE predictions ADD COLUMN city TEXT")
            if "accident_history" not in existing_cols:
                conn.execute("ALTER TABLE predictions ADD COLUMN accident_history TEXT")

            conn.execute("""
                INSERT INTO predictions
                (predicted_at, brand, model, year, kms_driven, fuel_type,
                 transmission, owner, mileage_kmpl, engine_cc, city,
                 accident_history, predicted_price_lakh)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                selected_brand, selected_model, int(year_value), float(kms_value),
                str(fuel_value), str(transmission_value), int(owner_value),
                float(mileage_value), float(engine_value), str(city_value),
                str(accident_value), float(predicted_price)
            ))
            conn.commit()


        st.success(
            f"### 💰 Estimated Car Price: ₹ {predicted_price:.2f} Lakh"
        )

        st.info(
            "Price predicted using Random Forest Machine Learning model."
        )

        # Show selected car image
        show_car_image(
            selected_brand,
            selected_model, year_value
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
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Car Price Prediction | Machine Learning Project | Random Forest Regression"
)
# ```python
# =========================================================
# EMI CALCULATOR
# =========================================================

st.markdown("---")
st.header("💰 Car Loan EMI Calculator")
st.write("Debug - Saved price:", st.session_state.get("predicted_car_price", "Not saved"))
if "predicted_car_price" in st.session_state:

    car_price = float(st.session_state["predicted_car_price"])

    st.success("Using the price from your latest prediction.")
    st.metric(
        "🚗 Predicted Car Price",
        f"₹ {car_price:.2f} Lakh"
    )

    col1, col2 = st.columns(2)

    with col1:
        down_payment = st.number_input(
            "Down Payment (₹ Lakh)",
            min_value=0.0,
            value=0.0,
            step=0.5,
            key="emi_down_payment"
        )

    with col2:
        interest_rate = st.number_input(
            "Annual Interest Rate (%)",
            min_value=0.0,
            value=9.0,
            step=0.5,
            key="emi_interest_rate"
        )

    loan_years = st.selectbox(
        "Loan Period (Years)",
        [1, 2, 3, 4, 5, 6, 7],
        index=4,
        key="emi_loan_years"
    )

    if car_price <= 0:
        st.warning("Predicted car price must be greater than zero.")

    elif down_payment >= car_price:
        st.warning("Down payment must be less than the car price.")

    else:
        loan_amount = (car_price - down_payment) * 100000
        monthly_rate = interest_rate / 1200
        months = loan_years * 12

        if monthly_rate > 0:
            emi = (
                loan_amount
                * monthly_rate
                * (1 + monthly_rate) ** months
                / ((1 + monthly_rate) ** months - 1)
            )
        else:
            emi = loan_amount / months

        total_payment = emi * months
        total_interest = total_payment - loan_amount

        st.subheader("📊 EMI Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Monthly EMI", f"₹ {emi:,.0f}")
        col2.metric("Total Interest", f"₹ {total_interest:,.0f}")
        col3.metric("Loan Amount", f"₹ {loan_amount:,.0f}")

        st.info(f"Total Repayment: ₹ {total_payment:,.0f}")

        st.caption(
            "EMI is an estimate. Actual charges may vary "
            "depending on the bank and loan conditions."
        )

else:
    st.info(
        "Please select a car and click 'Predict Car Price' first. "
        "The predicted price will appear here automatically."
    )
# =========================================================
# ACTUAL VS PREDICTED PRICE
# =========================================================

st.markdown("---")
st.header("📊 Actual vs Predicted Price")

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

st.line_chart(comparison)




 # =========================================================
# COMPARE TWO CARS
# =========================================================

st.markdown("---")
st.header("🚗 Compare Two Cars")

st.write(
    "Select two cars to compare their features "
    "and estimated prices."
)

# Create car options from the existing dataset
car_options = list(range(len(X)))

def car_label(i):
    car = X.iloc[i]
    return (
        f"{car['Brand']} {car['Model']} "
        f"({int(car['Year'])})"
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("🚘 Car 1")

    car1_index = st.selectbox(
        "Select First Car",
        car_options,
        format_func=car_label,
        key="compare_car1"
    )

with col2:
    st.subheader("🚙 Car 2")

    car2_index = st.selectbox(
        "Select Second Car",
        car_options,
        index=min(1, len(car_options) - 1),
        format_func=car_label,
        key="compare_car2"
    )

if st.button(
    "🔍 Compare Cars",
    use_container_width=True,
    key="compare_button"
):

    if car1_index == car2_index:
        st.warning("Please select two different dataset records.")

    else:
        car1 = X.iloc[[car1_index]][feature_columns]
        car2 = X.iloc[[car2_index]][feature_columns]

        price1 = float(pipeline.predict(car1)[0])
        price2 = float(pipeline.predict(car2)[0])

        st.markdown("---")
        st.subheader("📊 Comparison Results")

        result1, result2 = st.columns(2)

        with result1:
            st.markdown(f"### 🚘 {car_label(car1_index)}")

            st.metric(
                "Estimated Price",
                f"₹{price1:.2f} Lakh"
            )

            st.write(
                f"**Brand:** {X.iloc[car1_index]['Brand']}"
            )
            st.write(
                f"**Year:** {int(X.iloc[car1_index]['Year'])}"
            )
            st.write(
                f"**Fuel:** {X.iloc[car1_index]['Fuel_Type']}"
            )
            st.write(
                f"**Transmission:** "
                f"{X.iloc[car1_index]['Transmission']}"
            )
            st.write(
                f"**Kilometers Driven:** "
                f"{X.iloc[car1_index]['Kms_Driven']:,.0f}"
            )

        with result2:
            st.markdown(f"### 🚙 {car_label(car2_index)}")

            st.metric(
                "Estimated Price",
                f"₹{price2:.2f} Lakh"
            )

            st.write(
                f"**Brand:** {X.iloc[car2_index]['Brand']}"
            )
            st.write(
                f"**Year:** {int(X.iloc[car2_index]['Year'])}"
            )
            st.write(
                f"**Fuel:** {X.iloc[car2_index]['Fuel_Type']}"
            )
            st.write(
                f"**Transmission:** "
                f"{X.iloc[car2_index]['Transmission']}"
            )
            st.write(
                f"**Kilometers Driven:** "
                f"{X.iloc[car2_index]['Kms_Driven']:,.0f}"
            )

        st.markdown("---")

        difference = abs(price1 - price2)

        st.metric(
            "Price Difference",
            f"₹{difference:.2f} Lakh"
        )

        if price1 < price2:
            st.success(
                f"{car_label(car1_index)} has the lower "
                "estimated price."
            )
        elif price2 < price1:
            st.success(
                f"{car_label(car2_index)} has the lower "
                "estimated price."
            )
        else:
            st.info("Both cars have the same estimated price.")

# =========================================================
# BRAND-WISE MARKET ANALYSIS
# =========================================================

st.markdown("---")
st.header("🏷️ Brand-wise Market Analysis")
st.caption("Dataset-based analysis of listed car prices; this is not a live market-price feed.")

brand_analysis = df.copy()
brand_analysis[target] = pd.to_numeric(brand_analysis[target], errors="coerce")
brand_analysis = brand_analysis.dropna(subset=["Brand", target])

if not brand_analysis.empty:
    brand_summary = (
        brand_analysis.groupby("Brand")
        .agg(
            Average_Price_Lakh=("Present_Price_Lakh", "mean"),
            Median_Price_Lakh=("Present_Price_Lakh", "median"),
            Cars_Listed=("Present_Price_Lakh", "size")
        )
        .sort_values("Average_Price_Lakh", ascending=False)
    )
    brand_options = sorted(brand_analysis["Brand"].astype(str).unique().tolist())
    selected_analysis_brands = st.multiselect(
        "Choose brands to compare (leave empty to show top 10)",
        brand_options,
        default=[]
    )
    if selected_analysis_brands:
        plot_data = brand_summary.loc[
            brand_summary.index.astype(str).isin(selected_analysis_brands)
        ].sort_values("Average_Price_Lakh", ascending=False)
    else:
        plot_data = brand_summary.head(10).sort_values("Average_Price_Lakh", ascending=False)

    st.subheader("Average listed price by brand (₹ lakh)")
    st.bar_chart(plot_data[["Average_Price_Lakh"]])
    st.subheader("Brand summary")
    st.dataframe(
        plot_data.rename(columns={
            "Average_Price_Lakh": "Average Price (Lakh)",
            "Median_Price_Lakh": "Median Price (Lakh)",
            "Cars_Listed": "Cars in Dataset"
        }).round(2),
        use_container_width=True
    )
else:
    st.info("No valid brand and price data available for analysis.")


# =========================================================
# PERMANENT PREDICTION HISTORY
# =========================================================

st.markdown("---")
st.header("🕘 Prediction History")
st.caption("History is stored locally in prediction_history.db in the same folder as app.py.")

history_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_history.db")
with sqlite3.connect(history_db) as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_at TEXT NOT NULL,
            brand TEXT, model TEXT, year INTEGER,
            kms_driven REAL, fuel_type TEXT, transmission TEXT,
            owner INTEGER, mileage_kmpl REAL, engine_cc REAL,
            city TEXT, accident_history TEXT,
            predicted_price_lakh REAL
        )
    """)
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(predictions)")
    }
    if "city" not in existing_cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN city TEXT")
    if "accident_history" not in existing_cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN accident_history TEXT")

    history_df = pd.read_sql_query(
        "SELECT predicted_at AS 'Date & Time', brand AS Brand, model AS Model, "
        "year AS Year, kms_driven AS 'Kms Driven', fuel_type AS 'Fuel Type', "
        "transmission AS Transmission, owner AS Owner, mileage_kmpl AS 'Mileage (km/l)', "
        "engine_cc AS 'Engine (cc)', city AS City, "
        "accident_history AS 'Accident History', "
        "predicted_price_lakh AS 'Predicted Price (Lakh)' "
        "FROM predictions ORDER BY id DESC",
        conn
    )

if history_df.empty:
    st.info("No saved predictions yet. Predict a car price to start your history.")
else:
    st.metric("Total saved predictions", len(history_df))
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    csv_history = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download prediction history as CSV",
        data=csv_history,
        file_name="car_prediction_history.csv",
        mime="text/csv"
    )
    if st.button("🗑️ Clear prediction history", type="secondary"):
        with sqlite3.connect(history_db) as conn:
            conn.execute("DELETE FROM predictions")
            conn.commit()
        st.success("Prediction history cleared. Refresh the page to update the table.")


# =========================================================
# FREE CAR PRICE CHATBOT
# =========================================================

import re

st.markdown("---")
st.header("💬 Car Price AI Assistant")

st.caption(
    "Ask questions about car prices, brands, budget, "
    "dataset and model performance."
)

# Initialize chat history
if "free_chat_history" not in st.session_state:
    st.session_state.free_chat_history = []


# ---------------------------------------------------------
# CHATBOT RESPONSE FUNCTION
# ---------------------------------------------------------

def car_chatbot_reply(question):

    q = question.lower().strip()

    # Check required columns
    required_columns = [
        "Brand",
        "Model",
        "Present_Price_Lakh"
    ]

    for col in required_columns:
        if col not in df.columns:
            return f"Required dataset column '{col}' is missing."

    # Clean price data
    data = df.copy()

    data["Present_Price_Lakh"] = pd.to_numeric(
        data["Present_Price_Lakh"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Brand", "Model", "Present_Price_Lakh"]
    )

    if data.empty:
        return "Sorry, valid car price data was not found."

    brand_prices = (
        data.groupby("Brand")["Present_Price_Lakh"]
        .mean()
        .sort_values()
    )

    # 1. Greeting
    if any(word in q for word in [
        "hello", "hi", "hey", "হ্যালো"
    ]):
        return (
            "Hello! 👋 Welcome to Car Price Assistant.\n\n"
            "You can ask me about car prices, brands, "
            "budget, dataset and model performance."
        )

    # 2. Help
    if "help" in q or "what can you do" in q:
        return (
            "I can help you with these questions:\n\n"
            "- Which brand has the highest average price?\n"
            "- Which brand has the lowest average price?\n"
            "- Show cars under 10 lakh.\n"
            "- Compare Maruti and Hyundai.\n"
            "- What is the dataset size?\n"
            "- What is the R2 score?\n"
            "- What is the MAE?\n"
            "- What is Random Forest?"
        )

    # 3. Dataset size
    if any(word in q for word in [
        "dataset size",
        "how many rows",
        "total cars",
        "number of cars",
        "dataset"
    ]):
        return (
            f"📊 Dataset Information\n\n"
            f"- Total records: {len(df)}\n"
            f"- Total columns: {len(df.columns)}\n"
            f"- Brands available: {data['Brand'].nunique()}\n"
            f"- Car models available: {data['Model'].nunique()}"
        )

    # 4. Highest average price brand
    if any(word in q for word in [
        "highest price",
        "most expensive brand",
        "highest average",
        "costliest brand",
        "most costly"
    ]):
        brand = brand_prices.idxmax()
        price = brand_prices.max()

        return (
            f"🏆 Highest Average Price Brand\n\n"
            f"Brand: {brand}\n\n"
            f"Average price: ₹{price:.2f} lakh"
        )

    # 5. Lowest average price brand
    if any(word in q for word in [
        "lowest price",
        "cheapest brand",
        "lowest average",
        "least expensive"
    ]):
        brand = brand_prices.idxmin()
        price = brand_prices.min()

        return (
            f"💰 Lowest Average Price Brand\n\n"
            f"Brand: {brand}\n\n"
            f"Average price: ₹{price:.2f} lakh"
        )

    # 6. Top 5 brands
    if any(word in q for word in [
        "top brands",
        "top 5",
        "brand ranking",
        "rank brands"
    ]):
        top_brands = brand_prices.sort_values(
            ascending=False
        ).head(5)

        result = "🏆 Top 5 Brands by Average Price\n\n"

        for i, (brand, price) in enumerate(
            top_brands.items(), start=1
        ):
            result += (
                f"{i}. {brand}: ₹{price:.2f} lakh\n\n"
            )

        return result

    # 7. Average car price
    if "average price" in q or "mean price" in q:
        avg_price = data["Present_Price_Lakh"].mean()

        return (
            f"📈 Average car price in the dataset: "
            f"₹{avg_price:.2f} lakh"
        )

          # 8. Cars within a budget range
    numbers = re.findall(r"\d+(?:\.\d+)?", q)

    is_range = (
        len(numbers) >= 2
        and any(sep in q for sep in ["-", "to", "between"])
    )

    if is_range:
        min_budget = float(numbers[0])
        max_budget = float(numbers[1])

        if "crore" in q:
            min_budget *= 100
            max_budget *= 100

        filtered = data[
            (data["Present_Price_Lakh"] >= min_budget)
            & (data["Present_Price_Lakh"] <= max_budget)
        ].copy()

        title = f"Cars between {min_budget:g} and {max_budget:g} lakh"

    elif any(word in q for word in [
        "under", "below", "within budget",
        "budget", "less than", "cheaper than",
        "above", "over", "more than", "greater than"
    ]):

        if not numbers:
            return "Please mention a budget, e.g. Show cars under 10 lakh."

        budget = float(numbers[0])

        if "crore" in q:
            budget *= 100

        is_above = any(word in q for word in [
            "above", "over", "more than", "greater than"
        ])

        if is_above:
            filtered = data[
                data["Present_Price_Lakh"] > budget
            ].copy()
            title = f"Cars above {budget:g} lakh"
        else:
            filtered = data[
                data["Present_Price_Lakh"] < budget
            ].copy()
            title = f"Cars under {budget:g} lakh"

    else:
        filtered = None

    if filtered is not None:

        filtered = filtered.drop_duplicates(
            subset=["Brand", "Model"]
        )

        filtered = filtered.sort_values("Present_Price_Lakh")

        if filtered.empty:
            return f"No cars found: {title}"

        result = (
            f"🚗 **{title}**\n\n"
            f"Total different car models: {len(filtered)}\n\n"
        )

        for _, row in filtered.iterrows():
            result += (
                f"- {row['Brand']} {row['Model']}: "
                f"₹{row['Present_Price_Lakh']:.2f} lakh\n"
            )

        return result
    # 9. Compare two brands
    if "compare" in q or "comparison" in q:
        available_brands = data["Brand"].dropna().unique()

        mentioned_brands = [
            brand for brand in available_brands
            if str(brand).lower() in q
        ]

        # Remove duplicate matches
        mentioned_brands = list(dict.fromkeys(
            mentioned_brands
        ))

        if len(mentioned_brands) >= 2:
            brand1 = mentioned_brands[0]
            brand2 = mentioned_brands[1]

            price1 = brand_prices.get(brand1)
            price2 = brand_prices.get(brand2)

            return (
                f"⚖️ Brand Comparison\n\n"
                f"**{brand1}**\n"
                f"Average price: ₹{price1:.2f} lakh\n\n"
                f"**{brand2}**\n"
                f"Average price: ₹{price2:.2f} lakh\n\n"
                f"Difference: ₹{abs(price1 - price2):.2f} lakh"
            )

        return (
            "Please mention two brand names exactly as they "
            "appear in your dataset.\n\n"
            "Example: Compare Maruti and Hyundai."
        )

    # 10. Model evaluation: R2
    if "r2" in q or "r²" in q or "r square" in q:
        try:
            return (
                f"📊 Model R² Score: {r2:.4f}\n\n"
                "R² measures how well the model explains "
                "variation in car prices. A value closer "
                "to 1 generally indicates a better fit."
            )
        except NameError:
            return (
                "R² score is not available. Please check "
                "whether the model evaluation code ran successfully."
            )

    # 11. MAE
    if "mae" in q or "mean absolute error" in q:
        try:
            return (
                f"📉 Model MAE: {mae:.4f} lakh\n\n"
                "MAE measures the average absolute difference "
                "between actual and predicted prices."
            )
        except NameError:
            return "MAE is not available in the current session."

    # 12. Algorithm explanation
    if any(word in q for word in [
        "random forest",
        "which algorithm",
        "algorithm used",
        "model used",
        "machine learning model"
    ]):
        return (
            "🌳 Algorithm: Random Forest Regressor\n\n"
            "Random Forest combines predictions from multiple "
            "decision trees to estimate car prices.\n\n"
            "It can learn relationships between features such "
            "as brand, year, mileage and engine capacity."
        )

    # 13. Available brands
    if "list brands" in q or "available brands" in q:
        brands = sorted(data["Brand"].dropna().unique())

        return (
            "🚘 Available Brands\n\n"
            + ", ".join(map(str, brands))
        )
           # Search for a specific car model
    search_query = q.strip().lower()

    # Remove common search words
    for word in [
        "tell me about",
        "search car",
        "find car",
        "show details of",
        "details of",
        "information about",
        "info about",
        "please",
        "show",
        "details",
        "information",
        "info",
        "car"
    ]:
        search_query = search_query.replace(word, "")

    search_query = search_query.strip()

    if search_query:
        matches = data[
            data["Model"].astype(str).str.lower().str.contains(
                re.escape(search_query),
                na=False
            )
        ]

        if not matches.empty:
            matches = matches.drop_duplicates(
                subset=["Brand", "Model"]
            ).head(10)

            result = "🚗 **Car Search Results**\n\n"

            for _, row in matches.iterrows():
                result += (
                    f"### {row['Brand']} {row['Model']}\n"
                    f"- Price: ₹{row['Present_Price_Lakh']:.2f} lakh\n"
                )

                for col in [
                    "Year",
                    "Fuel_Type",
                    "Transmission",
                    "Mileage_kmpl",
                    "Engine_cc",
                    "Kms_Driven"
                ]:
                    if col in row.index and pd.notna(row[col]):
                        result += f"- {col}: {row[col]}\n"

                result += "\n"

            return result

    # 14. Brand-specific average price
    for brand in sorted(
        data["Brand"].dropna().unique(),
        key=lambda x: len(str(x)),
        reverse=True
    ):
        if str(brand).lower() in q:
            avg_price = brand_prices.get(brand)

            if avg_price is not None:
                return (
                    f"🚗 Brand: {brand}\n\n"
                    f"Average price: ₹{avg_price:.2f} lakh"
                )

    # 15. Unknown question
    return (
        "Sorry, I couldn't understand that question yet. 🤔\n\n"
        "Try asking:\n"
        "- Which brand has the highest average price?\n"
        "- Show cars under 10 lakh.\n"
        "- Compare two brands.\n"
        "- What is the dataset size?\n"
        "- What is the R2 score?\n"
        "- What is the MAE?\n"
        "- Which algorithm is used?"
    )


# ---------------------------------------------------------
# CHAT INTERFACE
# ---------------------------------------------------------

# Display previous messages
for message in st.session_state.free_chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
user_question = st.chat_input(
    "Ask something about car prices..."
)

if user_question:
    # Save user message
    st.session_state.free_chat_history.append({
        "role": "user",
        "content": user_question
    })

    # Generate answer
    answer = car_chatbot_reply(user_question)

    # Save assistant response
    st.session_state.free_chat_history.append({
        "role": "assistant",
        "content": answer
    })

    # Refresh the page to display the new messages
    st.rerun()


# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.free_chat_history = []
    st.rerun()


st.caption(
    "ℹ️ This is a free rule-based chatbot. "
    "It does not use an external AI API, and its answers "
    "are based on the loaded dataset and supported questions."
)
