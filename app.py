import streamlit as st
import pickle
import pandas as pd

# File paths
MODEL_FILE = "model_gb_tuned.pkl"
COLUMNS_FILE = "columns.pkl"
ENCODERS_FILE = "encoders.pkl"
# Page config
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")
st.title("🏦 SME Credit Risk Predictor")

# Load model and config
@st.cache_resource
def load_model_and_config():
    """Load model, column names, and encoders"""
    try:
        model = pickle.load(open(MODEL_FILE, "rb"))
        columns = pickle.load(open(COLUMNS_FILE, "rb"))
        encoders = pickle.load(open(ENCODERS_FILE, "rb"))
        return model, columns, encoders
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        st.stop()

model, expected_columns, encoders = load_model_and_config()

# Create sidebar for inputs
st.sidebar.header("Loan Application Details")

# Numeric inputs
loan = st.sidebar.number_input(
    "💰 Loan Amount ($)",
    min_value=0.0,
    value=50000.0,
    step=1000.0,
    help="Principal loan amount"
)

term = st.sidebar.number_input(
    "📅 Loan Term (months)",
    min_value=1,
    max_value=360,
    value=60,
    help="Loan duration in months"
)

rate = st.sidebar.number_input(
    "📊 Interest Rate (%)",
    min_value=0.0,
    value=5.0,
    step=0.1,
    help="Annual interest rate"
)

jobs = st.sidebar.number_input(
    "👥 Jobs Supported",
    min_value=0,
    value=10,
    help="Number of jobs created/retained"
)

# Revolver status (numeric)
revolver = st.sidebar.selectbox(
    "🔄 Revolver Status",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
    help="Is this a revolving loan?"
)

# Business type - Get from actual training data
business_type_options = [x for x in sorted(encoders['businesstype'].classes_.tolist()) if x and x.strip()]
if business_type_options:
    business_type = st.sidebar.selectbox(
        "🏢 Business Type",
        business_type_options,
        index=0,
        format_func=lambda x: x.title()
    )
else:
    st.sidebar.warning("No business types available")
    business_type = "Unknown"

# NAICS description - Get from actual training data
naics_options = [x for x in sorted(encoders['naicsdescription'].classes_.tolist()) if x and x.strip()]
if naics_options:
    naics_description = st.sidebar.selectbox(
        "📋 Industry Classification",
        naics_options,
        index=0,
        format_func=lambda x: x.title()
    )
else:
    st.sidebar.warning("No industries available")
    naics_description = "Unknown"

# Collateral - Get from encoder
collateral_options = [x for x in sorted(encoders['collateralind'].classes_.tolist()) if x and x.strip()]
if collateral_options:
    collateral = st.sidebar.selectbox(
        "🔐 Collateral Available",
        collateral_options,
        index=0,
        format_func=lambda x: x.title()
    )
else:
    st.sidebar.warning("No collateral options available")
    collateral = "Unknown"

# Prediction button
if st.sidebar.button("🔮 Predict Risk", use_container_width=True):
    try:
        # Create input dataframe
        input_df = pd.DataFrame([{
            'grossapproval': loan,
            'terminmonths': term,
            'initialinterestrate': rate,
            'jobssupported': jobs,
            'collateralind': collateral,
            'revolverstatus': revolver,
            'businesstype': business_type,
            'naicsdescription': naics_description
        }])

        # Validate all required columns are present
        missing_cols = set(expected_columns) - set(input_df.columns)
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
            st.stop()

        # Reorder columns to match training data
        input_df = input_df[expected_columns]

        # Encode categorical features using saved encoders
        for col in encoders.keys():
            if col in input_df.columns:
                input_df[col] = encoders[col].transform(input_df[col].astype(str))

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # Display results
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("### 🚨 HIGH RISK")
                risk_label = "High Risk"
                confidence = prediction_proba[1] * 100
            else:
                st.success("### ✅ LOW RISK")
                risk_label = "Low Risk"
                confidence = prediction_proba[0] * 100

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%",
                help="Model confidence in this prediction"
            )

        # Detailed probability breakdown
        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Risk Level': ['Low Risk', 'High Risk'],
            'Probability': [prediction_proba[0] * 100, prediction_proba[1] * 100]
        })
        st.bar_chart(prob_df.set_index('Risk Level'))

        # Application summary
        st.subheader("Application Summary")
        summary_df = pd.DataFrame({
            'Parameter': ['Loan Amount', 'Term (months)', 'Interest Rate', 'Jobs Supported', 'Collateral', 'Revolver Status'],
            'Value': [f"${loan:,.2f}", f"{term}", f"{rate}%", f"{jobs}", collateral, "Yes" if revolver == 1 else "No"]
        })
        st.table(summary_df)

    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

# Footer
st.divider()
st.caption("💡 **Note:** This model predicts credit risk based on historical SBA loan data. Always conduct additional due diligence.")
