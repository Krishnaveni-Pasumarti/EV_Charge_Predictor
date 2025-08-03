import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="EV Forecast", layout="wide")

# Load model
model = joblib.load("C:/Users/krish/Downloads/EV_Charge_Predictor-main/EV_Charge_Predictor-main/forecasting_ev_model.pkl")

# Styling
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #c2d3f2, #7f848a);
        }
        h1, h2, h3, h4, h5, h6, p {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🔮 EV Adoption Forecaster</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Washington State County-Level Forecast (Next 3 Years)</h3>", unsafe_allow_html=True)
st.image("ev-car-factory.jpg", use_container_width=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Select county
counties = sorted(df['County'].dropna().unique())
county = st.selectbox("Choose a county", counties)

county_df = df[df['County'] == county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]

# Forecast
history = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative = list(np.cumsum(history))
months = county_df['months_since_start'].max()
last_date = county_df['Date'].max()

forecast_data = []
horizon = 36

for i in range(1, horizon + 1):
    forecast_date = last_date + pd.DateOffset(months=i)
    months += 1
    lag1, lag2, lag3 = history[-1], history[-2], history[-3]
    roll_mean = np.mean([lag1, lag2, lag3])
    pct1 = (lag1 - lag2) / lag2 if lag2 else 0
    pct3 = (lag1 - lag3) / lag3 if lag3 else 0
    slope = np.polyfit(range(6), cumulative[-6:], 1)[0] if len(cumulative) >= 6 else 0

    features = pd.DataFrame([{
        'months_since_start': months,
        'county_encoded': county_code,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll_mean,
        'ev_total_pct_change_1': pct1,
        'ev_total_pct_change_3': pct3,
        'ev_growth_slope': slope
    }])

    prediction = model.predict(features)[0]
    forecast_data.append({'Date': forecast_date, 'Predicted EV Total': round(prediction)})

    history.append(prediction)
    history.pop(0)
    cumulative.append(cumulative[-1] + prediction)
    cumulative.pop(0)

# Plot
historical = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical['Cumulative EV'] = historical['Electric Vehicle (EV) Total'].cumsum()
historical['Source'] = 'Historical'

forecast_df = pd.DataFrame(forecast_data)
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical['Cumulative EV'].iloc[-1]
forecast_df['Source'] = 'Forecast'

combined = pd.concat([
    historical[['Date', 'Cumulative EV', 'Source']],
    forecast_df[['Date', 'Cumulative EV', 'Source']]
])

# Plot
st.subheader(f"📊 Cumulative EV Forecast for {county}")
fig, ax = plt.subplots(figsize=(12, 6))
for label, grp in combined.groupby("Source"):
    ax.plot(grp['Date'], grp['Cumulative EV'], label=label, marker='o')
ax.set_title(f"EV Adoption Trend in {county} (3-Year Forecast)", fontsize=14, color='white')
ax.set_facecolor("#1c1c1c")
fig.patch.set_facecolor('#1c1c1c')
ax.grid(True, alpha=0.3)
ax.tick_params(colors='white')
ax.set_xlabel("Date", color='white')
ax.set_ylabel("Cumulative EVs", color='white')
ax.legend()
st.pyplot(fig)

# Growth summary
initial = historical['Cumulative EV'].iloc[-1]
final = forecast_df['Cumulative EV'].iloc[-1]
if initial > 0:
    growth_pct = ((final - initial) / initial) * 100
    st.success(f"EV adoption in **{county}** is forecasted to **increase by {growth_pct:.2f}%** in the next 3 years.")
else:
    st.warning("No historical EV data available to compute growth.")

# Multiple county comparison
st.markdown("---")
st.header("Compare up to 3 Counties")
selected = st.multiselect("Select counties", counties, max_selections=3)

if selected:
    comparison = []
    for name in selected:
        c_df = df[df['County'] == name].sort_values("Date")
        c_code = c_df['county_encoded'].iloc[0]

        h = list(c_df['Electric Vehicle (EV) Total'].values[-6:])
        cum = list(np.cumsum(h))
        m = c_df['months_since_start'].max()
        d = c_df['Date'].max()

        preds = []
        for _ in range(horizon):
            d += pd.DateOffset(months=1)
            m += 1
            l1, l2, l3 = h[-1], h[-2], h[-3]
            rm = np.mean([l1, l2, l3])
            p1 = (l1 - l2) / l2 if l2 else 0
            p3 = (l1 - l3) / l3 if l3 else 0
            sl = np.polyfit(range(6), cum[-6:], 1)[0] if len(cum) >= 6 else 0

            features = pd.DataFrame([{
                'months_since_start': m,
                'county_encoded': c_code,
                'ev_total_lag1': l1,
                'ev_total_lag2': l2,
                'ev_total_lag3': l3,
                'ev_total_roll_mean_3': rm,
                'ev_total_pct_change_1': p1,
                'ev_total_pct_change_3': p3,
                'ev_growth_slope': sl
            }])

            y = model.predict(features)[0]
            preds.append({'Date': d, 'Predicted EV Total': round(y)})

            h.append(y)
            h.pop(0)
            cum.append(cum[-1] + y)
            cum.pop(0)

        hist = c_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist['Cumulative EV'] = hist['Electric Vehicle (EV) Total'].cumsum()
        fcst = pd.DataFrame(preds)
        fcst['Cumulative EV'] = fcst['Predicted EV Total'].cumsum() + hist['Cumulative EV'].iloc[-1]

        both = pd.concat([
            hist[['Date', 'Cumulative EV']],
            fcst[['Date', 'Cumulative EV']]
        ])
        both['County'] = name
        comparison.append(both)

    final_df = pd.concat(comparison)

    # Plot
    st.subheader("📈 Multi-County Forecast Comparison")
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    for county_name, grp in final_df.groupby("County"):
        ax2.plot(grp['Date'], grp['Cumulative EV'], label=county_name, marker='o')
    ax2.set_title("EV Adoption Forecasts", fontsize=16, color='white')
    ax2.set_facecolor("#1c1c1c")
    fig2.patch.set_facecolor("#1c1c1c")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='white')
    ax2.set_xlabel("Date", color='white')
    ax2.set_ylabel("Cumulative EVs", color='white')
    ax2.legend(title="County")
    st.pyplot(fig2)

    # % growth text summary
    summary = []
    for name in selected:
        rows = final_df[final_df['County'] == name].reset_index()
        initial = rows['Cumulative EV'].iloc[len(rows) - horizon - 1]
        final = rows['Cumulative EV'].iloc[-1]
        if initial > 0:
            g = ((final - initial) / initial) * 100
            summary.append(f"{name}: {g:.2f}%")
        else:
            summary.append(f"{name}: No data")
    st.success("Forecasted growth: " + " | ".join(summary))

# Footer
st.markdown("Prepared for the **AICTE Internship Cycle 2 by Sara Fareen K**")
