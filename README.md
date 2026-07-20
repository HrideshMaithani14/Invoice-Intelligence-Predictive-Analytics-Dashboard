
# 💼 Invoice Intelligence & Predictive Analytics Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link-here.streamlit.app/) <!-- Add your live link here -->

An end-to-end machine learning web application built to automate financial anomaly detection and forecast shipping expenses. This project takes custom-trained scikit-learn models out of the Jupyter Notebook and deploys them into an interactive, user-friendly Streamlit dashboard.

## 🚀 Features

* **🚢 Freight Cost Predictor:** 
  * Input single or bulk transaction amounts (in USD).
  * Generates instant freight cost estimates using a trained regression model.
  * Visualizes the predicted cost breakdown using interactive Plotly charts.
* **⚠️ Invoice Anomaly Flagger:** 
  * Acts as a financial gatekeeper by analyzing 5 key metrics: *Invoice Quantity, Invoice Dollars, Freight, Total Item Quantity,* and *Total Item Dollars*.
  * Automatically flags statistically irregular invoices that require manual review.
  * Displays feature weight distribution for flagged anomalies.

## 🛠️ Tech Stack

* **Frontend/UI:** [Streamlit](https://invoice-intelligence-predictive-analytics-dashboard.streamlit.app/)
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, Joblib
* **Data Visualization:** Plotly Express

## 📂 Project Structure

```text
├── Models/
│   └── best_models/
│       ├── freight.pkl               # Trained freight prediction model
│       └── Flagged_invoice.pkl       # Trained anomaly detection model
├── interface/
│   ├── Predict_freight.py            # Backend logic for freight model
│   └── predict_invoice_flag.py       # Backend logic for invoice model
├── app.py                            # Main Streamlit application
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation

```

## 💻 Local Setup & Installation

To run this project locally on your machine, follow these steps:

**1. Clone the repository**

```bash
git clone [(https://github.com/HrideshMaithani14/Invoice-Intelligence-Predictive-Analytics-Dashboard.git))
cd your-repo-name

```

**2. Install dependencies**
Make sure you have Python installed. Then, install the required packages:

```bash
pip install -r requirements.txt

```

**3. Run the application**

```bash
streamlit run app.py

```

*The app will automatically open in your default web browser at `http://localhost:8501`.*

## 📈 The "XYZ" Impact

Developed a **Predictive Analytics Dashboard [X]** capable of real-time financial risk assessment and bulk freight estimation **[Y]**, by engineering multi-feature scikit-learn models and deploying them through a responsive Streamlit web interface **[Z]**.

## 🤝 Connect with Me

* **LinkedIn:** [Hridesh Maithani](inkedin.com/in/hridesh-maithani-00a059391/?skipRedirect=true)
* **Portfolio:** [Portfolio Website]([https://www.google.com/search?q=https://your-portfolio.com](https://hrideshmaithaidatascientist.netlify.app/))

```

<FollowUp label="Need a requirements.txt file for your repo?" query="Generate the requirements.txt file for this Streamlit ML project."/>

```
