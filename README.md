# 🚗 OLA Ride Cancellation Analytics

A **full-stack Streamlit dashboard** for analyzing OLA ride booking data, uncovering cancellation patterns, running SQL queries, training ML models for cancellation prediction, and generating AI-powered business recommendations.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.5+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ Features

### 📊 KPI Dashboard
- Total bookings, completed rides, canceled rides, cancellation rates
- Driver vs customer cancellation breakdown
- Average fare, ride distance, driver ratings, estimated revenue loss

### 📈 EDA Insights (Interactive Plotly Charts)
- Cancellation by hour, day of week, month (time-series trends)
- Location-level cancellation hotspots
- Payment mode & vehicle type analysis
- Day × Hour heatmaps, ETA distribution by status
- Cancel reason breakdown

### 🗃️ SQL Query Engine
- 11 pre-built analytical queries (cancellation summary, hourly trends, hotspots, etc.)
- Custom SQL query editor with column browser
- Auto-visualization of query results
- Query result download as CSV

### 🤖 ML Predictions
- **RandomForest** and **Gradient Boosting** classifiers for cancellation prediction
- Training metrics: Accuracy, Precision, Recall, F1, ROC AUC, cross-validation
- Feature importance visualization
- Confusion matrix and probability distribution charts
- **Single booking prediction**: Input ride details → get cancellation risk (Low/Medium/High)
- **KMeans clustering** on cancellation hotspots by location

### 💡 AI Recommendations
- Rule-based recommendation engine generating actionable insights
- Executive summary with KPI overview
- Priority-based action items (Critical / High / Medium / Low)
- 30-day action plan
- Export report as TXT or CSV

### 🔧 Additional Features
- CSV upload with auto-cleaning (duplicates, missing values, type parsing)
- Demo dataset generation (12,000 synthetic rows)
- User roles: Analyst (full access) / Business User (demo data + recommendations)
- Paginated data tables, interactive filters
- Data export (cleaned CSV, canceled-only CSV)
- Responsive design for desktop and mobile

---

## 📁 Project Structure

```
Ola Ride Cancellation Analytics/
├── app.py                      # Main Streamlit entry point
├── requirements.txt            # Python dependencies
├── .gitignore
├── README.md
├── .streamlit/
│   └── config.toml             # Streamlit theme & server config
├── pages/
│   ├── __init__.py
│   ├── home.py                 # Home & Upload page
│   ├── eda.py                  # EDA Insights page
│   ├── sql_page.py             # SQL Query Engine page
│   ├── ml_page.py              # ML Predictions page
│   └── reco_page.py            # Recommendations page
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # CSV loading, cleaning, KPIs
│   ├── generate_data.py        # Synthetic dataset generator
│   ├── ml_model.py             # ML training, prediction, clustering
│   ├── recommendations.py      # AI recommendation engine
│   ├── sql_engine.py           # SQLite query engine
│   └── utils.py                # Export & utility functions
└── tests/
    ├── __init__.py
    ├── test_data_processing.py # Data processing unit tests
    ├── test_ml_model.py        # ML model unit tests
    └── test_sql_engine.py      # SQL engine unit tests
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/luckyjunghare881/Ola-Ride-Cancellation-Analytics.git
   cd Ola-Ride-Cancellation-Analytics
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   The app will open at `http://localhost:8501`

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_processing.py -v
pytest tests/test_ml_model.py -v
pytest tests/test_sql_engine.py -v
```

---

## 📊 Using the Dashboard

### Quick Start (Demo Data)
1. Launch the app with `streamlit run app.py`
2. Click **"Load Demo Data"** in the sidebar
3. Navigate between pages using the sidebar menu

### Upload Your Own Data
1. Go to **Home & Upload** page
2. Upload a CSV file with these columns:
   - `BookingID`, `BookingDate`, `RideStatus` (required)
   - `VehicleType`, `PickupLocation`, `DropLocation`, `PaymentMode`
   - `RideDistance_km`, `BookingValue_INR`, `DriverRating`, `CustomerRating`
   - `ETA_Pickup_min`, `RideDuration_min`, `CancelReason`
3. The app auto-cleans and preprocesses your data

### Generate Sample Dataset
```bash
python src/generate_data.py
```
This creates a `data/sample_ola_rides.csv` file with 12,000 synthetic rows.

---

## 🌐 Deployment

### Streamlit Community Cloud
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path: `app.py`
5. Deploy

### Vercel (Serverless)
1. Install Vercel CLI: `npm i -g vercel`
2. Add a `vercel.json`:
   ```json
   {
     "builds": [{"src": "app.py", "use": "@vercel/python"}],
     "routes": [{"src": "/(.*)", "dest": "app.py"}]
   }
   ```
3. Deploy: `vercel --prod`

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🛠️ Tech Stack

| Component           | Technology                          |
|---------------------|-------------------------------------|
| Frontend/Dashboard  | Streamlit                           |
| Visualization       | Plotly                              |
| Data Processing     | Pandas, NumPy                       |
| ML/AI               | Scikit-learn (RandomForest, GBM, KMeans) |
| Database            | SQLite (in-memory)                  |
| Model Persistence   | Joblib                              |
| Testing             | Pytest                              |

---

## 📝 Dataset Schema

| Column            | Type     | Description                              |
|-------------------|----------|------------------------------------------|
| BookingID         | String   | Unique booking identifier                |
| BookingDate       | DateTime | Date and time of booking                 |
| RideStatus        | String   | Completed / Canceled by Driver / Customer|
| VehicleType       | String   | Mini, Sedan, SUV, Auto, Bike             |
| PickupLocation    | String   | Pickup area name                         |
| DropLocation      | String   | Drop-off area name                       |
| PaymentMode       | String   | Cash, UPI, Credit Card, Debit Card, Wallet|
| RideDistance_km    | Float    | Trip distance in kilometers              |
| BookingValue_INR  | Float    | Fare amount in INR                       |
| DriverRating      | Float    | Driver rating (1-5)                      |
| CustomerRating    | Float    | Customer rating (1-5)                    |
| ETA_Pickup_min    | Float    | Estimated time to pickup (minutes)       |
| RideDuration_min  | Float    | Actual ride duration (null if canceled)   |
| CancelReason      | String   | Reason for cancellation (null if completed)|

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Lucky Junghare**  
GitHub: [@luckyjunghare881](https://github.com/luckyjunghare881)
