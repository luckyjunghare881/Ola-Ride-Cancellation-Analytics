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

## � How It Works — End-to-End Working Description

### Data Flow Architecture

```
 ┌──────────────────────────────┐
 │     USER INPUT               │
 │  CSV Upload / Load Demo Data │
 └──────────────┬───────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │   DATA PROCESSING            │
 │   (src/data_processing.py)   │
 │                              │
 │  1. Validate CSV schema      │
 │  2. Remove duplicates        │
 │  3. Parse dates & types      │
 │  4. Fill missing values      │
 │  5. Engineer features:       │
 │     BookingHour, DayOfWeek,  │
 │     Month, IsWeekend,        │
 │     TimePeriod, IsCanceled   │
 │  6. Compute KPIs             │
 └──────────────┬───────────────┘
                │
      Stores in Streamlit Session State:
      df_clean, kpis, cleaning_report
                │
   ┌────────────┼────────────┬──────────────┐
   ▼            ▼            ▼              ▼
┌────────┐ ┌────────┐ ┌──────────┐ ┌────────────┐
│ HOME   │ │  EDA   │ │ SQL PAGE │ │  ML PAGE   │
│Dashboard│ │Insights│ │  Engine  │ │ Train &    │
│ KPIs,  │ │ Charts │ │Pre-built │ │  Predict   │
│ Alerts │ │Filters │ │& Custom  │ │ Clustering │
│ Trends │ │Heatmaps│ │ Queries  │ │            │
└────────┘ └────────┘ └──────────┘ └────────────┘
                                          │
                                          ▼
                                   ┌────────────┐
                                   │  RECO PAGE │
                                   │ AI-Powered │
                                   │ Action Plan│
                                   │   Export   │
                                   └────────────┘
```

### Step-by-Step Workflow

#### Step 1: Data Loading
When you launch the app, you have two options:
- **Upload CSV**: Upload your own OLA ride data file. The app validates that required columns (`BookingID`, `BookingDate`, `RideStatus`) exist.
- **Load Demo Data**: Generates 12,000 synthetic ride records with realistic distributions for locations (Koramangala, Indiranagar, Whitefield, etc.), vehicle types (Mini, Sedan, SUV, Auto, Bike), payment modes, ratings, ETAs, and cancellation reasons.

#### Step 2: Data Cleaning & Feature Engineering
The `preprocess()` function automatically:
- Removes exact duplicate rows
- Parses `BookingDate` into datetime format
- Fills missing numeric values (fare, distance, ratings, ETA) with column medians
- Creates derived features: `BookingHour`, `DayOfWeek`, `Month`, `IsWeekend`, `TimePeriod` (Morning/Afternoon/Evening/Night), and binary `IsCanceled` flag
- Generates a cleaning report showing what was modified

#### Step 3: KPI Computation
`compute_kpis()` calculates all dashboard metrics:
- Total bookings, completed rides, canceled rides, cancellation rate (%)
- Driver-canceled vs customer-canceled counts and rates
- Average fare, ride distance, driver/customer ratings
- Revenue loss estimate from canceled rides
- Completion rate and average ETA

#### Step 4: Page Navigation & Analysis

**Home Dashboard** renders KPI cards with trend indicators, a daily completion-vs-cancellation bar chart, a donut chart for driver/customer cancellation breakdown, cancel reason analysis, day×hour heatmap, and a data preview table with export options. It also displays a red alert banner if any pickup location exceeds 30% cancellation rate.

**EDA Insights** provides interactive Plotly visualizations with filters for ride status, vehicle type, payment mode, location, and date range. Charts include hourly/daily/monthly trends, location hotspot analysis, payment & vehicle type breakdowns, ETA distribution by ride status, and cancel reason Pareto charts.

**SQL Query Engine** loads the cleaned data into an in-memory SQLite database. Users can run 11 pre-built analytical queries (cancellation summary, hourly trends, top 10 hotspots, payment mode analysis, revenue loss, ETA correlation, etc.) or write custom SQL. Only `SELECT` queries are allowed — destructive operations (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `CREATE`) are blocked. Results auto-visualize as bar/line charts and can be exported as CSV.

**ML Predictions** offers three tabs:
1. **Train & Evaluate**: Select RandomForest or GradientBoosting, set test split (10–40%), and train. Displays accuracy, precision, recall, F1, ROC-AUC, cross-validation score, feature importance chart, confusion matrix, and probability distribution.
2. **Predict New Booking**: Enter ride parameters (hour, distance, fare, ratings, ETA, vehicle type, payment mode, location) and get a cancellation risk prediction: **Low** (<30%), **Medium** (30–70%), or **High** (>70%) with a visual risk gauge.
3. **Cluster Analysis**: Runs KMeans clustering on location data to identify cancellation hotspot clusters, displayed as a scatter plot with cluster characteristics.

**AI Recommendations** analyzes computed KPIs through a rule-based engine that generates 8+ categories of actionable insights: critical cancellation rate alerts, driver accountability measures, customer retention strategies, peak hour optimization, location hotspot solutions, payment mode improvements, ETA reduction tactics, and vehicle rebalancing. Each recommendation includes a priority badge (Critical/High/Medium/Low), data-driven insight, specific action, and estimated business impact. A 30-day action plan is generated, and the full report can be exported as TXT or CSV.

#### Step 5: Export & Reporting
Every page supports data export:
- Home: Download cleaned data or canceled-only data as CSV
- EDA: Download filtered analysis results
- SQL: Download query results as CSV
- ML: View and save model metrics
- Recommendations: Export full report as TXT or CSV with KPI metrics and action items

---

## �🛠️ Tech Stack

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
