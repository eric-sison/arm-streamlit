# Market Basket Analysis Dashboard

A Streamlit-based web application for analyzing customer purchasing patterns and discovering product associations using association rule mining.

## 🎯 Features

- **Interactive Data Analysis:**

  - Association rule mining using Apriori algorithm
  - Product co-occurrence visualization
  - Rule strength metrics (Support, Confidence, Lift)
  - Dynamic filtering and search capabilities

- **Visualizations:**

  - Product Association Heatmap
  - Rule Metrics Scatter Plot
  - Rule Visualization Network
  - Metric Gauges for detailed analysis

- **Data Insights:**
  - Product purchase frequency analysis
  - Itemset size distribution
  - Detailed rule interpretation
  - Business recommendations

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/eric-sison/arm-streamlit.git
cd arm-streamlit
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
streamlit run app.py
```

### Docker Installation (Optional)

1. Build the Docker image:

```bash
docker build -t market-basket-analysis .
```

2. Run the container:

```bash
docker run -p 8501:8501 market-basket-analysis
```

## 📊 Data Format

The application expects a CSV file with the following format:

- Each row represents a transaction
- Each column represents a product
- Values should be boolean (True/False) or binary (1/0)

Example:

```csv
Product1,Product2,Product3
True,False,True
False,True,False
True,True,False
```

---

**Note:** This project is for educational and analytical purposes. Ensure you have the right to use any data you analyze with this tool.
