
---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Rider-04/cab_bike_auto_fare_prediction.git
cd cab_bike_auto_fare_prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
- Open `Merging_datasets.ipynb` to reproduce the merged dataset
- Open `Model.ipynb` to train and evaluate the fare prediction model

---

## 📈 Key Learnings

- Standardizing heterogeneous, multi-source real-world data into a single usable schema
- Handling missing/inconsistent fields across independently collected datasets
- Feature engineering from raw timestamps and categorical ride attributes
- Building a regression pipeline for fare prediction

---

## 🚀 Future Improvements

- Add geolocation-based distance calculation for datasets missing distance data
- Deploy the model via a FastAPI backend for real-time fare prediction
- Expand to include live traffic/weather-based fare adjustments
- Add a RAG-based chatbot interface for querying fare estimates conversationally

---

## 👤 Author

**Parth Sharma**
Data Scientist | ML/AI Enthusiast
[LinkedIn](https://linkedin.com/in/parthsharma-dsxpert) · [Portfolio](https://parth-portfolio-virid.vercel.app)

---

## 📄 License

See [LICENSE](LICENSE) file.