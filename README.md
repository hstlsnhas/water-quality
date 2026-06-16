# 💧 Water Quality Prediction & Treatment Recommendation System

*(WTP ITB Jatinangor)*

An intelligent web-based system that combines **Machine Learning** and **Generative AI (Gemini 2.5 Flash)** to monitor, predict, and recommend treatment actions for water quality in a Water Treatment Plant (WTP).

---

## 🚀 Why This Project Matters

Ensuring safe and clean water distribution requires accurate monitoring and fast decision-making. This project provides:

* ⚡ Real-time **water quality classification**
* 🌊 Monitoring of **pre- and post-filtration conditions**
* 🧠 AI-powered **treatment recommendations**
* 📊 Interactive dashboard for operational insights

👉 Result: A smart decision-support tool for improving water treatment efficiency and safety.

---

## 💡 Key Features

* **Automated Data Ingestion**

  * Downloads datasets, models, and scalers from Google Drive using `gdown`

* **Interactive Dashboard**

  * Visualizes water quality distribution (*Clean vs Non-Clean*) using Plotly

* **Model Evaluation Insights**

  * Compares performance of:

    * XGBoost
    * Logistic Regression
    * Support Vector Machine (SVM)
  * Includes confusion matrix visualization

* **Smart Prediction Simulator**

  * Real-time input of sensor parameters:

    * Flow1, Flow2, Turbidity, pH, TDS
  * Automatically selects prediction path (Panel A or Panel B)

* **AI-Powered Recommendations**

  * Uses **Gemini 2.5 Flash** to generate:

    * Water condition analysis
    * Structured treatment recommendations
    * Final conclusions

---

## 🔄 System Workflow

```id="workflow1"
User Input → Routing Logic → ML Prediction → GenAI Analysis → Structured Output
```

### Detailed Flow:

1. **Input Parameters**

   * Flow1, Flow2, Turbidity, pH, TDS

2. **Routing Logic**

   * If `Flow2 > 0` → Post-Filtration (Panel B)
   * Else → Pre-Filtration (Panel A)

3. **Machine Learning Prediction**

   * XGBoost model classifies water quality:

     * Clean (Putih)
     * Non-Clean

4. **Generative AI Processing**

   * Sensor data + prediction sent to Gemini API

5. **Output**

   * Interpretation
   * Treatment recommendations
   * Final summary

---

## 🧠 Key Strengths

* ✅ Hybrid system (ML + Generative AI)
* ✅ Real-world industrial use case (Water Treatment Plant)
* ✅ Intelligent routing system (Panel A vs Panel B)
* ✅ Focus on safety (optimized to minimize false negatives)
* ✅ Interactive and user-friendly dashboard

---

## 📊 Model Selection

XGBoost was selected due to:

* Highest overall performance
* Lowest **False Negative rate** (critical for water safety)

---

## 🛠️ Tech Stack

* **Frontend & App:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Visualization:** Plotly, Matplotlib
* **Generative AI:** Google GenAI SDK (Gemini 2.5 Flash)

---

## ▶️ Run Locally

```bash id="run1"
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎯 Use Cases

* Water treatment plant monitoring
* Environmental quality control
* Smart infrastructure systems
* AI-assisted operational decision-making

---

## 👤 Author

Portfolio project focused on **Data Science, Machine Learning, and AI-powered decision systems**.
