# 🤖 Machine Learning Project Portfolio

A comprehensive Machine Learning project showcasing four distinct predictive models integrated into a single Streamlit web application. This project demonstrates various ML algorithms implemented from scratch, including K-Means clustering, Linear Regression, KNN, and SVM.

## 👥 Project Team

- **Abdelali Saadali** - Company Classification AI
- **Marouane Mounir** - Salary Prediction (Morocco IT Market)
- **Asaad FETHALLAH** - Employee Salary Prediction
- **Muhammad Irfan Wahyudi** - Bankruptcy Prediction

---

## 📊 Projects Overview

### 1. 🏢 Company Classification AI
**Developer:** Abdelali Saadali

**Goal:** Classify companies based on worker data patterns extracted from CNSS (Caisse Nationale de Sécurité Sociale) declarations.

**Algorithm:** K-Means Clustering (implemented from scratch)

**Key Features:**
- Analyzes company workforce stability
- Identifies seasonal employment patterns
- Detects potential anomalies in company data
- Classifies companies into 4 categories:
  - **Stable Companies**: Consistent work days, high full-time ratio
  - **Seasonal Companies**: Fewer work days, lower pay (part-time/seasonal)
  - **Irregular Companies**: High variance in work days, mixed workforce
  - **Potentially Fraudulent Companies**: Anomalous data patterns

**Features Used:**
- Number of workers
- Average working days
- Standard deviation of working days
- Average salary
- Standard deviation of salary
- Full-time worker ratio

---

### 2. 🇲🇦 Salary Prediction (Morocco IT Market)
**Developer:** Marouane Mounir

**Goal:** Predict IT salaries in the Moroccan job market for 2025.

**Algorithm:** Linear Regression with Gradient Descent (implemented from scratch)

**Key Features:**
- Predicts salaries based on:
  - Professional profile
  - Years of experience
  - Education level
  - Technology stack
  - Target company
- Tailored for the Moroccan IT job market
- Uses gradient descent optimization

---

### 3. 💵 Employee Salary Prediction
**Developer:** Asaad FETHALLAH

**Goal:** Predict employee salaries based on historical payroll data.

**Algorithm:** Linear Regression

**Key Features:**
- Predicts individual employee salaries
- Takes into account:
  - Employee ID
  - Month and year
  - Historical salary patterns
- Useful for HR planning and budgeting

---

### 4. 🏢 Bankruptcy Prediction
**Developer:** Muhammad Irfan Wahyudi

**Goal:** Predict company bankruptcy risk using financial indicators.

**Algorithms:** 
- K-Nearest Neighbors (KNN) - implemented from scratch
- Support Vector Machine (SVM) - implemented from scratch

**Key Features:**
- Analyzes financial stability metrics
- Evaluates debt risk indicators
- Considers regional economic factors
- Binary classification: Bankruptcy risk (Yes/No)

---

## 🛠️ Technologies Used

### Programming & Libraries
- **Python 3.x** - Core programming language
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Data preprocessing and utilities

### Visualization
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization

### Web Application
- **Streamlit** - Interactive web application framework

### PDF Processing
- **PDFPlumber** - PDF text extraction for CNSS data

---

## 📁 Project Structure

```
Projet_ML/
│
├── ATTESTATIONS SALARIES DECLARES/   # Raw CNSS PDF declarations
├── Data CNSS/                         # Processed CNSS data
├── extracted_csvs/                    # CSV files extracted from PDFs
│
├── scripts/
│   ├── app.py                        # Main Streamlit application
│   ├── train.py                      # K-Means training script
│   ├── make_csvs.py                  # PDF to CSV extraction
│   └── extract_pdf_info.py           # PDF parsing utilities
│
├── other MLs/                        # Additional ML implementations
│   └── ML-Project-irfan/             # Irfan's bankruptcy prediction
│
├── .gitattributes                    # Git LFS configuration
└── README.md                         # This file
```

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
pip
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AbdelaliSaadali/Projet_ML.git
   cd Projet_ML
   ```

2. **Install required packages**
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn pdfplumber
   ```

### Running the Application

1. **Navigate to the scripts directory**
   ```bash
   cd scripts
   ```

2. **Launch the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser**
   The app will automatically open at `http://localhost:8501`

---

## 📖 Usage

### Company Classification

1. Navigate to "Company Classification" from the sidebar
2. Upload or select company worker data
3. View the classification results and cluster visualizations
4. Analyze company patterns and anomalies

### Salary Prediction (Morocco IT)

1. Navigate to "Salary Prediction (Marouane)"
2. Enter your professional details:
   - Profile/Role
   - Years of experience
   - Education level
   - Technology stack
   - Target company
3. Get predicted salary for 2025

### Employee Salary Prediction

1. Navigate to "Salary Prediction (Asaad)"
2. Select employee ID
3. Choose month and year
4. View predicted salary based on historical data

### Bankruptcy Prediction

1. Navigate to "Bankruptcy Prediction (Irfan)"
2. Enter company financial metrics
3. View bankruptcy risk assessment
4. Analyze risk factors

---

## 🔬 Machine Learning Algorithms

### K-Means Clustering (From Scratch)
- **Implementation:** Custom Python class
- **Features:** 
  - Random centroid initialization
  - Euclidean distance calculation
  - Iterative centroid updates
  - Convergence detection
- **Use Case:** Company classification

### Linear Regression with Gradient Descent
- **Implementation:** Custom optimization
- **Features:**
  - Cost function minimization
  - Learning rate tuning
  - Feature scaling
- **Use Cases:** Salary predictions

### K-Nearest Neighbors (From Scratch)
- **Implementation:** Distance-based classification
- **Features:**
  - Euclidean distance metric
  - Majority voting
  - Configurable K parameter
- **Use Case:** Bankruptcy prediction

### Support Vector Machine (From Scratch)
- **Implementation:** Margin-based classification
- **Features:**
  - Linear kernel
  - Optimization algorithm
  - Decision boundary computation
- **Use Case:** Bankruptcy prediction

---

## 📊 Data Sources

- **CNSS Data**: Social security declarations (PDF format)
- **IT Salary Data**: Moroccan job market data (2024-2025)
- **Employee Data**: Historical payroll records
- **Financial Data**: Company financial statements and indicators

---

## 🎯 Key Insights

### Company Classification
- Identified 4 distinct company categories based on employment patterns
- Detected seasonal employment trends
- Flagged potential fraudulent activities

### Salary Predictions
- Technology stack significantly impacts salary in Morocco
- Experience has non-linear relationship with compensation
- Company size and location are key factors

### Bankruptcy Prediction
- Financial ratios are strong predictors
- Regional economic factors play a role
- Early detection enables preventive measures

---

## 🔮 Future Enhancements

- [ ] Add deep learning models for improved predictions
- [ ] Implement real-time data updates
- [ ] Add multilingual support (Arabic, French, English)
- [ ] Create REST API for model serving
- [ ] Add model explainability features (SHAP, LIME)
- [ ] Implement automated model retraining
- [ ] Add data drift detection
- [ ] Create mobile application version

---

## 📝 License

This project is created for educational purposes as part of a Machine Learning course.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or feedback, please reach out to the project team members.

---

## 🙏 Acknowledgments

- CNSS Morocco for data structures
- Moroccan IT community for salary insights
- Open-source ML community for inspiration
- Course instructors for guidance

---

**Built with ❤️ in Morocco**