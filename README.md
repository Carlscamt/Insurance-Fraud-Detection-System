# Insurance Fraud Detection System

**Production-Ready ML Pipeline | 64% Fraud Detection | $1.54M Annual ROI**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-brightgreen.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)

## 🎯 Project Overview

A **production-grade machine learning system** that detects insurance fraud claims with 64.3% recall while maintaining operational efficiency and regulatory compliance. The system combines advanced ML techniques (XGBoost + SMOTE) with business-aware threshold optimization and SHAP explainability.

### Key Performance Metrics

| Metric | Value | Industry Benchmark |
|--------|-------|-------------------|
| **Fraud Detection Rate** | 64.3% | 40-50% typical |
| **Annual Savings** | $1.54M | - |
| **Precision** | 19.9% | 15-25% typical |
| **ROC-AUC** | 0.8554 | 0.80+ acceptable |
| **Model Accuracy** | 82.3% | - |
| **Human Review Cases** | 1.0% | <2% operationally feasible |

---

## 📊 Business Impact

### Financial Analysis (Annual)
```
Detected Fraud:           $1,785,000
Missed Fraud:             $  990,000
Investigation Costs:      $  240,000
Human Review Costs:       $   1,600
─────────────────────────────────────
NET ANNUAL SAVINGS:       $1,543,400
```

### Operational Metrics
- **Frauds Caught**: 119 out of 185 (64.3%)
- **False Alarms**: 480 out of 2,899 (16.6%)
- **Investigations Required**: 599 claims (19.4% of total)
- **Senior Investigator Workload**: 32 cases/year (1.0%)
- **Cost per Investigation**: $500
- **Cost per Human Review**: $50

---

## 🏗️ Technical Architecture

### Data Processing Pipeline

```
Raw Data (15,420 claims, 44 features)
    ↓
Train-Test Split (80-20, stratified)
    ↓
Feature Engineering (93 final features)
├─ Mapping categorical ranges to numeric
├─ Domain-specific risk features
├─ Cyclical encoding (temporal patterns)
└─ One-hot encoding (categorical variables)
    ↓
SMOTE Oversampling (balance class imbalance)
    ↓
XGBoost Training
    ↓
Recall-Optimized Threshold Selection
    ↓
Production Routing & SHAP Explainability
```

### Model Architecture

**Algorithm**: XGBoost Classification  
**Class Balance**: SMOTE (5.9% fraud → 50% after sampling)  
**Hyperparameters**:
- n_estimators: 300
- max_depth: 7
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8

**Threshold Strategy**: Recall-optimized with 15% precision floor
- Tests 50 thresholds (0.10-0.60)
- Filters low-precision thresholds
- Selects threshold with highest recall
- Selected threshold: **0.10** (64.3% recall, 19.9% precision)

### Feature Engineering (Top 10)

| Feature | Type | Business Logic |
|---------|------|----------------|
| `Fault_Third Party` | Binary | Liability vs. at-fault indicator |
| `BasePolicy_Liability` | Binary | Policy coverage type |
| `Month_Cos / Month_Sin` | Cyclical | Seasonal fraud patterns |
| `DayOfWeekClaimed_*` | Binary | Temporal fraud indicators |
| `High_Risk_Driver` | Engineered | Age < 30 + Low rating + Prior claims |
| `Suspicious_Circumstances` | Engineered | No witness AND no police report |
| `Quick_Claim` | Engineered | Claim filed within 4 days |
| `Old_Vehicle_High_Claims` | Engineered | Vehicle age ≥ 7 years + Prior claims |
| `Deductible_Price_Ratio` | Ratio | Financial incentive indicator |
| `Age / DriverRating` | Original | Actuarial risk factors |

---

## 🚀 Deployment Strategy

### Hybrid Automated + Human Review System

```
Claim Submitted
    ↓
Model Predicts Fraud Probability (0-1)
    ↓
Is Probability ≥ 0.10?
├─ NO → APPROVE claim
└─ YES → Check demographic group
    ├─ Young driver (16-25)?
    │  └─ HUMAN_REVIEW (1% of cases)
    ├─ Low-fraud policy type?
    │  └─ HUMAN_REVIEW (1% of cases)
    └─ Standard case
       └─ AUTO_INVESTIGATE (567 cases)
    ↓
Generate SHAP Explanation
    ↓
Investigator Reviews Case
    ↓
Final Decision (Approve/Deny/Investigate Further)
```

### Production Artifacts Generated

```
fraud_detection_model.pkl          # Trained XGBoost model
feature_columns.txt                # Feature list for inference
deployment_config.json             # Thresholds, parameters, metrics
```

---

## 📋 Installation & Usage

### Requirements

```bash
python>=3.8
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
xgboost>=1.5.0
imbalanced-learn>=0.8.0
shap>=0.41.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
```

### Quick Start

```python
import pandas as pd
import joblib
import json

# Load production artifacts
model = joblib.load('fraud_detection_model.pkl')

with open('feature_columns.txt', 'r') as f:
    feature_columns = f.read().strip().split('\n')

with open('deployment_config.json', 'r') as f:
    config = json.load(f)

# Load new claim data
claim = pd.read_csv('new_claim.csv')

# Engineer features (use provided function from code)
claim_engineered = engineer_features(claim)
claim_engineered = claim_engineered[feature_columns]

# Get prediction
fraud_probability = model.predict_proba(claim_engineered)[0, 1]

# Routing decision
threshold = config['optimal_threshold']
if fraud_probability >= threshold:
    print(f"⚠️ FLAG FOR INVESTIGATION (Fraud probability: {fraud_probability:.1%})")
else:
    print(f"✅ APPROVE (Fraud probability: {fraud_probability:.1%})")
```

### Running Full Pipeline

```bash
python fraud_detection_FINAL_PRODUCTION.py
```

Output includes:
- EDA visualizations
- Threshold analysis table
- Performance metrics
- Confusion matrix
- ROC curve and Precision-Recall curve
- Feature importance plots
- SHAP explainability plots
- Deployment summary

---

## 🔍 Model Explainability (SHAP)

Every prediction is explained with SHAP (SHapley Additive exPlanations), meeting regulatory compliance requirements.

### Top Fraud Drivers

| Rank | Feature | SHAP Impact | Interpretation |
|------|---------|-------------|-----------------|
| 1 | Fault_Third Party | 1.183 | Third-party claims highly correlated with fraud |
| 2 | BasePolicy_Liability | 0.559 | Liability policies show elevated risk |
| 3 | Month_Cos | 0.518 | Seasonal patterns in fraud submissions |
| 4 | DayOfWeekClaimed_Tuesday | 0.447 | Mid-week submissions more suspicious |
| 5 | Month_Sin | 0.429 | Temporal seasonality (cyclical) |

### Case Explanation Example

For any flagged claim, the system generates individual SHAP force plots showing:
- Which features pushed prediction toward fraud
- Which features pushed toward legitimacy
- Magnitude of each factor's impact
- Cumulative decision path

---

## ⚖️ Fairness & Bias Mitigation

### Disparate Impact Prevention

The system implements a **two-tier deployment strategy** to ensure fair treatment of protected demographics:

1. **Actuarial Justification**: Model uses legitimate risk factors (e.g., young drivers have 2.5x higher claim rates per insurance industry data)

2. **Human Oversight for Sensitive Groups**: 
   - Ages 16-25 → Senior investigator review
   - Low-data policy types → Manual verification
   - Result: Individual fairness through case-by-case review

3. **Operational Cost**: $1,600/year for human reviews = acceptable trade-off for legal defensibility

4. **Audit Trail**: SHAP explanations document why each case was flagged

---

## 📈 Performance Comparison

### Different Optimization Strategies

| Strategy | Recall | Precision | Savings | Use Case |
|----------|--------|-----------|---------|----------|
| **Recall-Optimized** | 64.3% | 19.9% | $1.54M | Max fraud detection ✅ |
| F1-Balanced | 45.4% | 23.0% | $1.12M | Limited resources |
| Fairness-Constrained | 33.0% | 17.0% | $0.77M | Strict demographic parity |

### Why Recall-Optimized?

- **Higher fraud detection**: Catches nearly 2 in 3 frauds
- **Better ROI**: $1.54M savings vs. alternatives
- **Operationally feasible**: 19.4% investigation rate manageable
- **Minimal human burden**: Only 1% require senior review
- **Acceptable precision**: 1 in 5 flags is real fraud

---

## 🎓 Key Technical Decisions

### 1. Data Leakage Prevention
**Problem**: Feature engineering applied to entire dataset before train-test split introduces data leakage
**Solution**: Split raw data first, then engineer features separately per set
**Impact**: Prevents inflated performance metrics; ensures honest evaluation

### 2. SMOTE Only on Training Data
**Problem**: Applying SMOTE to combined data leaks test information
**Solution**: SMOTE applied only to training set; test set remains pristine
**Impact**: Realistic performance estimates on unseen data

### 3. No Redundant Class Weighting
**Problem**: Using both SMOTE and scale_pos_weight creates conflicting signals
**Solution**: Remove scale_pos_weight when using SMOTE
**Impact**: Cleaner optimization without competing objectives

### 4. Recall-Optimized Threshold
**Problem**: F1-optimization balances precision/recall but misses fraud
**Solution**: Optimize for recall with 15% precision floor
**Impact**: 64.3% fraud detection vs. 45% with F1-balanced approach

---

## 📚 Code Structure

```
fraud_detection_FINAL_PRODUCTION.py
├─ 1. Data Loading & EDA
├─ 2. Train-Test Split (prevent leakage)
├─ 3. Feature Engineering Function
├─ 4. Apply Engineering (separately per set)
├─ 5. SMOTE Oversampling (training only)
├─ 6. XGBoost Model Training
├─ 7. Recall-Optimized Threshold Selection
├─ 8. Production Routing Logic
├─ 9. Performance Evaluation
├─ 10. Business Impact Analysis
├─ 11. Visualizations (4 plots)
├─ 12. SHAP Explainability
├─ 13. Save Production Artifacts
├─ 14. Inference Pipeline
└─ 15. Deployment Summary
```

---

## 🔮 Future Improvements

### Short Term (Production)
- [ ] A/B test threshold values in production
- [ ] Monitor feature drift over time
- [ ] Retrain quarterly with new data
- [ ] Implement automated model monitoring

### Medium Term (3-6 months)
- [ ] Add ensemble methods (XGBoost + LightGBM + CatBoost)
- [ ] Implement model calibration for probability estimates
- [ ] Develop claim-specific routing rules
- [ ] Create automated retraining pipeline

### Long Term (6-12 months)
- [ ] Develop fraud ring detection (multivariate analysis)
- [ ] Implement sequential pattern analysis
- [ ] Add external data sources (weather, traffic, news)
- [ ] Develop real-time streaming predictions

---

## 📊 Dataset Information

**Source**: [Vehicle Claim Fraud Detection - Kaggle](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)

**Characteristics**:
- 15,420 insurance claims
- 44 original features
- 5.99% fraud rate (928 frauds total)
- 20 categorical, 24 numerical features
- 320 missing values (handled via mapping)

**Class Imbalance Challenge**:
- Problem: Only 5.99% fraud vs. 94.01% legitimate
- Solution: SMOTE oversampling (11,598 synthetic frauds for training)
- Result: Balanced training set for effective learning

---

## 💼 For Hiring Managers

This project demonstrates:

✅ **Machine Learning Fundamentals**
- Advanced class imbalance handling (SMOTE, threshold optimization)
- Proper train-test procedures (prevent data leakage)
- Performance metrics beyond accuracy (precision, recall, F1, ROC-AUC)

✅ **Business Acumen**
- ROI calculation and financial impact analysis
- Operational feasibility (investigation workload)
- Trade-off analysis (recall vs. precision vs. cost)

✅ **Production Readiness**
- Explainability (SHAP for regulatory compliance)
- Fairness & bias mitigation (protected demographics)
- Deployment strategy (hybrid automated + manual review)
- Artifact generation (model, config, inference pipeline)

✅ **Professional Coding**
- Clean, documented code with clear structure
- Comprehensive error handling
- Modular functions for reusability
- Production-grade comments

---

## 📞 Contact & Links

- **LinkedIn**: [www.linkedin.com/in/carlscamt]

---

**Last Updated**: December 2025  
**Model Version**: 2.0 (Recall-Optimized)  
**Status**: ✅ Production Ready

