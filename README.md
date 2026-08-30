# ✈️ Predictive Maintenance System using CNN-LSTM

A Machine Learning-based Predictive Maintenance System that predicts the **Remaining Useful Life (RUL)** of aircraft engines using historical sensor data from the **N-CMAPSS dataset**.

The project uses a **CNN-LSTM deep learning architecture** to learn both local patterns and temporal dependencies from multivariate sensor data. A Flask web application provides an easy-to-use interface for selecting an aircraft/unit and obtaining its predicted RUL and maintenance status.

---

## 📌 Project Overview

Aircraft engines are monitored using multiple sensors that continuously record information about their operating condition.

As an engine operates, its condition gradually changes. Predictive maintenance aims to estimate **how many operating cycles remain before maintenance is required**.

This project uses historical sensor observations to predict:

> **Remaining Useful Life (RUL) in cycles**

The system then converts the predicted RUL into a maintenance recommendation.

### Example

For Aircraft/Unit **66**:

```text
Aircraft / Unit ID : 66
Observations       : 1149
Predicted RUL      : 7.16 cycles
Maintenance Status : HIGH PRIORITY - Maintenance Recommended