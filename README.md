# News as Sources of Realized Volatility Forecasting and Risk Propagation

This repository provides code and methodology for forecasting realized volatility (RV) at longer horizons using **HAR-type models**, **regularized regressions (Lasso/Ridge)**, and **graph-based methods** that leverage **news attributes**.  

The framework follows the methodology in *Graph-based Methods for Forecasting Realized Covariances* and extends it to incorporate news-driven features and volatility risk premium (VRP)-based strategies.  

---

## 📂 Repository Structure

### 1. `1a_HAR_News.py`
- Applies **HAR**, **Lasso**, and **Ridge** to predict realized volatility (RV).  
- Forecasting horizon: longer-term future RV.  
- Features: **HAR lags + news attributes**.  
- Methodology reference: *Graph-based Methods for Forecasting Realized Covariances*.  

---

### 2. `2a_Graph_News_LW.py`
- Applies **graph-based methods** for RV forecasting using news attributes.  
- Three types of models are implemented:

1. **GHAR + News**  
   - Graph model based on news co-coverage.  
   - Features = HAR features + graph-generated features.  

2. **GHAR + NewsPropagation**  
   - Adds **news propagation through the graph** to capture spillover effects.  

3. **GHAR + NewsPropagationAttribute**  
   - Comprehensive model combining:  
     - HAR features  
     - News attributes  
     - Graph-based features  
     - News propagation  

---

### 3. `3a_GRAPH_LOCAL.py`
- Extension of `2a_Graph_News_LW.py`.  
- Applies **different graphs to different feature groups**:  
  - HAR features  
  - News attributes  

---

### 4. `4a_TEST_LOCAL.py`
- Testing and validation module for the implemented models.  

---

### 5. `5a_strategy.py`
- Implements a **trading strategy** based on the **Volatility Risk Premium (VRP)**. 
