# Data Provenance & Context Log

**Purpose:** Document data source, limitations, and geographic/market considerations  
**Phase:** Project Kickoff & Planning  
**Status:** In Progress

---

## Entry 1: Data Source Investigation

**Timestamp:** 2025-12-06 15:15 UTC  
**Task:** Confirm data origin - KC likely refers to King County, Seattle WA (not Kansas City)  
**Status:** Research in Progress

### Key Findings

**File Naming Convention:** "kc_house_data.csv"
- **Most Likely Interpretation:** KC = King County, Seattle, Washington
- **Why:** Column data shows Seattle area zipcodes (98118, 98115, 98040, 98042, etc.)
- **Confirmation:** Latitude/longitude data consistent with Seattle region

**Geographic Scope:**
- **Zipcodes Present:** 83 unique zipcodes in greater Seattle metro area
- **Market:** Pacific Northwest, specifically Puget Sound region
- **Data Period:** House sales from 2014-2015 era (based on date format in README)
- **Region Type:** Mixed urban, suburban, semi-rural zipcodes

### Market Characteristics

**Seattle Real Estate Context:**
- Tech-driven market (Amazon HQ, Microsoft nearby)
- Rapidly appreciating values during 2014-2015 period
- Urban center (Seattle proper) + suburban sprawl (Bellevue, Tacoma, Olympia areas)
- Waterfront premium areas (Puget Sound)
- Mountain/forest areas with lower values

**Data Representativeness:**
- **STRENGTH:** Local market data (most relevant for Seattle predictions)
- **STRENGTH:** 21,613 examples = statistically significant sample
- **LIMITATION:** 2014-2015 vintage (prices have appreciated significantly since)
- **LIMITATION:** Time-bounded (single market cycle)
- **LIMITATION:** Regional specificity (not generalizable to other markets)

### Implications for Model

**For Version 1 (Current):**
- Use as-is - phData focus is deployment, not research
- Document the temporal limitation
- Note: Predictions will be lower than 2025 market values (11 years of appreciation)
- Works well for relative comparisons, absolute values dated

**For Version 2 (Future Enhancement):**
- Consider retraining with more recent data
- Add time-series features if updating model
- Document model vintage in all predictions
- Note applicability window

### Hyper-Local Real Estate Notes

**Why Geographic Specificity Matters:**
- Same square footage in different zipcodes = 2-10x price difference
- Demographic data (income, education) is proxy for neighborhood desirability
- School districts, crime, walkability captured partially via demographics
- Waterfront vs inland = major premium in Seattle market

**Data Columns That Capture This:**
- `zipcode` (primary driver, joined with demographics)
- `lat/long` (secondary, location within zipcode)
- `waterfront` flag (premium feature)
- `condition`, `grade`, `view` (quality indicators)
- Demographic income/education (neighborhood proxy)

### Data Quality Assessment

**Observations from Sample Inspection:**
- No null values visible in sample rows
- Zipcodes consistent format (5-digit strings)
- Price values reasonable for 2014-2015 Seattle market ($221K to $538K in samples)
- Feature values all present (no missing columns)

**Potential Issues:**
- None identified in sample inspection
- Data appears clean and well-curated for ML
- Kaggle source typically means peer-reviewed quality

---

## Entry 2: DEMOGRAPHICS_PATH Bug - Fix Strategy

**Timestamp:** 2025-12-06 15:20 UTC  
**Decision:** Fix bug AFTER GitHub setup in dev branch  
**Rationale:** Professional DevOps practice

**Current Status:** Line 14 in create_model.py
```python
DEMOGRAPHICS_PATH = "data/kc_house_data.csv"  # WRONG
```

**Will Fix To:**
```python
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # CORRECT
```

**When:** In dev branch, first commit will include this fix with explanation in commit message  
**Why:** Shows professional Git workflow - bug tracking and fix in VCS history

---

## Entry 3: Feature Utilization Analysis

**Timestamp:** 2025-12-06 15:20 UTC  
**Task:** Document unused features for v2 roadmap  
**Status:** Completed

### Current Model (v1)
**Features Used: 35 total**
- 8 Home Features: bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement, zipcode
- 27 Demographic Features: Population, income, education (from zipcode_demographics.csv)

**Features NOT Used: 13 columns**
- `id` - Identifier only, no predictive value
- `date` - Temporal feature, could add time series value
- `price` - Target variable (excluded correctly)
- `waterfront` - Binary flag for waterfront property (0-1)
- `view` - View quality rating
- `condition` - Property condition rating
- `grade` - Construction grade rating
- `yr_built` - Year constructed (age proxy)
- `yr_renovated` - Renovation year
- `lat` / `long` - Precise coordinates (more granular than zipcode)
- `sqft_living15` - Neighbor average living sqft
- `sqft_lot15` - Neighbor average lot sqft

### Version 2 Roadmap (Future)

**Recommended Feature Additions (Priority Order):**
1. **`condition`** (high impact) - Condition ratings typically predict price
2. **`grade`** (high impact) - Construction grade strong predictor
3. **`yr_built`** (medium impact) - Age of property affects value
4. **`waterfront`** (high impact) - Massive premium in Seattle
5. **`view`** (medium impact) - View quality adds value

**Why Phase v1 Excludes These:**
- phData requirement: "Build simplest solution first"
- These would require additional feature engineering
- Current model works with basic features
- Can validate deployment first, enhance model later

---

## Entry 4: Temporal Limitation - Critical v2 Priority

**Timestamp:** 2025-12-06 16:00 UTC  
**Task:** Document temporal limitation and plan for v2 mitigation  
**Status:** Completed  
**Importance:** CRITICAL

### The Issue

**Data Vintage:** 2014-2015  
**Current Date:** December 2025  
**Age:** ~11 years old  
**Impact:** HIGH - Real estate prices have appreciated significantly

### Why This Matters

**Price Appreciation in Seattle (2014-2025):**
- Average home appreciation: 120-150% (conservative estimate)
- Some neighborhoods: 150-200%+ (high-demand tech areas)
- Model trained on 2014-2015 prices will predict LOW for 2025 homes

**Example:**
- 2014: Home valued at $300k
- 2025: Same home worth ~$600-750k
- Model trained on 2014 data will predict ~$300k
- Error: -50% to -60%

### Implications for v1

**v1 (Current) Strategy:**
- ✅ Use as-is for relative comparisons
- ✅ Suitable for "is this property over/undervalued relative to similar homes?"
- ✅ NOT suitable for absolute price prediction
- ⚠️ Document limitation in all presentations and predictions
- ⚠️ Add timestamp field to all API responses showing data vintage

### v2 Enhancement Plan - PRIORITY 1

**Solution: Retrain with Fresh Data**

1. **Option A: Retrain with 2024 Data (Recommended)**
   - New source: Recent King County MLS data
   - Time required: Minimal (same process as v1)
   - Expected improvement: R² increase 0.20-0.30+
   - Accuracy: Current market prices
   
2. **Option B: Time-Series Adjustment**
   - Keep 2014-2015 model
   - Apply appreciation multiplier (year-dependent)
   - Formula: Predicted_2025 = Predicted_2014 * appreciation_factor
   - Less ideal but faster than retraining

3. **Option C: Hybrid Approach**
   - Retrain with new data
   - Include "price_inflation_index" as feature
   - Allow model to learn time-dependent adjustments

**Recommendation:** Option A (full retrain with fresh data)
- Highest accuracy
- Most professional (not a quick fix)
- Enables proper evaluation of model improvements
- Data should be available for Seattle area

### Implementation Timing

**v1 (Now):**
- [ ] Use as-is with limitation clearly documented
- [ ] Include data_vintage in all responses: "Trained on: 2014-2015 data"
- [ ] Add disclaimer: "Predictions reflect 2014-2015 market conditions"

**v2 (Next iteration):**
- [ ] Research fresh Seattle real estate data source
- [ ] Retrain model with 2024 data
- [ ] Compare v1 vs v2 performance
- [ ] Update model registry with v2
- [ ] Promote v2 to Production after validation

**v3 (Future):**
- [ ] Annual retraining cycle
- [ ] Automated retraining triggers
- [ ] A/B testing v2 vs v3

### Data Sources for v2

**Possible sources:**
- Zillow API (requires registration, free tier limited)
- Redfin data (public records available)
- King County Assessor's office
- Kaggle (sometimes has recent datasets)
- MLS data through real estate API

**Estimated effort:** 2-4 hours to source and prepare data

### Documentation Strategy

**For v1 Presentations:**
```
"This model was trained on King County housing data from 2014-2015.
It performs well for relative comparisons but should not be used for
absolute price estimates in 2025, as real estate values have 
appreciated approximately 120-150% since training data collection.

For v2, we will retrain with current market data for production accuracy."
```

**For API Responses:**
```json
{
  "prediction": 425000,
  "model_version": "v1",
  "data_vintage": "2014-2015",
  "note": "Prediction reflects 2014-2015 market values. Actual 2025 value likely 120-150% higher",
  "timestamp": "2025-12-06T15:00:00Z"
}
```

---

## Summary

**Data Assessment:** KC = King County, Seattle. Data is local, relevant, clean.  
**Temporal Note:** 2014-2015 vintage - ~11 years old. CRITICAL LIMITATION for v1.  
**v1 Strategy:** Use with clear documentation of limitation. Good for relative comparisons.  
**v2 Priority:** Retrain with fresh Seattle real estate data for production accuracy.  
**Bug Fix:** Will do in dev branch on GitHub.  
**Features v2:** Plan to add condition, grade, waterfront, yr_built PLUS fresh training data.

---



**Purpose:** Document data source, limitations, and geographic/market considerations  
**Phase:** Project Kickoff & Planning  
**Status:** In Progress

---

## Entry 1: Data Source Investigation

**Timestamp:** 2025-12-06 15:15 UTC  
**Task:** Confirm data origin - KC likely refers to King County, Seattle WA (not Kansas City)  
**Status:** Research in Progress

### Key Findings

**File Naming Convention:** "kc_house_data.csv"
- **Most Likely Interpretation:** KC = King County, Seattle, Washington
- **Why:** Column data shows Seattle area zipcodes (98118, 98115, 98040, 98042, etc.)
- **Confirmation:** Latitude/longitude data consistent with Seattle region

**Geographic Scope:**
- **Zipcodes Present:** 83 unique zipcodes in greater Seattle metro area
- **Market:** Pacific Northwest, specifically Puget Sound region
- **Data Period:** House sales from 2014-2015 era (based on date format in README)
- **Region Type:** Mixed urban, suburban, semi-rural zipcodes

### Market Characteristics

**Seattle Real Estate Context:**
- Tech-driven market (Amazon HQ, Microsoft nearby)
- Rapidly appreciating values during 2014-2015 period
- Urban center (Seattle proper) + suburban sprawl (Bellevue, Tacoma, Olympia areas)
- Waterfront premium areas (Puget Sound)
- Mountain/forest areas with lower values

**Data Representativeness:**
- **STRENGTH:** Local market data (most relevant for Seattle predictions)
- **STRENGTH:** 21,613 examples = statistically significant sample
- **LIMITATION:** 2014-2015 vintage (prices have appreciated significantly since)
- **LIMITATION:** Time-bounded (single market cycle)
- **LIMITATION:** Regional specificity (not generalizable to other markets)

### Implications for Model

**For Version 1 (Current):**
- Use as-is - phData focus is deployment, not research
- Document the temporal limitation
- Note: Predictions will be lower than 2025 market values (11 years of appreciation)
- Works well for relative comparisons, absolute values dated

**For Version 2 (Future Enhancement):**
- Consider retraining with more recent data
- Add time-series features if updating model
- Document model vintage in all predictions
- Note applicability window

### Hyper-Local Real Estate Notes

**Why Geographic Specificity Matters:**
- Same square footage in different zipcodes = 2-10x price difference
- Demographic data (income, education) is proxy for neighborhood desirability
- School districts, crime, walkability captured partially via demographics
- Waterfront vs inland = major premium in Seattle market

**Data Columns That Capture This:**
- `zipcode` (primary driver, joined with demographics)
- `lat/long` (secondary, location within zipcode)
- `waterfront` flag (premium feature)
- `condition`, `grade`, `view` (quality indicators)
- Demographic income/education (neighborhood proxy)

### Data Quality Assessment

**Observations from Sample Inspection:**
- No null values visible in sample rows
- Zipcodes consistent format (5-digit strings)
- Price values reasonable for 2014-2015 Seattle market ($221K to $538K in samples)
- Feature values all present (no missing columns)

**Potential Issues:**
- None identified in sample inspection
- Data appears clean and well-curated for ML
- Kaggle source typically means peer-reviewed quality

---

## Entry 2: DEMOGRAPHICS_PATH Bug - Fix Strategy

**Timestamp:** 2025-12-06 15:20 UTC  
**Decision:** Fix bug AFTER GitHub setup in dev branch  
**Rationale:** Professional DevOps practice

**Current Status:** Line 14 in create_model.py
```python
DEMOGRAPHICS_PATH = "data/kc_house_data.csv"  # WRONG
```

**Will Fix To:**
```python
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # CORRECT
```

**When:** In dev branch, first commit will include this fix with explanation in commit message  
**Why:** Shows professional Git workflow - bug tracking and fix in VCS history

---

## Entry 3: Feature Utilization Analysis

**Timestamp:** 2025-12-06 15:20 UTC  
**Task:** Document unused features for v2 roadmap  
**Status:** Completed

### Current Model (v1)
**Features Used: 35 total**
- 8 Home Features: bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement, zipcode
- 27 Demographic Features: Population, income, education (from zipcode_demographics.csv)

**Features NOT Used: 13 columns**
- `id` - Identifier only, no predictive value
- `date` - Temporal feature, could add time series value
- `price` - Target variable (excluded correctly)
- `waterfront` - Binary flag for waterfront property (0-1)
- `view` - View quality rating
- `condition` - Property condition rating
- `grade` - Construction grade rating
- `yr_built` - Year constructed (age proxy)
- `yr_renovated` - Renovation year
- `lat` / `long` - Precise coordinates (more granular than zipcode)
- `sqft_living15` - Neighbor average living sqft
- `sqft_lot15` - Neighbor average lot sqft

### Version 2 Roadmap (Future)

**Recommended Feature Additions (Priority Order):**
1. **`condition`** (high impact) - Condition ratings typically predict price
2. **`grade`** (high impact) - Construction grade strong predictor
3. **`yr_built`** (medium impact) - Age of property affects value
4. **`waterfront`** (high impact) - Massive premium in Seattle
5. **`view`** (medium impact) - View quality adds value

**Why Phase v1 Excludes These:**
- phData requirement: "Build simplest solution first"
- These would require additional feature engineering
- Current model works with basic features
- Can validate deployment first, enhance model later

---

## Entry 4: Temporal Limitation - Critical v2 Priority

**Timestamp:** 2025-12-06 16:00 UTC  
**Task:** Document temporal limitation and plan for v2 mitigation  
**Status:** Completed  
**Importance:** CRITICAL

### The Issue

**Data Vintage:** 2014-2015  
**Current Date:** December 2025  
**Age:** ~11 years old  
**Impact:** HIGH - Real estate prices have appreciated significantly

### Why This Matters

**Price Appreciation in Seattle (2014-2025):**
- Average home appreciation: 120-150% (conservative estimate)
- Some neighborhoods: 150-200%+ (high-demand tech areas)
- Model trained on 2014-2015 prices will predict LOW for 2025 homes

**Example:**
- 2014: Home valued at $300k
- 2025: Same home worth ~$600-750k
- Model trained on 2014 data will predict ~$300k
- Error: -50% to -60%

### Implications for v1

**v1 (Current) Strategy:**
- ✅ Use as-is for relative comparisons
- ✅ Suitable for "is this property over/undervalued relative to similar homes?"
- ✅ NOT suitable for absolute price prediction
- ⚠️ Document limitation in all presentations and predictions
- ⚠️ Add timestamp field to all API responses showing data vintage

### v2 Enhancement Plan - PRIORITY 1

**Solution: Retrain with Fresh Data**

1. **Option A: Retrain with 2024 Data (Recommended)**
   - New source: Recent King County MLS data
   - Time required: Minimal (same process as v1)
   - Expected improvement: R² increase 0.20-0.30+
   - Accuracy: Current market prices
   
2. **Option B: Time-Series Adjustment**
   - Keep 2014-2015 model
   - Apply appreciation multiplier (year-dependent)
   - Formula: Predicted_2025 = Predicted_2014 * appreciation_factor
   - Less ideal but faster than retraining

3. **Option C: Hybrid Approach**
   - Retrain with new data
   - Include "price_inflation_index" as feature
   - Allow model to learn time-dependent adjustments

**Recommendation:** Option A (full retrain with fresh data)
- Highest accuracy
- Most professional (not a quick fix)
- Enables proper evaluation of model improvements
- Data should be available for Seattle area

### Implementation Timing

**v1 (Now):**
- [ ] Use as-is with limitation clearly documented
- [ ] Include data_vintage in all responses: "Trained on: 2014-2015 data"
- [ ] Add disclaimer: "Predictions reflect 2014-2015 market conditions"

**v2 (Next iteration):**
- [ ] Research fresh Seattle real estate data source
- [ ] Retrain model with 2024 data
- [ ] Compare v1 vs v2 performance
- [ ] Update model registry with v2
- [ ] Promote v2 to Production after validation

**v3 (Future):**
- [ ] Annual retraining cycle
- [ ] Automated retraining triggers
- [ ] A/B testing v2 vs v3

### Data Sources for v2

**Possible sources:**
- Zillow API (requires registration, free tier limited)
- Redfin data (public records available)
- King County Assessor's office
- Kaggle (sometimes has recent datasets)
- MLS data through real estate API

**Estimated effort:** 2-4 hours to source and prepare data

### Documentation Strategy

**For v1 Presentations:**
```
"This model was trained on King County housing data from 2014-2015.
It performs well for relative comparisons but should not be used for
absolute price estimates in 2025, as real estate values have 
appreciated approximately 120-150% since training data collection.

For v2, we will retrain with current market data for production accuracy."
```

**For API Responses:**
```json
{
  "prediction": 425000,
  "model_version": "v1",
  "data_vintage": "2014-2015",
  "note": "Prediction reflects 2014-2015 market values. Actual 2025 value likely 120-150% higher",
  "timestamp": "2025-12-06T15:00:00Z"
}
```

---

## Summary

**Data Assessment:** KC = King County, Seattle. Data is local, relevant, clean.  
**Temporal Note:** 2014-2015 vintage - ~11 years old. CRITICAL LIMITATION for v1.  
**v1 Strategy:** Use with clear documentation of limitation. Good for relative comparisons.  
**v2 Priority:** Retrain with fresh Seattle real estate data for production accuracy.  
**Bug Fix:** Will do in dev branch on GitHub.  
**Features v2:** Plan to add condition, grade, waterfront, yr_built PLUS fresh training data.

---


