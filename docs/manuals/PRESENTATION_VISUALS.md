# Presentation Visuals: Real Estate Estimator

**Purpose:** These diagrams support the "Whiteboard Script" for the technical presentation. You can draw these on a whiteboard or present this document directly.

---

## 1. The "Before & After" (The Hook)

### Legacy State (The Problem)
A fragile, manual process with broken links and outdated data.

```mermaid
graph LR
    A[Legacy Script] -->|Broken Paths| B[Missing Data]
    B -->|Leakage| C[Weak KNN Model]
    C -->|Manual Run| D[Local Output]
    style A fill:#ffcccc,stroke:#333,stroke-width:2px
    style B fill:#ffcccc,stroke:#333,stroke-width:2px
    style C fill:#ffcccc,stroke:#333,stroke-width:2px
```

### Current State (The Solution)
A robust, automated enterprise pipeline.

```mermaid
graph LR
    A[Fresh Data 2024] -->|ETL| B[CI/CD Pipeline]
    B -->|Auto-Train| C[XGBoost Model]
    C -->|Deploy| D[FastAPI Service]
    style A fill:#ccffcc,stroke:#333,stroke-width:2px
    style B fill:#ccffcc,stroke:#333,stroke-width:2px
    style C fill:#ccffcc,stroke:#333,stroke-width:2px
    style D fill:#ccffcc,stroke:#333,stroke-width:2px
```

---

## 2. High-Level Architecture (The "What")

This diagram shows how the system handles a user request in production.

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI (src/main.py)
    participant Feat as FeatureService
    participant Model as ModelService
    participant MLflow as Model Registry

    Note over API, MLflow: Startup Phase
    API->>MLflow: Load Production Model
    API->>Feat: Load Demographics Cache

    Note over User, API: Request Phase
    User->>API: POST /predict (17 features)
    API->>API: Validate (Pydantic)
    API->>Feat: Enrich (Add Zipcode Stats)
    Feat-->>API: Return 43 Features
    API->>Model: Predict(Features)
    Model-->>API: Return Price ($850k)
    API-->>User: JSON Response
```

**ASCII Version:**
```text
[ User ]
   |
   v
[ FastAPI Application ]
   |
   +--- 1. Validate Input (Pydantic)
   |
   +--- 2. Enrich Features (FeatureService) <--- [ Demographics Cache ]
   |       (Adds Income, Education, etc.)
   |
   +--- 3. Predict Price (ModelService) <------- [ XGBoost Model ]
   |
   v
[ JSON Response ]
```

---

## 3. The Training Pipeline (The "How")

This diagram shows how raw data becomes a deployable model.

```mermaid
flowchart TD
    Raw[Raw Assessment CSVs] -->|transform_assessment_data.py| Clean[Cleaned DataFrame]
    Clean -->|train_with_mlflow.py| Split{GroupKFold Split}
    
    subgraph Training_Loop
    Split -->|Train Set| Pipe[Sklearn Pipeline]
    Pipe -->|1. Impute| Imputer
    Imputer -->|2. Scale| Scaler
    Scaler -->|3. Fit| XGB[XGBoost Regressor]
    end
    
    subgraph Evaluation
    Split -->|Test Set| Eval[Evaluate Fresh]
    Eval -->|Metrics| MAE[MAE / R2]
    Eval -->|Plots| Res[Residual Plots]
    end
    
    XGB -->|Log Artifacts| MLflow[(MLflow Tracking)]
    MAE -->|Log Metrics| MLflow
```

---

## 4. Honest Evaluation (The "Why")

Visualizing why `GroupKFold` is necessary to prevent data leakage from repeat sales.

### The Wrong Way (Random Split)
*The model "memorizes" the house.*

```text
Parcel A (Sold 2021)  ----> [ TRAIN SET ]
Parcel A (Sold 2023)  ----> [ TEST SET  ]  <-- LEAKAGE!
                                              Model knows this house!
```

### The Right Way (GroupKFold)
*The model must generalize to unseen houses.*

```text
Parcel A (Sold 2021)  ----> [ TEST SET ]
Parcel A (Sold 2023)  ----> [ TEST SET ]

Parcel B (Sold 2022)  ----> [ TRAIN SET ]
Parcel C (Sold 2024)  ----> [ TRAIN SET ]

Result: Model has NEVER seen Parcel A. 
        It must predict based on features, not memory.

```
