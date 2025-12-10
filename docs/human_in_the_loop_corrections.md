# Human-in-the-Loop Corrections Log

**Project:** Real-Estate-Estimator (phData MLE Coding Test)
**Purpose:** Document instances where human oversight corrected AI direction, provided guidance, or served as a safety net
**Value:** Demonstrates responsible AI collaboration and the importance of human judgment

---

## Correction #1: Emoji Usage in Code

**Date:** 2025-12-07
**Phase:** Phase B - Training Scripts

**What AI Did Wrong:**
- Used emoji characters in src/train.py output messages (lines 412, 419, 424)
- Emojis used: checkmark, X mark for success/error indicators

**Human Intervention:**
- User flagged: "did you just put emoji's in that code? that goes against our protocols"
- Directed AI to re-read master protocol

**Root Cause:**
- AI failed to internalize Generation Sub-Protocol Section 3.3
- Rule explicitly states: "NO special characters or emojis in code/documentation UNLESS explicitly authorized"

**Resolution:**
- Replaced emojis with text markers: [SUCCESS], [ERROR], [WARNING]
- AI re-read master protocol in full

**Lesson Learned:**
- Reading a protocol is not the same as following it
- Must actively check generated output against protocol rules

---

## Correction #2: Missing Logging Discipline

**Date:** 2025-12-07
**Phase:** Phase B - Training Scripts

**What AI Did Wrong:**
- Created src/train.py and src/evaluate.py without updating logs
- Logging Sub-Protocol Section 3.4 requires: "Update after each significant action"

**Human Intervention:**
- User asked: "are you documenting as you're going per the logging protocol?"
- Forced AI to acknowledge the gap

**Root Cause:**
- AI was focused on code generation, not process discipline
- Did not treat logging as a mandatory step

**Resolution:**
- Created logs/2025-12-07_implementation_log.md
- Updated logs/master_log.md
- Established habit of logging after each action

**Lesson Learned:**
- Protocols must be followed as habits, not afterthoughts
- Logging is part of the deliverable, not optional documentation

---

## Correction #3: Missing copilot-instructions.md

**Date:** 2025-12-07
**Phase:** Phase B - Training Scripts

**What AI Did Wrong:**
- Never created .github/copilot-instructions.md despite it being the intended entry point
- Template existed in Reference_Docs but was never instantiated

**Human Intervention:**
- User asked: "we established a copilot-instructions.md, correct?"
- Forced AI to verify existence (it didn't exist)

**Root Cause:**
- AI received template in early sessions but never created the actual file
- Gap between "planning" and "execution"

**Resolution:**
- Created .github/copilot-instructions.md with full protocol enforcement
- Now serves as mandatory checklist for every AI response

**Lesson Learned:**
- Planning documents are not the same as executable artifacts
- Must verify that foundational infrastructure actually exists

---

## Correction #4: Skipping Model Training

**Date:** 2025-12-07
**Phase:** Phase B to C Transition

**What AI Did Wrong:**
- Was about to proceed to FastAPI implementation (Phase C)
- Had not actually run training to create model.pkl
- Had not verified data/ directory existed
- Sequence error: building API to serve a model that doesn't exist

**Human Intervention:**
- User asked: "so it looks like the models haven't been trained yet... Is that why it feels like we're skipping the actual training?"
- Requested detailed explanation of what was actually happening

**Root Cause:**
- AI conflated "writing training script" with "training the model"
- Focused on code artifacts, not operational state
- Did not verify prerequisites before moving to next phase

**Resolution:**
- Explained ML lifecycle (training vs serving) in detail
- Verified data/ directory doesn't exist yet
- Corrected sequence: must set up data, run training, THEN build API

**Lesson Learned:**
- Writing code is not the same as running code
- Must verify operational prerequisites, not just code prerequisites
- Human intuition caught a fundamental sequencing error

---

## Correction #5: LightGBM Tuning Taking Too Long

**Date:** 2025-12-08
**Phase:** V2.4 - Model Alternatives

**What AI Did Wrong:**
- Created `tune_top_models.py` with wide parameter ranges for LightGBM
- `num_leaves`: 15-100 combined with `max_depth`: 3-15
- Some random combinations created extremely slow model configurations
- User reported 30+ minutes of execution with no progress

**Human Intervention:**
- User flagged: "it has been running for over 30 minutes... LightGBM is taking a long time specifically"
- User had to Ctrl+C to cancel the stuck process

**Root Cause:**
- AI didn't constrain parameter search space appropriately
- Large `num_leaves` × large `max_depth` × high `n_estimators` = slow models
- RandomizedSearchCV doesn't discriminate between fast and slow configurations

**Resolution:**
- Created focused `tune_xgboost.py` with constrained parameters
- Dropped multi-model tuning in favor of winner-only tuning
- Reduced `max_depth` range, limited `num_leaves`

**Lesson Learned:**
- Parameter spaces must be constrained for practical runtimes
- "Light" optimization should actually be light
- Test parameter combinations before including in search space

---

## Correction #6: Model Not Actually Deployed

**Date:** 2025-12-08
**Phase:** V2.4 - Model Alternatives

**What AI Did Wrong:**
- Issued `Copy-Item` command to copy XGBoost model to production
- Shell reported exit code 0 (success)
- But model.pkl still contained old KNN model

**Human Intervention:**
- User ran `evaluate.py` and noticed metrics matched V2.3 KNN ($84,494 MAE)
- User asked: "Is this evaluating the newest model?"
- Forced investigation of what was actually deployed

**Root Cause:**
- Shell terminal in broken state (pid: -1) after window restart
- Commands ran but didn't execute properly
- Exit code 0 was misleading - no actual file operation occurred

**Resolution:**
- Created `check_model.py` diagnostic script
- User ran copy command manually in working terminal
- Verified deployment with `check_model.py`

**Lesson Learned:**
- Always verify file operations, don't trust exit codes alone
- After deployments, check the deployed artifact directly
- Terminal issues can cause silent failures

---

## Correction #7: Model Service Reporting Wrong Version

**Date:** 2025-12-08
**Phase:** V2.4 - Model Alternatives

**What AI Did Wrong:**
- API health endpoint showed `model_version: v2.1` even after XGBoost deployment
- Model service determined version by feature count, not actual model type
- Any 43-feature model would always report "v2.1"

**Human Intervention:**
- User noticed API test results showed v2.1 instead of v2.4.1
- Question prompted investigation of version detection logic

**Root Cause:**
- `model_service.py` lines 104-110 used feature count heuristic:
  ```python
  if len(self.feature_names) >= 40:
      self.model_version = "v2.1"
  ```
- This was adequate for V1 vs V2.1 but broke with V2.3+

**Resolution:**
- Updated `model_service.py` to read version from `metrics.json`
- Added `model_type` attribute for better tracking
- Version now accurately reflects deployed model

**Lesson Learned:**
- Heuristics that work initially may break with evolution
- Version tracking should be explicit, not inferred
- Test API reporting, not just API functionality

---

## Correction #8: NumPy Bool JSON Serialization

**Date:** 2025-12-08
**Phase:** V2.5 - Robust Evaluation

**What AI Did Wrong:**
- Created `robust_evaluate.py` with JSON serialization for results
- The `convert()` function handled numpy int/float but not numpy bool
- Script crashed when saving results: `TypeError: Object of type bool_ is not JSON serializable`

**Human Intervention:**
- User ran script and reported the traceback
- Error was clear: `when serializing dict item 'is_normal'`

**Root Cause:**
- NumPy boolean (`np.bool_`) is different from Python `bool`
- Residual analysis used `scipy.stats.normaltest()` which returns numpy types
- The type conversion was incomplete

**Resolution:**
- Added numpy bool handling to the convert function:
  ```python
  elif isinstance(obj, (np.bool_, bool)):
      return bool(obj)
  ```

**Lesson Learned:**
- NumPy has many types beyond int/float that need conversion
- Test JSON serialization with actual data before assuming it works
- When converting types for JSON, enumerate all possible numpy types

---

## Correction #9: Optimal Tier Split - Percentile vs Dollar Amount

**Date:** 2025-12-08
**Phase:** V2.7 - Tiered Models Exploration

**What AI Did Wrong:**
- Initially designed tier split using median (50th percentile) arbitrarily
- Also considered fixed dollar amounts ($400k, $750k) for tier boundaries
- Both approaches have limitations with price appreciation over time

**Human Intervention:**
- User insight: "Let's experiment with, explore and shape the tier split to find the optimum percentile instead of the middle arbitrarily or a dollar amount, as that will change with price appreciation over time"
- User recognized that optimal split might not be at 50%
- Suggested there could be a percentile that yields >91.3% accuracy

**Root Cause:**
- AI defaulted to intuitive choices (median, round numbers) without optimization
- Did not consider that tier separability varies across the price distribution
- Did not account for model retraining resilience

**Resolution:**
- Created `optimize_tier_split.py` to systematically test percentiles 25-75%
- Added multiple optimization criteria (CV accuracy, min-tier accuracy, balanced)
- Implemented fine-tuning with 1% steps around optimal
- Made system percentile-based for retraining resilience

**Lesson Learned:**
- Default choices (median, round numbers) should be validated, not assumed
- Optimization should consider real-world constraints (price appreciation)
- Human domain insight often identifies constraints AI misses

---

## Correction #10: F-String Escape Sequence Error

**Date:** 2025-12-08
**Phase:** V2.7 - Tiered Models Exploration

**What AI Did Wrong:**
- Used escaped single quote in f-string: `{'Cohen\\'s d':>10}`
- Python f-strings don't allow backslash escapes in the expression part
- Script failed with `SyntaxError: invalid syntax`

**Human Intervention:**
- User ran script and reported syntax error

**Root Cause:**
- Backslash escapes are not allowed inside f-string curly braces
- AI didn't test the script before providing it

**Resolution:**
- Simplified to `{'Cohens d':>10}` (removed apostrophe)
- Alternative: use a variable outside the f-string

**Lesson Learned:**
- F-string expressions have restrictions on backslashes
- Test code snippets, especially with complex string formatting
- Simple solutions (removing special chars) often better than complex escaping

---

## Correction #11: ROI vs Complexity - Tiered Models Decision

**Date:** 2025-12-08
**Phase:** V2.7 - Tiered Models Exploration

**What AI Did (Not Wrong, But Incomplete):**
- Ran extensive experiments on tiered model approaches
- Found a configuration that technically "won" (+0.17% improvement)
- Was prepared to continue iterating to find more improvements
- Created 6 experimental scripts with routing logic

**Human Intervention:**
- User insight: "it doesn't look like this is having enough return on value for the added complexity"
- Suggested pivoting to V3.1 (MLOps infrastructure) instead
- Prioritized long-term infrastructure over marginal model gains

**Root Cause:**
- AI was optimizing locally (find the best tiered config)
- Did not step back to evaluate ROI of the entire approach
- Added complexity (multiple models, routing logic) for minimal gain
- Engineering judgment required: when to stop optimizing

**Resolution:**
- Documented V2.7 as "Explored - Insufficient ROI"
- Preserved experimental scripts for future reference
- Pivoted focus to V3.1 (CI/CD, MLflow, GitHub Actions)

**Lesson Learned:**
- **Not all improvements are worth implementing**
- 0.17% MAE improvement = $111/prediction
- But: 2x models to maintain, routing logic, increased failure modes
- Human judgment on ROI vs complexity is essential
- AI tends to continue optimizing; humans know when to stop

---

## Correction #12: Test Fixture Feature Name Mismatch

**Date:** 2025-12-08
**Phase:** V3.1 - MLOps & CI/CD

**What AI Did Wrong:**
- Created `tests/test_model.py` with a `sample_input` fixture
- Used made-up demographic feature names like `avg_household_size`, `households`, `housing_density`
- Model expects specific names from `zipcode_demographics.csv` like `ppltn_qty`, `per_urbn`, `medn_hshld_incm_amt`
- 7 out of 13 tests failed with `ValueError: The feature names should match those that were passed during fit`

**Human Intervention:**
- User ran `pytest tests/ -v` and reported failures
- Error message clearly showed feature name mismatch

**Root Cause:**
- AI invented plausible-sounding demographic feature names instead of checking the actual data
- Did not reference `model/model_features.json` (the source of truth)
- Assumed generic feature naming conventions

**Resolution:**
- Read `model/model_features.json` to get exact 43 feature names
- Updated `sample_input` fixture with correct names:
  - Home features: `bedrooms`, `bathrooms`, `sqft_living`, etc.
  - Demographics: `ppltn_qty`, `urbn_ppltn_qty`, `medn_hshld_incm_amt`, etc.

**Lesson Learned:**
- **Always check the source of truth** for feature names, schemas, configs
- Test fixtures must match production data schemas exactly
- Don't invent feature names - look them up

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total Corrections | 12 |
| Protocol Violations | 3 |
| Sequencing Errors | 1 |
| Performance/Runtime Issues | 1 |
| Deployment Verification Issues | 2 |
| Type/Serialization Issues | 2 |
| Design/Optimization Issues | 1 |
| ROI/Complexity Decisions | 1 |
| Test/Fixture Issues | 1 |
| Corrections Leading to New Artifacts | 4 |

---

## V2.4 Specific Insights

V2.4 introduced new categories of corrections:

1. **Runtime Constraints** - AI-generated parameter spaces were impractical
2. **Silent Failures** - Shell issues caused commands to "succeed" without effect
3. **Heuristic Decay** - Old version detection logic broke with new models

These highlight the importance of:
- Testing AI-generated code with realistic data
- Verifying operations, not just command completion
- Reviewing legacy logic when introducing changes

---

## V2.7 Specific Insights

V2.7 (Tiered Models) taught important lessons about exploration vs exploitation:

1. **Hypothesis-Driven Experimentation** - User suggested percentile-based splits (Correction #9)
2. **Systematic Grid Search** - Tested 49 configurations to find optimum
3. **Knowing When to Stop** - 0.17% gain rejected due to complexity (Correction #11)

**Technical Learnings:**
- Misrouting penalty can negate specialist benefits
- High-price homes have too much variance for specialization
- Low-price specialist works but gains are marginal
- Single model often handles heterogeneity internally (XGBoost trees)

**Process Learnings:**
- Document experiments even when they fail
- ROI evaluation requires stepping back from local optimization
- Infrastructure investment often beats marginal model improvements

---

## V3.1 Specific Insights

V3.1 (MLOps & CI/CD) highlighted the importance of schema consistency:

1. **Test Fixtures Must Match Production** - Correction #12 showed test data must use exact feature names
2. **Source of Truth** - `model/model_features.json` is authoritative, not assumptions
3. **Quick Feedback Loops** - Tests catch schema mismatches before deployment

**Technical Learnings:**
- MLflow needs `Path.as_uri()` for Windows artifact paths
- Ruff ignores (E402, B023, B904) should be documented with rationale
- Black + Ruff work well together for consistent code style

---

## Correction #13: Model Should Be Version-Controlled

**Date:** 2025-12-09
**Phase:** V3.2 - Fresh Data Integration

**What AI Did Wrong:**
- `.gitignore` was excluding `model/` directory
- Model file not tracked in version control

**Human Intervention:**
- User flagged: "I saw this message and thought we'd want the models in the repo"
- Directed AI to track the model

**Root Cause:**
- Default assumption that binary files should be gitignored
- Did not consider that 1.8MB model is small enough to version-control

**Resolution:**
- Updated `.gitignore` to include `model/` (removed from exclusion)
- Model now tracked alongside code

**Lesson Learned:**
- Small binary artifacts can and should be version-controlled
- "Best practice" gitignore templates may not fit all projects

---

## Correction #14: Misunderstanding Join Rate vs Feature Mapping

**Date:** 2025-12-09
**Phase:** V3.2 - Fresh Data Integration

**What AI Did Wrong:**
- Interpreted 72% parcel join rate as 72% feature coverage
- Conflated data join success with feature mapping completeness

**Human Intervention:**
- User clarified: "That was the rate at which parcels were joined to sales, not the rate at which you're mapping features from kc_house_data.csv to new data"
- Forced precise understanding of the distinction

**Root Cause:**
- Sloppy terminology conflated two different metrics
- Did not carefully distinguish data pipeline stages

**Resolution:**
- Clarified: 15/17 features directly mapped from assessment data
- 2 features (lat/long) required external GIS data
- Proper accounting: 88% direct mapping, 12% needs external source

**Lesson Learned:**
- Be precise about what percentages measure
- Different pipeline stages have different success metrics
- User domain knowledge is essential for correct interpretation

---

## Correction #15: GIS Data for Real Coordinates

**Date:** 2025-12-09
**Phase:** V3.2 - Fresh Data Integration

**What AI Did Wrong:**
- Used synthetic lat/long approximations (zipcode centroids + random offset)
- Did not know King County provides real parcel coordinates

**Human Intervention:**
- User uploaded 844MB GeoJSON file with real parcel geometries
- Provided the authoritative data source

**Root Cause:**
- AI unaware of available external data sources
- Settled for "good enough" synthetic solution

**Resolution:**
- Created `scripts/extract_parcel_centroids.py` to extract centroids from polygon geometries
- Generated `data/parcel_centroids.csv` with 637,540 real lat/long values
- Achieved 100% coordinate match rate

**Impact:**
- CV MAE improved from $258,958 to $236,161 (8.8% reduction)
- Model now uses real geographic coordinates

**Lesson Learned:**
- Domain experts know what data exists
- Ask about available data sources before creating synthetic alternatives
- Real data beats approximations

---

## V3.2 Specific Insights

V3.2 (Fresh Data Integration) highlighted the value of real data:

1. **External Data Sources** - User provided GIS data AI didn't know existed (Correction #15)
2. **Precise Terminology** - Join rates vs feature mapping are different metrics (Correction #14)
3. **Version Control Decisions** - Small binaries can be tracked (Correction #13)

**Technical Learnings:**
- King County GeoJSON contains parcel polygons, not centroids
- Centroid extraction from GeoJSON requires simple coordinate averaging
- Streaming JSON parsing handles 844MB files efficiently
- Real lat/long provides ~9% MAE improvement over synthetic

**Process Learnings:**
- Always ask users about available data sources
- Distinguish between pipeline stages when reporting metrics
- Question default gitignore patterns for project-specific needs

---

## Correction #16: Questioning High R2 (Overfitting Detection)

**Date:** 2025-12-09
**Phase:** V3.3 - Optimization

**What AI Did Wrong:**
- Reported test R2 of 0.966 without questioning it
- Presented inflated metric as a success

**Human Intervention:**
- User asked: "that 96.6 feels high, is the model overfitted?"
- Prompted investigation into potential data leakage

**Root Cause:**
- AI focused on metric improvement without skepticism
- Did not recognize that 96.6% R2 is unusually high for real estate prediction

**Resolution:**
- Investigated dataset: found 12.4% repeat sales (same property sold multiple times)
- Random train/test split leaked information (same property in both sets)
- Implemented GroupKFold CV to split by property ID
- Honest R2 = 0.868 vs inflated R2 = 0.966

**Lesson Learned:**
- High metrics warrant skepticism, not celebration
- Always consider data leakage in panel/time-series data
- Entity-level splits (GroupKFold) prevent information leakage

---

## Correction #17: CI/CD Using Wrong Data Source

**Date:** 2025-12-09
**Phase:** V3.3 - Optimization

**What AI Did Wrong:**
- Updated training scripts to use fresh 2020+ data
- Left `train_with_mlflow.py` pointing to old `kc_house_data.csv`
- CI/CD workflows not updated for new data source

**Human Intervention:**
- User asked: "teach me about how our ci/cd github/mlflow MLOPS performed in 3.2 and 3.3"
- Exposed gap between local development and CI/CD configuration

**Root Cause:**
- AI updated training scripts but forgot MLOps integration
- Tunnel vision on model development, not full pipeline

**Resolution:**
- Added `--data-source` flag to `train_with_mlflow.py` (fresh/original)
- Updated GitHub Actions workflows to support feature/* branches
- Train workflow now defaults to fresh data

**Lesson Learned:**
- When data sources change, all pipeline components must update
- CI/CD is part of the deliverable, not an afterthought
- End-to-end testing catches integration gaps

---

## V3.3 Specific Insights

V3.3 (Optimization) highlighted the importance of skepticism:

1. **Metric Skepticism** - User's intuition caught overfitting AI missed (Correction #16)
2. **Full Pipeline Thinking** - MLOps scripts must evolve with data sources (Correction #17)

**Technical Learnings:**
- Repeat property sales cause data leakage in random splits
- GroupKFold CV provides honest evaluation for panel data
- 12.4% repeat sales = significant leakage potential
- R2 0.868 is honest and still excellent for real estate

**Process Learnings:**
- Question metrics that seem "too good"
- CI/CD configuration is part of feature development
- Human skepticism is a safety net for AI optimism

---

## Value of Human Oversight

These corrections demonstrate that:

1. **AI can follow instructions but miss intent** - Following "write train.py" literally without understanding it must be run
2. **Protocols require active enforcement** - Reading rules is not the same as applying them
3. **Human intuition catches gaps** - User sensed something was wrong before AI recognized it
4. **Collaboration produces better outcomes** - Neither AI nor human alone would have caught all issues
5. **Domain experts provide critical data** - V3.2 showed humans know what external sources exist
6. **Human skepticism catches inflated metrics** - V3.3 showed overfitting that AI celebrated

---

**Log Maintained By:** AI Assistant
**Last Updated:** 2025-12-09 (V3.3 Session End)
