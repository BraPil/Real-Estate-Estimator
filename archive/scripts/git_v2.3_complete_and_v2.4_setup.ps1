# Git Script: Complete V2.3 and Setup V2.4
# Run this from the Real-Estate-Estimator directory

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host " V2.3 COMPLETION AND V2.4 SETUP" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan

# Step 1: Check current status
Write-Host "`n[STEP 1] Current Git Status" -ForegroundColor Yellow
git status
git branch -a
git log --oneline -3

# Step 2: Stage all changes
Write-Host "`n[STEP 2] Staging all changes..." -ForegroundColor Yellow
git add -A

# Step 3: Show what will be committed
Write-Host "`n[STEP 3] Changes to be committed:" -ForegroundColor Yellow
git status --short

# Step 4: Commit V2.3 changes
Write-Host "`n[STEP 4] Committing V2.3 changes..." -ForegroundColor Yellow
$commitMessage = @"
feat(v2.3): Complete hyperparameter tuning - MAE improved 5.9%

V2.3: Hyperparameter Tuning with GridSearchCV
- Tested 126 parameter combinations with 5-fold CV
- Best params: n_neighbors=7, weights=distance, metric=manhattan
- Test MAE: $89,769 -> $84,494 (-5.9% improvement)
- Test R²: 0.7682 -> 0.7932 (+3.3% improvement)
- Overfitting gap: 5.6% (acceptable)

New files:
- src/tune.py - GridSearchCV tuning script
- model/model_v2.3_tuned.pkl - Tuned model artifact
- model/tuning_results.json - Best parameters and metrics
- logs/v2.3_grid_search_results.csv - All 126 combinations

Documentation:
- docs/V2.3_Completion_Summary.md
- docs/Changes_from_v2.1_to_v2.3_20251208.md
- docs/V2.3_Lessons_Learned.md

Production model updated: model/model.pkl

Next: V2.4 Model Alternatives (Random Forest, XGBoost)
"@

git commit -m $commitMessage

# Step 5: Check which branch we're on
Write-Host "`n[STEP 5] Current branch:" -ForegroundColor Yellow
$currentBranch = git rev-parse --abbrev-ref HEAD
Write-Host "  Current branch: $currentBranch"

# Step 6: Push to remote
Write-Host "`n[STEP 6] Pushing to remote..." -ForegroundColor Yellow
git push origin $currentBranch

# Step 7: Merge to develop
Write-Host "`n[STEP 7] Merging to develop..." -ForegroundColor Yellow
git checkout develop
git merge $currentBranch --no-edit
git push origin develop
Write-Host "  Merged $currentBranch to develop" -ForegroundColor Green

# Step 8: Create V2.4 branch
Write-Host "`n[STEP 8] Creating V2.4 feature branch..." -ForegroundColor Yellow
git checkout -b feature/v2.4-model-alternatives
git push -u origin feature/v2.4-model-alternatives

# Step 9: Final status
Write-Host "`n[STEP 9] Final Status:" -ForegroundColor Yellow
git branch -a
git log --oneline -3

Write-Host "`n" + "=" * 70 -ForegroundColor Cyan
Write-Host " COMPLETE! Now on feature/v2.4-model-alternatives" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Cyan

Write-Host "`n[SUMMARY] Version Progress:" -ForegroundColor Magenta
Write-Host "  Original MAE: ~$120,000" -ForegroundColor White
Write-Host "  V1 MAE:       $102,045  (-15%)" -ForegroundColor White
Write-Host "  V2.1 MAE:     $ 89,769  (-12%)" -ForegroundColor White
Write-Host "  V2.3 MAE:     $ 84,494  (-5.9%)" -ForegroundColor Green
Write-Host "  Total:        -30% improvement!" -ForegroundColor Green
