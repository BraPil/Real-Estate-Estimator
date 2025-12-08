# Git Script: Complete V2.1.x and Setup V2.3
# Run this from the Real-Estate-Estimator directory

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host " V2.1.x COMPLETION AND V2.3 SETUP" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan

# Step 1: Check current status
Write-Host "`n[STEP 1] Current Git Status" -ForegroundColor Yellow
git status
git branch -a
git log --oneline -5

# Step 2: Stage all changes
Write-Host "`n[STEP 2] Staging all changes..." -ForegroundColor Yellow
git add -A

# Step 3: Show what will be committed
Write-Host "`n[STEP 3] Changes to be committed:" -ForegroundColor Yellow
git status --short

# Step 4: Commit V2.1.x changes
Write-Host "`n[STEP 4] Committing V2.1.x changes..." -ForegroundColor Yellow
$commitMessage = @"
feat(v2.1): Complete V2.1.x feature expansion and experimental endpoints

V2.1: Feature Expansion
- Expanded model from 33 to 43 features (+10 home features)
- Added: waterfront, view, condition, grade, yr_built, yr_renovated,
  lat, long, sqft_living15, sqft_lot15
- MAE improved from $102k to $90k (-12%)
- R² improved from 0.728 to 0.768 (+5.5%)
- Overfitting gap reduced by 20%

V2.1.1: /predict-full Endpoint
- New endpoint accepts all 17 home features without zipcode
- Uses average demographics
- Best single strategy for no-zipcode predictions

V2.1.2: /predict-adaptive Endpoint (Experimental)
- Discovered price-tier pattern (confirmed statistically)
- LOW PRIORITY: Routing accuracy (52%) negates benefit
- Decision: Accept /predict-full as best no-zipcode strategy

Documentation:
- docs/Changes_from_v1_to_v2.1_20251208.md
- docs/V2.1_Lessons_Learned.md
- docs/V2.1.2_Adaptive_Endpoint_Discovery.md
- docs/V2.1.2_Adaptive_Routing_Conclusion.md
- docs/V2_Detailed_Roadmap.md (updated)

Next: V2.3 Hyperparameter Tuning (skipping V2.2 for higher ROI)
"@

git commit -m $commitMessage

# Step 5: Check which branch we're on
Write-Host "`n[STEP 5] Current branch:" -ForegroundColor Yellow
$currentBranch = git rev-parse --abbrev-ref HEAD
Write-Host "  Current branch: $currentBranch"

# Step 6: Push to remote
Write-Host "`n[STEP 6] Pushing to remote..." -ForegroundColor Yellow
git push origin $currentBranch

# Step 7: Merge to develop if on feature branch
Write-Host "`n[STEP 7] Merging feature branch to develop..." -ForegroundColor Yellow
if ($currentBranch -like "feature/*") {
    Write-Host "  On feature branch, merging to develop..."
    git checkout develop
    git merge $currentBranch --no-edit
    git push origin develop
    Write-Host "  Merged $currentBranch to develop" -ForegroundColor Green
} else {
    Write-Host "  Not on feature branch, skipping merge" -ForegroundColor Gray
}

# Step 8: Create V2.3 branch
Write-Host "`n[STEP 8] Creating V2.3 feature branch..." -ForegroundColor Yellow
git checkout -b feature/v2.3-hyperparameter-tuning
git push -u origin feature/v2.3-hyperparameter-tuning

# Step 9: Final status
Write-Host "`n[STEP 9] Final Status:" -ForegroundColor Yellow
git branch -a
git log --oneline -3

Write-Host "`n" + "=" * 70 -ForegroundColor Cyan
Write-Host " COMPLETE! Now on feature/v2.3-hyperparameter-tuning" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Cyan
