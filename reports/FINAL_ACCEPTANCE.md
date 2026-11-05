# Final project acceptance

Audit date: 2026-11-05. Branch: `finalization/submission`. Baseline: `c0ae203` (latest fetched origin/main at audit start). Previous local main: `22dba50`. Working tree was clean before changes. All remote branches were fetched; `origin/feature-danh` and `origin/nmt05` contain older work, while current Week 4/5 contributions are on main. No branch was merged or pushed by this task.

## Verdict

**CONDITIONAL GO for academic submission**: independent package execution passed; team confirmation of the provisional Champion and the official portal/rubric checks remain manual. Executable artifact verification is independent of the earlier clean-environment report. This is not a claim of unbiased end-to-end CV or production readiness. If the rubric requires leakage-free end-to-end CV, the historical experiment does not satisfy that requirement and a new fold-local experiment is needed.

## Requirement coverage and findings

The inspected five-week specification is the team's local `Bản sao của BTL_ML.md` (60,645 bytes), sections Data, Features, Model + Training, Evaluation + Prediction and Deployment + Delivery. It requires the report Sections 1–6, CLI, four integrated notebooks, artifacts, figures, 12–15 slides, a 10–12 minute script and an ID-named ZIP. The original official assignment/rubric is not present. The team plan gives no numerical upload-size cap. User confirmed the ID string is a naming format; package order follows Person 1→4 in README.

| Phase | Baseline findings | Final disposition |
|---|---|---|
| 1 Data/EDA | Four notebooks, required figures, correct 11,162-row CSV present; no consolidated editable report | All four notebooks executed unchanged; report integrates observed findings |
| 2 Features | 24 features, one MI strategy, 74 encoded candidates, 32 selected; saved pipeline and splits present | Raw/saved train and test values and row labels independently match; original source/artifacts preserved |
| 3 Models | Three trained models, CV records and learning curve figure present; 05 notebook missing from main | Restored from existing local commit `2f94d03`, with provenance and safe temporary outputs; executed from clean kernel |
| 4 Evaluation | Lợi metrics, Trí plots, Danh error analysis and Nhân benchmark present; notebook still says Trí missing | 06 corrected, original figures integrated, fresh execution passes; predictive metrics independently match |
| 5 Delivery | CLI, web, Docker, slides/script present; Report.pdf, report_draft.docx and ZIP missing | Added report PDF and editable Markdown/DOCX, corrected presentation text, supplied package builder and manifest |

The team plan originally asks for 2–4 engineered features and 3–5 core challenges; earlier explicit project instructions requested 24 features and approximately 35 observations. Finalization preserves the accepted 24-feature implementation and summarizes four evidence-based core data challenges. It does not add arbitrary features or inflate challenges to satisfy a mistaken reading of a range.

## Substantiated defects corrected

1. `run_pipeline.py --mode evaluate` failed on a clean Windows Python with `TclError: Can't find a usable init.tcl`. The Week 5 CLI now sets the noninteractive `Agg` backend before importing plot modules. Evaluation then completed. No teammate plotting logic was rewritten.
2. `src/predict.py --model` previously labeled sigmoid SVC margins as probabilities and assigned certainty 1 to label-only estimators. Margin-only models now export `decision_score`, unavailable probability/confidence fields and explicit `score_kind`; RF continues to export predict_proba labeled uncalibrated. Removed blanket UserWarning suppression and imported the missing `Sequence` typing name.
3. README named nonexistent figures/files, described a 200-tree/depth-10 Champion despite the actual 300-tree/depth-20 artifact, claimed team approval, confused GitHub's per-file limit with repository/submission size, and omitted notebook tooling. Corrected affected entries, added explicit scipy (direct runtime import) and nbconvert (notebook execution), and documented safe reproduction in copies.
4. Restored 05 from genuine Git history. Original code/narrative remain substantially intact; outputs go to a temporary directory so reproduction cannot overwrite historical JSON/report/figure. Added a methodological qualification and retained actual new outputs. Logistic default gives 0.6995 in this environment versus historical 0.6990; tuned CV and final learning-curve summaries match to four decimals. An additional sequential five-fold check with warnings enabled found all default Logistic fits reached max_iter=100 and emitted ConvergenceWarning (mean 0.6994773388). This baseline is not fully converged; the precise cause of the historical numerical difference is unproven. The notebook and report disclose this rather than changing the inherited estimator. No search history or training duration was invented.
5. Slide/report statements that a small CV–Test gap proves no overfit, that paired McNemar p-values test Macro-F1 or prove equivalence, and that duration establishes persuasion/irreducible error were corrected. Timing values now use the saved 03/11 baseline consistently. The 0.41 threshold is labeled retrospective Test illustration, not an approved operating rule. Slide count/layout/media retained.
6. Final evaluation notebook now recognizes Trí's module, displays its three figures and uses current benchmark numbers; earlier reports remain historical evidence rather than being silently overwritten.

## Scientific qualifications that remain

- **Within-Train CV leakage:** preprocessing and supervised MI selection were fitted once on all 8,929 Train rows before CV. Test never participates in fit, but validation folds influence the selected representation. Bias magnitude is unknown. Report, recovered notebook, slides and script disclose this. Existing trained artifacts and historical scores were not replaced with a different experiment. A claim of fully leakage-free pipeline CV would require a separate fold-local experiment.
- **Post-selection Test:** the held-out set is now used for comparing models, recommending Champion and error analysis. Future deployment assessment needs new temporal/campaign data; retrospective threshold exploration is not an independent estimate.
- **Input timing:** duration is removed. campaign/contact/day/month require a real pre-call availability contract; historical records alone do not establish availability at campaign launch.
- **Clipping semantics:** previous may become fractional after IQR clipping (upper bound 2.5). Model compatibility is preserved; this is a disclosed historical transformation limitation.
- **Probability semantics:** RF/LR probabilities are not calibrated. Historic SVC sigmoid scores are useful for ranking but not probability claims. The recorded PR-AUC field uses Average Precision.
- **Approval:** champion recommendation JSON remains provisional. No team vote or production SLA was invented. Official portal size cap and full original rubric remain external information.

## Independent execution evidence

Created a new venv outside the checkout at `E:\Downloads\BTL_ML_acceptance_20261105\venv`, using Python 3.13.0. Installed `requirements-dev.txt` and nbconvert (now included there). Runtime versions: sklearn 1.9.1, NumPy 2.5.3, pandas 3.0.5, scipy 1.18.1. `pip check`: no broken requirements. Document authoring packages were installed afterward for report/PDF work, not required by ML.

| Command/workflow | Actual outcome |
|---|---|
| `python -m pytest -q` | 18 passed, including target/duration invariance, missing schema, margin semantics and headless runner regression |
| `python -m compileall -q src scripts` | Passed |
| `python scripts/verify_submission.py --output reports/final_acceptance_evidence.json` | Passed: raw-to-saved matrices, target row alignment, finite schema, all six metrics per family, original artifact hashes |
| `python src/predict.py --input data/sample_test.csv --output <outside-workspace>/cli_predictions.csv` | Passed, 10 rows, 8 no / 2 yes, no notebook state |
| `python run_pipeline.py --mode preprocess` in Git archive copy | Passed; regenerated data/pipeline checked against preserved originals |
| `python run_pipeline.py --mode evaluate` in archive copy | Initially failed Tk backend; passed after isolated runner fix, including metrics, plots, error analysis and benchmark |
| `AppTest.from_file('app.py').run()` | No Streamlit exceptions on initial render; not a complete browser/UI or hosting acceptance test |
| Docker build | Optional; not executed because Docker availability was not established |
| Full three-model GridSearch retraining | Not performed; protected artifacts preserved. Notebook reproduces CV/learning curves with in-memory clones, not replacement models |

Notebook execution used an explicitly registered temporary kernel pointing to the new venv, in a Git archive copy so historical notebook writes affect only that copy:

| Notebook | Result | Seconds |
|---|---|---:|
| 00_EDA_Final | Passed | 19.145 |
| 01_data_quality | Passed | 10.510 |
| 02_distributions | Passed | 8.495 |
| 03_relationships | Passed | 5.477 |
| 04_features_pipeline | Passed | 5.271 |
| 05_model_experiments | Passed | 346.334 |
| 06_evaluation, baseline | Passed | 6.491 |
| 06_evaluation, corrected final integration | Passed | 8.294 |

Exact execution records are in `final_notebook_validation.json`. The new 05/06 notebook outputs are current reproductions. Other notebook files remain unchanged. Jupyter emitted Windows event-loop/kernel transport warnings, with no cell execution errors.

## Metric and artifact verification

| Model | Macro-F1 | Precision yes | Recall yes | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|
| Logistic | 0.7097707112 | 0.7383918460 | 0.6162570888 | 0.7605236697 | 0.7653725231 |
| Random Forest | 0.7262630555 | 0.7626146789 | 0.6285444234 | 0.7747472147 | 0.7773827504 |
| SVC | 0.7184555640 | 0.7835443038 | 0.5850661626 | 0.7728785746 | 0.7726802181 |

All values recomputed from saved estimators and current raw/saved held-out data. Champion SHA-256 matches best_tree_model.pkl (`a7768013e4feaaca536fb858123720d24625b23f84e36d287deb44e5edb8d283`). Model-only benchmark values in the report/slides use the preserved c0ae203 snapshot (Intel 20 logical CPUs, Python 3.12.3, 03/11/2026), not a fabricated timing requirement. Re-running timing on another machine is expected to vary.

## Academic artifacts and preservation

`docs/Report.pdf`: 15 pages, Sections 1–6, five numbered tables, six figures, feature appendix, evidence map and references. Editable content: `docs/Report.md`, `reports/report_draft.docx`; build script included. Report created from teammate sources with attribution. No historical report was replaced. Native Word PDF export was used because the bundled document runtime was unavailable; page renders were visually reviewed.

`docs/Slides.pptx` and PDF: 14 slides; original layouts/media retained, specific text/numbers and corresponding speaker notes corrected. Danh's speaking roles, approximate 11:15 schedule and Q&A structure preserved. Final artifact claims were checked against metric JSON/CSV and direct predictions. PPTX and PDF were reopened/rendered after editing; no missing glyphs detected. No assertion of team approval or probability calibration was added.

Week 1–3 source modules, existing notebooks 00–04, all model/preprocessing artifacts, processed datasets, figures and historical CSV reports are unchanged relative to c0ae203. Only restored notebook 05 is added. Existing Week 4 notebook 06 is updated; evaluation source modules stay unchanged. The deployment CLI/runner are the only existing runtime source files changed.

## Submission package

Builder: `python scripts/package_submission.py --output-dir <directory outside checkout>`.
Filename and top-level folder: `2411969-2313598-2310409-2412432` (Lợi, Trí, Danh, Nhân).

Includes source, tests, all notebooks/figures, all three models plus Champion and preprocessing, raw bank.csv, processed splits, sample input, report source/PDF, slides/PDF/script, provenance evidence and instructions. Raw archive.zip is omitted because bank.csv is included; duplicate sample predictions and internal planning documents are omitted. Git metadata, venvs and caches are excluded. A manifest records bytes and SHA-256 for every packaged file. All runtime data/artifacts are local; no external download link is required or fabricated.

The package is approximately 29.4 decimal MB, with 115 files including the manifest. The exact final byte count and SHA-256 are supplied in the delivery handoff, avoiding a self-referential archive checksum. Unknown portal cap is not replaced with an assumed 100 MB rule.

The candidate ZIP was extracted to a new directory under the Windows temporary directory on C:, outside the development repository. PYTHONPATH was cleared. Installation from its own requirements-dev.txt and pip check passed. Its own pytest suite passed all 18 tests (5.03 s); compileall and verify_submission passed. The CLI was invoked from an unrelated working directory using absolute paths to the extracted script and sample CSV: all 10 rows returned, 8 no / 2 yes. Preprocess and evaluate then completed within the disposable extraction; verification after reproduction again matched the saved feature values and reported predictive metrics. Model fitting was limited to preprocessing in this disposable copy, not retraining the three saved estimators.

The first extraction attempt on E: failed with OSError 28 because that volume was full. Only the task-created baseline audit directory/archive were moved to C: to free space; no user or repository work was removed. Testing on C: succeeded. This machine-specific storage failure is not hidden as a passing attempt. Final delivery uses a fresh extraction and verifies every manifest hash before running read-only tests again.

All commits are local on finalization/submission; no push or merge. The original repository's model/data hashes and protected Week 1–3 files are unchanged. Git delivery hashes and final status are supplied in the handoff.
