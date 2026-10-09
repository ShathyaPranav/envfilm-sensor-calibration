# Pre-registration 5: label-scarce calibration (written 2026-10-06, before any run)

Hypothesis: JEPA pretraining on ~32M unlabelled fleet minutes helps when labels are scarce.
Setting: TESLA protocol on the authors' CSVs, but every training unit and the validation unit keep
only their FIRST 1440 labelled minutes (24 h; TSL_LABEL_MINUTES=1440). Test unit: full, as published.
Recipe for every model: Adam 1e-3, batch 512, up to 200 epochs, early stopping with patience 15 on
the (24 h) validation unit.
Models: LogTCN from scratch (control); CalJEPA-TCN (init), CalJEPA-frozen, CalJEPA-gentle.
The JEPA variant is chosen by mean validation RMSE ratio to LogTCN over the 9 cells (seed 0).
WIN for the chosen variant (3 seeds), vs LogTCN from scratch in the same label budget:
  lower mean test RMSE in >= 7 of 9 cells, city-averaged gain >= 5% for at least two PM sizes,
  and no cell worse by more than 5%.
Also reported: the same comparison at 6 h and 7 days of labels (one seed), as a label-budget curve.
