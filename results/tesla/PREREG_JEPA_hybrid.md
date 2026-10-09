# Pre-registration 4: JEPA hybrids (written 2026-10-06, before any hybrid result)

Data: the authors' own preprocessed CSVs (Google Drive, row counts identical to TESLA Table A1).
Control: LogTCN from scratch on the same data, 3 seeds (orig_logtcn run).
Pretrained trunks: the existing CalJEPA trunks (unchanged; units used as validation/test excluded).

Variants (code in experiments/tsl_jepa.py, fixed now):
  CalJEPA-joint   pretrained trunk + JEPA auxiliary loss on unlabelled fleet windows during fine-tuning
                  (lambda 0.1, EMA 0.996, same VICReg terms as pretraining)
  CalJEPA-frozen  fresh LogTCN trunk + frozen pretrained JEPA trunk, head reads both
  CalJEPA-gentle  pretrained trunk fine-tuned at 0.1 x the head learning rate

Selection (validation only): each variant trained once (seed 0) on all 9 cells; the variant with the
lowest mean VALIDATION RMSE relative to LogTCN seed 0 (mean over cells of val_variant / val_LogTCN) is
selected. Test numbers of the unselected variants are reported but do not change the choice.

WIN for the selected variant (3 seeds), against LogTCN from scratch:
 1. lower mean test RMSE in >= 6 of 9 cells; city-averaged gain >= 3% for at least two PM sizes;
    no cell worse by more than 5%;
 2. unit rotation, PM2.5, all cities: better than LogTCN on >= 60% of units, median per-unit gain >= 3%.
If no variant wins, JEPA is dropped from this line of work.

## Outcome (2026-10-06)
Stopped early after the PM10 cells (deviation from the 9-cell plan, documented): every variant had a worse VALIDATION RMSE than LogTCN seed 0 in every cell run (Antwerp: joint 9.15, frozen 9.29, gentle 9.79 vs 9.01; Oslo: joint 8.57, frozen 6.91, gentle 7.71 vs LogTCN val ~7.2-7.3), and worse test RMSE (Antwerp 14.98 / 11.77 / 12.78 vs ~10.5). No variant can be selected as an improvement; JEPA hybrids FAILED. JEPA is dropped from this line of work.
