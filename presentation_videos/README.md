# Matched Rollout Videos

These videos are presentation-ready qualitative comparisons between Plain GRU
and Reliability-Gated GRU under the controlled corruption setting:

- training noise probability: `0.2`;
- inference noise probability: `0.2`;
- inference noise seed: `42`;
- corruption: replace the current WayObject Costmap with a zero map;
- evaluation: blacklist-filtered HM3D IIN validation episodes.

## How to Read the Videos

- Left panel: Plain GRU.
- Right panel: Reliability-Gated GRU.
- A red border marks a step where costmap corruption was injected.
- The right panel reports the reliability gate's current `alpha`.
- Both methods receive the same deterministic per-episode corruption schedule.

The displayed `alpha` should not be interpreted as a calibrated corruption
detector. The original gate achieved pooled ROC-AUC `0.312` when `1 - alpha`
was used as the corruption score. The videos demonstrate navigation behavior,
not successful frame-level failure detection.

## Selected Episodes

| Task | Video | Plain GRU | Reliability-Gated GRU | Final-distance improvement |
|---|---|---:|---:|---:|
| Imitate | [video](imitate_6s7QHgap2fW_0000000_tv_monitor_34_matched.mp4) | exceeded steps, 2.729 m | success, 0.993 m | 1.735 m |
| Alt-Goal | [video](alt_goal_yr17PDCnDDW_0000000_plant_32_matched.mp4) | exceeded steps, 6.875 m | success, 0.978 m | 5.897 m |
| Shortcut | [video](shortcut_bxsVRursffK_0000000_toilet_27_matched.mp4) | exceeded steps, 8.710 m | success, 0.991 m | 7.719 m |
| Reverse | [video](reverse_wcojb4TFT35_0000000_bed_17_matched.mp4) | exceeded steps, 5.741 m | success, 0.996 m | 4.746 m |

## Selection and Interpretation

For each task, the analysis script considered non-blacklisted episodes where
Reliability-Gated GRU succeeded and Plain GRU failed, then selected the episode
with the largest final-distance improvement. These are deliberately selected
success cases for qualitative mechanism analysis. They do not establish that
Reliability-Gated GRU wins on every episode and must be presented alongside the
aggregate results:

- Reliability-Gated GRU noisy Avg SPL: `60.16`;
- Plain GRU noisy Avg SPL: `58.80`;
- difference: `+1.36` points, from one training seed.

For a short presentation, the Shortcut or Alt-Goal video gives the clearest
visual contrast. The complete contact sheet is available at
[`paper_figures/matched_rollouts_contact_sheet.pdf`](../paper_figures/matched_rollouts_contact_sheet.pdf).
