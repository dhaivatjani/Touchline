# Benchmark tables

Live-play filter: grass test, plus ≥ 4 players and median player height ≤ 20% of frame. Applied to every table except 1b.

## 1. One full match per angle (equal basis: `Camera N.mp4`, 10 fps)

| angle | PVR | Players/frame | ≥18 visible | Worst 10% (players) | BVR | Player height (% frame) | Pan speed (frame-w/s) | Pan p95 | Zoom rate (log-scale/s) | Zoom p95 | Gameplay s | Gameplay share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.845 | 18.9 | 0.677 | 15 | 0.775 | 6.72 | 0.0927 | 0.299 | 0.0256 | 0.107 | 256 | 0.733 |
| 2 | 0.791 | 17.6 | 0.6 | 12 | 0.731 | 8.14 | 0.0913 | 0.293 | 0.0429 | 0.152 | 266 | 0.59 |
| 3 | 0.734 | 16.3 | 0.405 | 11 | 0.736 | 8.57 | 0.0942 | 0.318 | 0.0667 | 0.234 | 289 | 0.801 |
| 4 | 0.821 | 18.6 | 0.611 | 13 | 0.749 | 7.14 | 0.0908 | 0.318 | 0.0797 | 0.26 | 291 | 0.718 |
| 5 | 0.759 | 17 | 0.514 | 10 | 0.749 | 8.9 | 0.109 | 0.36 | 0.0421 | 0.17 | 319 | 0.67 |

## 1a. 95% block-bootstrap intervals (30 s blocks), table 1

| angle | PVR | ≥18 visible | BVR | Zoom rate |
|---|---|---|---|---|
| 1 | 0.845 [0.814, 0.876] | 0.678 [0.565, 0.781] | 0.775 [0.706, 0.839] | 0.026 [0.021, 0.034] |
| 2 | 0.794 [0.733, 0.842] | 0.607 [0.466, 0.717] | 0.734 [0.675, 0.777] | 0.043 [0.037, 0.052] |
| 3 | 0.735 [0.705, 0.770] | 0.405 [0.307, 0.505] | 0.739 [0.676, 0.795] | 0.067 [0.053, 0.080] |
| 4 | 0.822 [0.773, 0.867] | 0.612 [0.479, 0.735] | 0.747 [0.699, 0.788] | 0.080 [0.071, 0.089] |
| 5 | 0.762 [0.706, 0.812] | 0.520 [0.409, 0.630] | 0.746 [0.671, 0.808] | 0.041 [0.035, 0.049] |

## 1b. Table 1 with the grass filter only (cutscenes included)

| angle | PVR | Players/frame | ≥18 visible | Worst 10% (players) | BVR | Player height (% frame) | Pan speed (frame-w/s) | Pan p95 | Zoom rate (log-scale/s) | Zoom p95 | Gameplay s | Gameplay share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.768 | 17.2 | 0.608 | 8 | 0.708 | 6.77 | 0.0939 | 0.299 | 0.0315 | 0.126 | 285 | 0.812 |
| 2 | 0.651 | 14.5 | 0.478 | 1 | 0.639 | 8.23 | 0.0944 | 0.309 | 0.0488 | 0.168 | 334 | 0.744 |
| 3 | 0.685 | 15.2 | 0.375 | 8 | 0.694 | 8.6 | 0.097 | 0.327 | 0.0705 | 0.239 | 312 | 0.867 |
| 4 | 0.707 | 16 | 0.517 | 2 | 0.666 | 7.31 | 0.0911 | 0.321 | 0.0769 | 0.26 | 344 | 0.852 |
| 5 | 0.683 | 15.3 | 0.447 | 5 | 0.696 | 9.01 | 0.108 | 0.361 | 0.0488 | 0.196 | 366 | 0.77 |

## 1c. Sensitivity of full-match PVR to the live-play filter

| filter | C1 PVR | C2 PVR | C3 PVR | C4 PVR | C5 PVR | PVR ranking |
|---|---|---|---|---|---|---|
| players ≥ 0, median height ≤ 1.00 | 0.768 | 0.651 | 0.685 | 0.707 | 0.683 | C1 > C4 > C3 > C5 > C2 |
| players ≥ 4, median height ≤ 0.20 | 0.845 | 0.791 | 0.734 | 0.821 | 0.759 | C1 > C4 > C2 > C5 > C3 |
| players ≥ 6, median height ≤ 0.20 | 0.845 | 0.799 | 0.74 | 0.824 | 0.768 | C1 > C4 > C2 > C5 > C3 |
| players ≥ 6, median height ≤ 0.15 | 0.845 | 0.807 | 0.744 | 0.831 | 0.778 | C1 > C4 > C2 > C5 > C3 |

## 2. All full-match recordings per angle

| angle | clips | PVR | Players/frame | ≥18 visible | Worst 10% (players) | BVR | Player height (% frame) | Pan speed (frame-w/s) | Pan p95 | Zoom rate (log-scale/s) | Zoom p95 | Gameplay s | Gameplay share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.845 | 18.9 | 0.677 | 15 | 0.775 | 6.72 | 0.0927 | 0.299 | 0.0256 | 0.107 | 256 | 0.733 |
| 2 | 1 | 0.791 | 17.6 | 0.6 | 12 | 0.731 | 8.14 | 0.0913 | 0.293 | 0.0429 | 0.152 | 266 | 0.59 |
| 3 | 2 | 0.7 | 15.5 | 0.298 | 10 | 0.754 | 8.54 | 0.125 | 0.463 | 0.0593 | 0.21 | 564 | 0.785 |
| 4 | 3 | 0.811 | 18.4 | 0.612 | 13 | 0.742 | 7 | 0.0821 | 0.288 | 0.0761 | 0.259 | 805 | 0.695 |
| 5 | 1 | 0.759 | 17 | 0.514 | 10 | 0.749 | 8.9 | 0.109 | 0.36 | 0.0421 | 0.17 | 319 | 0.67 |

## 3. KickOff clips (the only situation recorded on all five angles)

| angle | PVR | Players/frame | ≥18 visible | Worst 10% (players) | BVR | Player height (% frame) | Pan speed (frame-w/s) | Pan p95 | Zoom rate (log-scale/s) | Zoom p95 | Gameplay s | Gameplay share | Median track length (s) | Detections in tracks ≥2 s | New IDs per player-minute |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.773 | 17 | 0.443 | 14 | 0.853 | 6.98 | 0.0868 | 0.29 | 0.0371 | 0.106 | 18 | 1 | 4.64 | 0.705 | 44.8 |
| 2 | 0.86 | 19.5 | 0.786 | 16 | 0.767 | 8.07 | 0.0895 | 0.303 | 0.0359 | 0.119 | 16.7 | 1 | 3.94 | 0.839 | 30.5 |
| 3 | 0.783 | 17.2 | 0.49 | 15 | 0.734 | 8.84 | 0.0586 | 0.287 | 0.0916 | 0.344 | 17.1 | 1 | 4.15 | 0.792 | 31.1 |
| 4 | 0.916 | 21.1 | 0.827 | 17 | 0.753 | 6.22 | 0.0738 | 0.249 | 0.0759 | 0.322 | 20.8 | 1 | 10.7 | 0.839 | 51.4 |
| 5 | 0.801 | 18.6 | 0.445 | 13 | 0.753 | 9.02 | 0.117 | 0.407 | 0.0355 | 0.121 | 23.3 | 1 | 5.06 | 0.725 | 54.5 |

## 4. PVR by situation (situation clips, full frame rate)

| situation | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| CornerKick | 0.944 |  | 0.825 | 0.902 | 0.782 |
| CounterAttack |  | 0.673 | 0.777 | 0.864 | 0.772 |
| FreeKick |  |  |  | 0.839 | 0.471 |
| GoalKick | 0.849 |  | 0.753 | 0.925 | 0.598 |
| KickOff | 0.773 | 0.86 | 0.783 | 0.916 | 0.801 |
| SustainedAttack | 0.881 | 0.833 | 0.685 | 0.872 | 0.785 |
| SwitchOfPlay |  |  |  | 0.87 |  |

## 4. BVR by situation (situation clips, full frame rate)

| situation | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| CornerKick | 0.961 |  | 0.573 | 0.757 | 0.692 |
| CounterAttack |  | 0.79 | 0.85 | 0.751 | 0.873 |
| FreeKick |  |  |  | 0.732 | 0.841 |
| GoalKick | 0.803 |  | 0.857 | 0.843 | 0.799 |
| KickOff | 0.853 | 0.767 | 0.734 | 0.753 | 0.753 |
| SustainedAttack | 0.866 | 0.799 | 0.728 | 0.769 | 0.948 |
| SwitchOfPlay |  |  |  | 0.748 |  |

## 4. Player height (% frame) by situation (situation clips, full frame rate)

| situation | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| CornerKick | 6.64 |  | 9.12 | 6.7 | 11.1 |
| CounterAttack |  | 8.19 | 8.59 | 6.93 | 8.94 |
| FreeKick |  |  |  | 6.77 | 8.89 |
| GoalKick | 6.63 |  | 8.28 | 6.51 | 8.57 |
| KickOff | 6.98 | 8.07 | 8.84 | 6.22 | 9.02 |
| SustainedAttack | 7.18 | 8.24 | 8.83 | 6.6 | 9.23 |
| SwitchOfPlay |  |  |  | 6.55 |  |

## 4. Pan speed (frame-w/s) by situation (situation clips, full frame rate)

| situation | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| CornerKick | 0.0694 |  | 0.0406 | 0.0449 | 0.0354 |
| CounterAttack |  | 0.0943 | 0.0773 | 0.0902 | 0.113 |
| FreeKick |  |  |  | 0.073 | 0.194 |
| GoalKick | 0.134 |  | 0.0944 | 0.0653 | 0.152 |
| KickOff | 0.0868 | 0.0895 | 0.0586 | 0.0738 | 0.117 |
| SustainedAttack | 0.0615 | 0.0782 | 0.103 | 0.0426 | 0.0593 |
| SwitchOfPlay |  |  |  | 0.0797 |  |

## 5. All situation clips per angle, incl. tracking

| angle | clips | PVR | Players/frame | ≥18 visible | Worst 10% (players) | BVR | Player height (% frame) | Pan speed (frame-w/s) | Pan p95 | Zoom rate (log-scale/s) | Zoom p95 | Gameplay s | Gameplay share | Median track length (s) | Detections in tracks ≥2 s | New IDs per player-minute |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.842 | 18.9 | 0.659 | 15 | 0.868 | 6.8 | 0.0871 | 0.352 | 0.0346 | 0.118 | 41.7 | 1 | 3.12 | 0.71 | 42.9 |
| 2 | 3 | 0.778 | 17.7 | 0.612 | 9 | 0.782 | 8.14 | 0.0896 | 0.303 | 0.0521 | 0.178 | 39.6 | 0.968 | 4.16 | 0.788 | 35.8 |
| 3 | 5 | 0.747 | 16.4 | 0.439 | 11 | 0.769 | 8.66 | 0.0824 | 0.325 | 0.0769 | 0.29 | 88.4 | 0.953 | 5.7 | 0.858 | 29.6 |
| 4 | 10 | 0.888 | 20.6 | 0.758 | 16 | 0.779 | 6.6 | 0.0695 | 0.271 | 0.0754 | 0.305 | 192 | 1 | 5.73 | 0.84 | 40.6 |
| 5 | 6 | 0.714 | 16.1 | 0.368 | 10 | 0.813 | 8.96 | 0.118 | 0.456 | 0.0429 | 0.171 | 77.2 | 1 | 5.06 | 0.81 | 43.5 |

## 6. Per clip

| angle / clip | situation | PVR | ≥18 visible | BVR | Player height (% frame) | Pan speed (frame-w/s) | Zoom rate (log-scale/s) | Gameplay share | Detections in tracks ≥2 s |
|---|---|---|---|---|---|---|---|---|---|
| 1 / Camera 1 | FullMatch | 0.845 | 0.677 | 0.775 | 6.72 | 0.0927 | 0.0256 | 0.733 |  |
| 1 / Camera 1 - AdressingConcern | SustainedAttack | 0.881 | 0.722 | 0.866 | 7.18 | 0.0615 | 0.0519 | 1 | 0.846 |
| 1 / Camera 1 - CornerKick | CornerKick | 0.944 | 1 | 0.961 | 6.64 | 0.0694 | 0.0178 | 1 | 0.607 |
| 1 / Camera 1 - GoalKick | GoalKick | 0.849 | 0.721 | 0.803 | 6.63 | 0.134 | 0.0306 | 1 | 0.703 |
| 1 / Camera 1 - KickOff | KickOff | 0.773 | 0.443 | 0.853 | 6.98 | 0.0868 | 0.0371 | 1 | 0.705 |
| 2 / Camera 2 | FullMatch | 0.791 | 0.6 | 0.731 | 8.14 | 0.0913 | 0.0429 | 0.59 |  |
| 2 / Camera 2 - AdressingConcern | SustainedAttack | 0.833 | 0.588 | 0.799 | 8.24 | 0.0782 | 0.0505 | 1 | 0.825 |
| 2 / Camera 2 - CounterAttack | CounterAttack | 0.673 | 0.444 | 0.79 | 8.19 | 0.0943 | 0.0693 | 0.926 | 0.703 |
| 2 / Camera 2 - KickOff | KickOff | 0.86 | 0.786 | 0.767 | 8.07 | 0.0895 | 0.0359 | 1 | 0.839 |
| 3 / Camera 3 | FullMatch | 0.734 | 0.405 | 0.736 | 8.57 | 0.0942 | 0.0667 | 0.801 |  |
| 3 / Camera 3 (1) | FullMatch | 0.663 | 0.186 | 0.773 | 8.48 | 0.158 | 0.0516 | 0.769 |  |
| 3 / Camera 3 (1) - AdressingConcern | SustainedAttack | 0.685 | 0.266 | 0.728 | 8.83 | 0.103 | 0.0523 | 0.973 | 0.883 |
| 3 / Camera 3 - CornerKick | CornerKick | 0.825 | 0.938 | 0.573 | 9.12 | 0.0406 | 0.0497 | 1 | 0.98 |
| 3 / Camera 3 - CounterAttack | CounterAttack | 0.777 | 0.463 | 0.85 | 8.59 | 0.0773 | 0.13 | 0.864 | 0.875 |
| 3 / Camera 3 - GoalKick | GoalKick | 0.753 | 0.505 | 0.857 | 8.28 | 0.0944 | 0.0316 | 1 | 0.785 |
| 3 / Camera 3 - KickOff | KickOff | 0.783 | 0.49 | 0.734 | 8.84 | 0.0586 | 0.0916 | 1 | 0.792 |
| 4 / Camera 4 | FullMatch | 0.821 | 0.611 | 0.749 | 7.14 | 0.0908 | 0.0797 | 0.718 |  |
| 4 / Camera 4 (1) | FullMatch | 0.802 | 0.634 | 0.723 | 6.8 | 0.0718 | 0.0717 | 0.633 |  |
| 4 / Camera 4 (1) - AttackInMotion | SustainedAttack | 0.874 | 0.765 | 0.869 | 6.58 | 0.0578 | 0.066 | 1 | 0.857 |
| 4 / Camera 4 (1) - ChangingSideOfAttack | SwitchOfPlay | 0.87 | 0.737 | 0.748 | 6.55 | 0.0797 | 0.0754 | 1 | 0.793 |
| 4 / Camera 4 (1) - CounterAttack | CounterAttack | 0.838 | 0.743 | 0.674 | 7.03 | 0.0672 | 0.116 | 1 | 0.881 |
| 4 / Camera 4 (1) - GoalKick | GoalKick | 0.947 | 0.941 | 0.91 | 6.25 | 0.0586 | 0.054 | 1 | 0.936 |
| 4 / Camera 4 (1) - InBoxDrama | SustainedAttack | 0.869 | 0.679 | 0.673 | 6.62 | 0.0279 | 0.05 | 1 | 0.921 |
| 4 / Camera 4 (2) | FullMatch | 0.81 | 0.587 | 0.757 | 7.09 | 0.0838 | 0.077 | 0.758 |  |
| 4 / Camera 4 (2) - CornerKick | CornerKick | 0.902 | 0.778 | 0.757 | 6.7 | 0.0449 | 0.0786 | 1 | 0.874 |
| 4 / Camera 4 (2) - ShrotFreeKick&ChangeOfWingAndCross | FreeKick | 0.839 | 0.601 | 0.732 | 6.77 | 0.073 | 0.0658 | 1 | 0.657 |
| 4 / Camera 4 - Counter | CounterAttack | 0.879 | 0.638 | 0.796 | 6.85 | 0.103 | 0.102 | 1 | 0.803 |
| 4 / Camera 4 - GoalKick&BuildUp | GoalKick | 0.912 | 0.831 | 0.805 | 6.64 | 0.0691 | 0.065 | 1 | 0.816 |
| 4 / Camera 4 - KickOff&Clearence | KickOff | 0.916 | 0.827 | 0.753 | 6.22 | 0.0738 | 0.0759 | 1 | 0.839 |
| 5 / Camera 5 | FullMatch | 0.759 | 0.514 | 0.749 | 8.9 | 0.109 | 0.0421 | 0.67 |  |
| 5 / Camera 5 - CornerKick | CornerKick | 0.782 | 0.618 | 0.692 | 11.1 | 0.0354 | 0.0522 | 1 | 0.882 |
| 5 / Camera 5 - CounterAttack | CounterAttack | 0.772 | 0.506 | 0.873 | 8.94 | 0.113 | 0.0454 | 1 | 0.895 |
| 5 / Camera 5 - FreeKick | FreeKick | 0.471 | 0.055 | 0.841 | 8.89 | 0.194 | 0.0276 | 1 | 0.801 |
| 5 / Camera 5 - GoalKick | GoalKick | 0.598 | 0.115 | 0.799 | 8.57 | 0.152 | 0.0531 | 1 | 0.756 |
| 5 / Camera 5 - KickOff | KickOff | 0.801 | 0.445 | 0.753 | 9.02 | 0.117 | 0.0355 | 1 | 0.725 |
| 5 / Camera 5 - NearGoalLine | SustainedAttack | 0.785 | 0.459 | 0.948 | 9.23 | 0.0593 | 0.0549 | 1 | 0.89 |
