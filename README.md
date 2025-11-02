# Bio-Interfaced Translational Nanoengineering Group (VIP)
Jan 2025 – Apr 2025

This project investigates how to improve EEG-based finger movement classification by applying Riemannian geometry to represent the multidimensional structure of neural data more accurately. Traditional Euclidean methods fail to capture the curvature of covariance matrices derived from EEG signals, resulting in low classification accuracy.

# Approach
- Data Source: EEG recordings of individual finger movements.
- Problem: EEG signals exhibit complex geometric relationships that are poorly modeled in Euclidean space.
- Method:
    - Used Riemannian geometry to extract covariance-based features from EEG data.
    - Compared performance across Euclidean, Log-Euclidean, and Riemannian feature spaces.
    - Trained LDA classifiers for movement classification.
 
| Method         | Accuracy |
| -------------- | -------- |
| Euclidean      | 18%      |
| Log-Euclidean  | 83%      |
| **Riemannian** | **89%**  |

Riemannian geometry significantly outperformed traditional approaches, demonstrating its ability to capture the intrinsic structure of EEG signals on symmetric positive definite (SPD) manifolds.

🚀 Future Work
- Test on new EEG datasets to validate generalization.
- Explore methods to reduce class overlap in feature space.
- Develop a real-time brain–computer interface (BCI) that decodes individual finger movements.
