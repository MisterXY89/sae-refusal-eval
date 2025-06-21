The analysis of the Sparse Autoencoder (SAE) training results reveals several key insights into the interplay between training data composition, hyperparameter settings, and SAE performance. The most significant finding is the clear benefit of in-domain training. SAEs evaluated on pre-training data (c4-sample) perform best when trained on pre-training-heavy data mixtures (PRE, EQ), while SAEs evaluated on instruction-following data (GAIR/lima) perform best when trained on instruction-heavy mixtures (INS, lmsys).

Increasing the SAE's expansion_factor and the training token size generally leads to better reconstruction quality, but with diminishing returns, especially for token size. A clear trade-off exists between reconstruction fidelity and sparsity, where higher expansion factors lead to better reconstruction but lower sparsity. The results from different model layers (6 and 25) show similar overall trends, but with layer 25 consistently achieving higher explained variance, suggesting it captures more complex and important features.
Detailed Analysis and Interpretation
1. The Critical Role of In-Domain Training Data

The most striking pattern in the results is the strong performance of SAEs when their training data domain matches the evaluation data domain.

    On Pre-training Data (c4-sample): When evaluated on the c4-sample dataset, SAEs trained on data with a higher proportion of pre-training text (PRE and EQ) consistently outperform those trained on instruction-heavy data (INS and lmsys). This is evident across all metrics, including higher explained variance and lower mean squared error (MSE).

    On Instruction-Following Data (GAIR/lima): Conversely, when evaluated on the GAIR/lima instruction-following dataset, SAEs trained on INS and lmsys (which have more instruction-tuning data) perform the best.

This "in-domain advantage" is clearly visible in the plots below, which show the explained variance for different training and evaluation dataset combinations, separated by model layer.

Interpretation: This strongly suggests that SAEs learn features specific to the domain of their training data. For optimal performance, the SAE should be trained on data that is representative of the data it will encounter during inference. The EQ (50/50) and INS/PRE (70/30) mixes act as sensible intermediate points, but the most specialized datasets (lmsys for instructions and by extension, a pure pre-training dataset for pre-training) provide the best performance in their respective domains.
2. Impact of Expansion Factor, Training Tokens, and Model Layer
Expansion Factor

A larger sae_expansion_factor consistently leads to better reconstruction quality (higher explained_variance and lower mse). This is expected, as a larger expansion factor provides the SAE with more capacity (more "neurons") to represent the features of the input data. However, this comes at the cost of sparsity, as shown in the trade-off analysis below.
Training Token Size

Increasing the number of training tokens (sae_token_size_mil) also improves reconstruction performance, but with diminishing returns. The biggest performance jump is seen when moving from 50M to 125M tokens. The gains from 125M to 250M and further to 419M are progressively smaller. This suggests that after a certain point, simply adding more training data is less effective than other interventions, like changing the data mixture or adjusting the expansion factor.
Model Layer (6 vs. 25)

The trends observed are consistent across both layer 6 and layer 25. However, layer 25, being a deeper layer in the model, consistently achieves a higher explained_variance than layer 6. This is also expected, as deeper layers in a transformer model tend to capture more abstract and higher-level features, which are more critical for the model's overall performance. The SAEs on layer 25 are therefore reconstructing more of the variance of these more important features.
3. The Sparsity vs. Reconstruction Trade-off

A fundamental trade-off in SAE design is between reconstruction fidelity (how well the original input can be recovered) and the sparsity of the learned features. The results demonstrate this clearly:

    Higher explained_variance correlates with lower sparsity (activation_sparsity_l0): As the SAE's ability to reconstruct the input improves (higher explained_variance), the number of activated features also tends to increase (lower sparsity).
    Expansion factor is a key driver: The sae_expansion_factor is a primary lever for controlling this trade-off. Larger expansion factors lead to better reconstruction but at the cost of denser activations.

This relationship is visualized in the scatter plot below, which shows explained_variance against activation_sparsity_l0, with points colored by the sae_expansion_factor.

Interpretation: The choice of sae_expansion_factor is not a simple matter of "bigger is better." It requires a careful balancing act. For applications where preserving the model's performance is paramount, a higher expansion factor is desirable. However, for applications where interpretability and computational efficiency are the main goals, a lower expansion factor (leading to higher sparsity) might be preferred, even at the cost of some reconstruction accuracy.
Conclusions and Recommendations

    Prioritize In-Domain Training Data: The single most important factor for achieving high-performing SAEs is the use of training data that matches the target application domain. If the goal is to understand and intervene on a model's behavior on instruction-following tasks, the SAEs must be trained on instruction-following data.

    Tune Expansion Factor for the Desired Trade-off: The sae_expansion_factor is the primary hyperparameter for controlling the trade-off between reconstruction quality and sparsity. The optimal value will depend on the specific goals of the SAE application.

    Invest in Data Quality over Quantity (Beyond a Certain Point): While a sufficient amount of training data is necessary, the results suggest that after a certain threshold (around 125M-250M tokens in this experiment), the composition of the training data is more impactful than simply increasing its size.

    Focus on Deeper Layers for Maximum Impact: Intervening on deeper layers (like layer 25) is likely to have a more significant impact on the model's behavior, as these layers capture more of the variance of the model's internal representations.