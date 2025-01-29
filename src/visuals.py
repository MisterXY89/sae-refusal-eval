import torch
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_activations(activations):
    """
    CLI-based visualization of the activations.
    Prompts the user to input an activation key to visualize.

    Args:
    activations (dict): Dictionary of activations with layer names as keys.
    """

    # List all activation keys
    print("Available activations:")
    for i, key in enumerate(activations.keys()):
        print(f"{i}: {key}")

    # Ask the user to input a choice
    selected_index = int(input("Enter the index of the activation to visualize: "))

    # Get the key from the user’s input
    selected_key = list(activations.keys())[selected_index]
    activation = activations[selected_key]

    # Check if the activation is 3D or 2D and handle accordingly
    if len(activation.shape) == 3:  # e.g., (batch_size, seq_length, hidden_dim)
        activation = activation[0].detach().cpu()  # Select the first element in the batch
    elif len(activation.shape) == 2:  # e.g., (seq_length, hidden_dim)
        activation = activation.detach().cpu()
    else:
        print(f"Cannot visualize activation with shape {activation.shape}")
        return

    # Plot the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(activation, cmap="viridis", cbar=True)
    plt.title(f"Activation Heatmap for {selected_key}")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Sequence Position")
    plt.show()

# Example usage:
# visualize_activations_cli(activations)
