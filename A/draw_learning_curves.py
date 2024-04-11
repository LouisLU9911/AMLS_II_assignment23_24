import matplotlib.pyplot as plt

# Assuming 'loss_values' is your list of loss values recorded after each epoch or iteration
base_model_loss = [
    0.9249,
    0.5028,
    0.4421,
    0.4009,
    0.3821,
    0.3672,
    0.3544,
    0.3287,
    0.327,
    0.3239,
    0.3113,
    0.3146,
    0.3031,
    0.3075,
    0.3071,
    0.3028,
]
expert_1_loss = [
    1.2488,
    1.2488,
    1.2488,
    1.2488,
    1.2488,
    1.2488,
    1.2488,
    1.2488,
    1.2488,
    1.2488,
    0.6707,
    0.6707,
    0.6707,
    0.5104,
    0.5104,
    0.5104,
]
expert_2_loss = [
    1.5107,
    1.0022,
    0.4503,
    0.3704,
    0.2996,
    0.2405,
    0.2268,
    0.2036,
    0.1957,
    0.1885,
    0.1866,
    0.1771,
    0.1752,
    0.1744,
    0.1636,
    0.1636,
]
switch_gate_loss = [
    0.6169,
    0.3115,
    0.2375,
    0.1999,
    0.1916,
    0.1771,
    0.168,
    0.1592,
    0.1542,
    0.1521,
    0.1501,
    0.1426,
    0.1389,
    0.1391,
    0.136,
    0.1327,
]
moe_loss = [
    0.3197,
    0.3197,
    0.3197,
    0.3197,
    0.3197,
    0.3197,
    0.3197,
    0.2995,
    0.2995,
    0.2995,
    0.2995,
    0.2996,
    0.2996,
    0.2996,
    0.2996,
    0.3024,
]

# Plotting the learning curve
plt.figure(figsize=(6, 6))  # Set the figure size (optional)
plt.plot(base_model_loss, label="Base Model", marker="o")
plt.plot(expert_1_loss, label="Expert 1", marker="o")
plt.plot(expert_2_loss, label="Expert 2", marker="o")
plt.plot(switch_gate_loss, label="Switch Gate Network", marker="o")
plt.plot(moe_loss, label="MoE Model", marker="o")
plt.title("Learning Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curves.png")
