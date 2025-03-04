import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load GA and RL results CSVs.
# (Assume you have saved these files as "ga_overall_stats.csv" and "rl_overall_stats.csv")
ga_df = pd.read_csv("ga_overall_stats.csv")
rl_df = pd.read_csv("rl_overall_stats.csv")


# Combine data from all epochs into a single DataFrame for easier plotting.
# For example, create a new column that indicates the absolute generation number across epochs.
def add_absolute_generation(df, generations_per_epoch):
    df = df.copy()
    df['Absolute Generation'] = (df['Epoch'] - 1) * generations_per_epoch + df['Generation']
    return df


ga_df = add_absolute_generation(ga_df, generations_per_epoch=50)
rl_df = add_absolute_generation(rl_df, generations_per_epoch=50)

# Extract data for the first and last epoch
ga_first = ga_df[ga_df['Epoch'] == 1]
ga_last = ga_df[ga_df['Epoch'] == ga_df['Epoch'].max()]
rl_first = rl_df[rl_df['Epoch'] == 1]
rl_last = rl_df[rl_df['Epoch'] == rl_df['Epoch'].max()]

# Group the data by Epoch, summing reproduction and death counts over all generations in an epoch.
ga_epoch = ga_df.groupby("Epoch")[["Reproduction Count", "Death Count"]].sum().reset_index()
rl_epoch = rl_df.groupby("Epoch")[["Reproduction Count", "Death Count"]].sum().reset_index()

# Compute the overall average (across epochs) for reproduction and death counts.
ga_avg_repro = ga_epoch["Reproduction Count"].mean()
ga_avg_death = ga_epoch["Death Count"].mean()
rl_avg_repro = rl_epoch["Reproduction Count"].mean()
rl_avg_death = rl_epoch["Death Count"].mean()

# Create a multi-page PDF report
with PdfPages("reports/simulation_comparison_plots_reprodcost20.pdf") as pdf:
    # Plot 1: Total Creatures Comparison
    plt.figure(figsize=(10, 6))
    # plt.plot(ga_df['Absolute Generation'], ga_df['Total Creatures'], marker='o', label='GA Total Creatures')
    # plt.plot(rl_df['Absolute Generation'], rl_df['Total Creatures'], marker='o', label='RL Total Creatures')
    plt.plot(ga_df['Absolute Generation'], ga_df['Total Creatures'], label='GA Total Creatures')
    plt.plot(rl_df['Absolute Generation'], rl_df['Total Creatures'], label='RL Total Creatures')
    plt.xlabel("Absolute Moves")
    plt.ylabel("Total Creatures")
    plt.title("Population Dynamics Comparison (GA vs. RL)")
    plt.legend()
    plt.grid(True)
    pdf.savefig()
    plt.close()

    # Plot 2: Average Energy Comparison
    plt.figure(figsize=(10, 6))
    # plt.plot(ga_df['Absolute Generation'], ga_df['Average Energy'], marker='o', color='blue', label='GA Average Energy')
    # plt.plot(rl_df['Absolute Generation'], rl_df['Average Energy'], marker='o', color='red', label='RL Average Energy')
    plt.plot(ga_df['Absolute Generation'], ga_df['Average Energy'], color='blue', label='GA Average Energy')
    plt.plot(rl_df['Absolute Generation'], rl_df['Average Energy'], color='red', label='RL Average Energy')
    plt.xlabel("Absolute Moves")
    plt.ylabel("Average Energy")
    plt.title("Average Energy Over Moves (GA vs. RL)")
    plt.legend()
    plt.grid(True)
    pdf.savefig()
    plt.close()

    # Plot 3: Speed and Size Distribution Comparison (First vs. Last Epoch)
    # Group 1: Speed Distribution Comparison
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # GA First Epoch Speed Distribution
    axs[0, 0].plot(ga_first['Generation'], ga_first['Speed 1'], marker='o', label='Speed 1')
    axs[0, 0].plot(ga_first['Generation'], ga_first['Speed 2'], marker='o', label='Speed 2')
    axs[0, 0].plot(ga_first['Generation'], ga_first['Speed 3'], marker='o', label='Speed 3')
    axs[0, 0].set_title("GA Speed (Epoch 1)")
    axs[0, 0].set_xlabel("Move")
    axs[0, 0].set_ylabel("Count")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # GA Last Epoch Speed Distribution
    axs[0, 1].plot(ga_last['Generation'], ga_last['Speed 1'], marker='o', label='Speed 1')
    axs[0, 1].plot(ga_last['Generation'], ga_last['Speed 2'], marker='o', label='Speed 2')
    axs[0, 1].plot(ga_last['Generation'], ga_last['Speed 3'], marker='o', label='Speed 3')
    axs[0, 1].set_title("GA Speed (Last Epoch)")
    axs[0, 1].set_xlabel("Move")
    axs[0, 1].set_ylabel("Count")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # RL First Epoch Speed Distribution
    axs[1, 0].plot(rl_first['Generation'], rl_first['Speed 1'], marker='o', label='Speed 1')
    axs[1, 0].plot(rl_first['Generation'], rl_first['Speed 2'], marker='o', label='Speed 2')
    axs[1, 0].plot(rl_first['Generation'], rl_first['Speed 3'], marker='o', label='Speed 3')
    axs[1, 0].set_title("RL Speed (Epoch 1)")
    axs[1, 0].set_xlabel("Move")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # RL Last Epoch Speed Distribution
    axs[1, 1].plot(rl_last['Generation'], rl_last['Speed 1'], marker='o', label='Speed 1')
    axs[1, 1].plot(rl_last['Generation'], rl_last['Speed 2'], marker='o', label='Speed 2')
    axs[1, 1].plot(rl_last['Generation'], rl_last['Speed 3'], marker='o', label='Speed 3')
    axs[1, 1].set_title("RL Speed (Last Epoch)")
    axs[1, 1].set_xlabel("Move")
    axs[1, 1].set_ylabel("Count")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # Group 2: Size Distribution Comparison
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # GA First Epoch Size Distribution
    axs[0, 0].plot(ga_first['Generation'], ga_first['Size 1'], marker='o', label='Size 1')
    axs[0, 0].plot(ga_first['Generation'], ga_first['Size 2'], marker='o', label='Size 2')
    axs[0, 0].plot(ga_first['Generation'], ga_first['Size 3'], marker='o', label='Size 3')
    axs[0, 0].set_title("GA Size (Epoch 1)")
    axs[0, 0].set_xlabel("Move")
    axs[0, 0].set_ylabel("Count")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # GA Last Epoch Size Distribution
    axs[0, 1].plot(ga_last['Generation'], ga_last['Size 1'], marker='o', label='Size 1')
    axs[0, 1].plot(ga_last['Generation'], ga_last['Size 2'], marker='o', label='Size 2')
    axs[0, 1].plot(ga_last['Generation'], ga_last['Size 3'], marker='o', label='Size 3')
    axs[0, 1].set_title("GA Size (Last Epoch)")
    axs[0, 1].set_xlabel("Move")
    axs[0, 1].set_ylabel("Count")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # RL First Epoch Size Distribution
    axs[1, 0].plot(rl_first['Generation'], rl_first['Size 1'], marker='o', label='Size 1')
    axs[1, 0].plot(rl_first['Generation'], rl_first['Size 2'], marker='o', label='Size 2')
    axs[1, 0].plot(rl_first['Generation'], rl_first['Size 3'], marker='o', label='Size 3')
    axs[1, 0].set_title("RL Size (Epoch 1)")
    axs[1, 0].set_xlabel("Move")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # RL Last Epoch Size Distribution
    axs[1, 1].plot(rl_last['Generation'], rl_last['Size 1'], marker='o', label='Size 1')
    axs[1, 1].plot(rl_last['Generation'], rl_last['Size 2'], marker='o', label='Size 2')
    axs[1, 1].plot(rl_last['Generation'], rl_last['Size 3'], marker='o', label='Size 3')
    axs[1, 1].set_title("RL Size (Last Epoch)")
    axs[1, 1].set_xlabel("Move")
    axs[1, 1].set_ylabel("Count")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()


    # Plot 4: Total Reproduction vs. Death Events per Epoch for GA and RL
    plt.figure(figsize=(10, 6))
    # Plot GA totals
    plt.plot(ga_epoch['Epoch'], ga_epoch['Reproduction Count'], marker='o', linestyle='-',
             label='GA Total Reproduction')
    plt.plot(ga_epoch['Epoch'], ga_epoch['Death Count'], marker='o', linestyle='-', label='GA Total Death')
    # Plot RL totals
    plt.plot(rl_epoch['Epoch'], rl_epoch['Reproduction Count'], marker='s', linestyle='-',
             label='RL Total Reproduction')
    plt.plot(rl_epoch['Epoch'], rl_epoch['Death Count'], marker='s', linestyle='-', label='RL Total Death')

    # Add horizontal lines for overall averages.
    plt.axhline(y=ga_avg_repro, color='blue', linestyle='--', label='GA Avg Reproduction')
    plt.axhline(y=ga_avg_death, color='orange', linestyle='--', label='GA Avg Death')
    plt.axhline(y=rl_avg_repro, color='green', linestyle='--', label='RL Avg Reproduction')
    plt.axhline(y=rl_avg_death, color='red', linestyle='--', label='RL Avg Death')

    plt.xlabel("Epoch")
    plt.ylabel("Total Count")
    plt.title("Total Reproduction vs. Death Events per Epoch (GA vs. RL)")
    # Place legend outside the plot area.
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='8')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    pdf.savefig(bbox_inches='tight')
    plt.close()

print("All figures have been saved to 'reports/simulation_comparison_plots.pdf'")
