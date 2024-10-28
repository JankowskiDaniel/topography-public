import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_epsilon_distribution(df, title):
    plt.hist(df['epsilon'], bins=10, range=(0, 1), color='red', alpha=0.5, edgecolor='black')
    plt.title(title)
    plt.xlabel('Epsilon')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.show()


def generate_merged_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df1 = df1.rename(columns={"epsilon": "model_epsilon"})
    df2 = df2.rename(columns={"epsilon": "exact_epsilon"})
    both = df1.merge(df2, how='inner', on='filename')
    both["abs_diff"] = abs(both["model_epsilon"]-both["exact_epsilon"])
    both["cmae"] = np.min(np.array([list(both["abs_diff"]), list(1-both["abs_diff"])]), axis=0)
    both["filenr"] = both["filename"].apply(lambda x: int(x.split(".")[0]))
    def calculate_adjusted_loss(row):
        # Calculate the absolute difference
        absolute_difference = abs(row["model_epsilon"] - row["exact_epsilon"])
        # Considering the cyclic nature, find the minimum difference
        adjusted_loss = min(absolute_difference, 1 - absolute_difference)
        
        # Determine if it's an underestimation or overestimation
        if row["model_epsilon"] < row["exact_epsilon"]:
            if row["exact_epsilon"] - row["model_epsilon"] > 0.5:  # the values are on opposite sides of the cycle
                adjusted_loss = abs(adjusted_loss)  # it's an overestimation
            else:
                adjusted_loss = -abs(adjusted_loss)  # it's an underestimation
        else:
            if row["model_epsilon"] - row["exact_epsilon"] > 0.5:  # the values are on opposite sides of the cycle
                adjusted_loss = -abs(adjusted_loss)  # it's an underestimation
            else:
                adjusted_loss = abs(adjusted_loss)  # it's an overestimation

        return adjusted_loss
    both['adjusted_loss'] = both.apply(calculate_adjusted_loss, axis=1)
    return both

SPECIAL_RANGES = [i for i in range(9344, 9518)] + \
                [i for i in range(27351, 27514)] + \
                [i for i in range(45351, 45518)] + \
                [i for i in range(63351, 63518)] + \
                [i for i in range(81351, 81518)] + \
                [i for i in range(99351, 99518)]

def plot_average_cmae(df: pd.DataFrame, title: str, special_ranges: list = SPECIAL_RANGES, threshold: float = 0.1):
    indices = []
    cmaes = []
    for x in range(0, 99000, 100):

        sample = df[(df["filenr"] > x) & (df["filenr"] < x+100)]
        cmae = sample["cmae"].mean()
        cmaes.append(cmae)
        indices.append(x)
    
    # Plot the average CMAE, mark in green special ranges, and mark in red ranges with CMAE > threshold
    plt.plot(indices, cmaes)
    plt.title(title)
    plt.xlabel('File number')
    plt.ylabel('Average CMAE')
    for special_range in special_ranges:
        plt.axvline(x=special_range, color='r', linestyle='--')
    # set fixed y-axis limits from 0.0 to 0.5
    plt.ylim(0.0, 0.5)


def plot_ranged_loss(df: pd.DataFrame, title: str):
    # Plot the average CMAE in prediction ranges [[0.0 - 0.1], [0.1 - 0.2], ..., [0.9 - 1.0]]
    indices = []
    cmaes = []
    for x in range(0, 10):
        sample = df[(df["exact_epsilon"] >= x/10) & (df["exact_epsilon"] < (x+1)/10)]
        cmae = sample["cmae"].mean()
        cmaes.append(cmae)
        indices.append(x/10)
    # plot the histogram
    plt.bar(indices, cmaes, width=0.1)
    # Separate the bars
    plt.xticks(np.arange(0, 1.1, 0.1))



# Function to plot individual distributions
def plot_epsilon_distribution_ax(df, title, ax, color='red'):
    ax.hist(df['epsilon'], bins=10, range=(0, 1), color=color, alpha=0.5, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Frequency')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(0, 50000)

# New function to create the 5-subplot figure
def plot_multiple_epsilon_distributions(dfs, titles):
    # if len(dfs) != 5 or len(titles) != 5:
    #     raise ValueError("The function requires exactly 5 dataframes and 5 titles.")
    
    # Create a 2x3 grid for subplots
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(35, 8))

    # Plot each dataframe on its respective subplot
    plot_epsilon_distribution_ax(dfs[0], titles[0], axes[0])
    plot_epsilon_distribution_ax(dfs[1], titles[1], axes[1])
    plot_epsilon_distribution_ax(dfs[2], titles[2], axes[2])
    plot_epsilon_distribution_ax(dfs[3], titles[3], axes[3])
    plot_epsilon_distribution_ax(dfs[4], titles[4], axes[4])
    plot_epsilon_distribution_ax(dfs[5], titles[5], axes[5], color='blue')


    # Turn off the last unused subplot
    # axes[1, 2].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def plot_average_cmae_ax(df: pd.DataFrame, title: str, ax, special_ranges: list = SPECIAL_RANGES):
    indices = []
    cmaes = []
    for x in range(0, 99000, 100):

        sample = df[(df["filenr"] > x) & (df["filenr"] < x+100)]
        cmae = sample["cmae"].mean()
        cmaes.append(cmae)
        indices.append(x)
    
    # Plot the average CMAE, mark in green special ranges, and mark in red ranges with CMAE > threshold
    ax.plot(indices, cmaes)
    ax.set_title(title)
    ax.set_xlabel('File number')
    ax.set_ylabel('Average CMAE')
    for special_range in special_ranges:
        ax.axvline(x=special_range, color='r', linestyle='--')
    # set fixed y-axis limits from 0.0 to 0.5
    ax.set_ylim(0.0, 0.5)

def plot_multiple_average_cmae(dfs, titles):
    # if len(dfs) != 5 or len(titles) != 5:
    #     raise ValueError("The function requires exactly 5 dataframes and 5 titles.")
    
    # Create a 2x3 grid for subplots
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(35, 8))

    # Plot each dataframe on its respective subplot
    for i in range(5):
        plot_average_cmae_ax(dfs[i], titles[i], axes[i])


    # Turn off the last unused subplot
    # axes[1, 2].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def plot_subarea(df: pd.DataFrame, title: str, start: int, samples: int = 1000):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(df['filenr'][start:start+samples], df['model_epsilon'][start:start+samples])
    plt.show()


def plot_subarea_ax(df: pd.DataFrame, title: str, ax, start: int, samples: int = 1000):
    ax.plot(df['filenr'][start:start+samples], df['model_epsilon'][start:start+samples], c="r", alpha=0.5)
    ax.plot(df['filenr'][start:start+samples], df['exact_epsilon'][start:start+samples], c="b", alpha=0.5, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("File Number")
    ax.set_ylabel("Epsilon")
    # legend position
    ax.legend(["Model", "Exact Method"], loc="upper right")

def plot_multiple_subareas(dfs, titles, start: int = 0, samples: int = 250):
    # if len(dfs) != 5 or len(titles) != 5:
    #     raise ValueError("The function requires exactly 5 dataframes and 5 titles.")
    
    # Create a 2x3 grid for subplots
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(16, 24))

    # Plot each dataframe on its respective subplot
    for i in range(5):
        plot_subarea_ax(dfs[i], titles[i], axes[i], start, samples)


    # Turn off the last unused subplot
    # axes[1, 2].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

