import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import codecs
# from prefixspan import PrefixSpan
from weighted_prefixspan import *
from fp_growth import *

def preprocess(dataset_path, dataset_saved_path,task):

    if os.path.isdir(dataset_saved_path):
        if task == 'non-overlap':
            return pd.read_csv(os.path.join(dataset_saved_path,"processed_data.csv"))
        elif task == 'user-overlap':
            # TODO: use the implementation from C2DSR
            pass
    try:
        if 'douban' in dataset_path:
            dataset = pd.read_csv(dataset_path,sep='\t')
            max_seq_length = 200
        elif 'epinions' in dataset_path:
            dataset = pd.read_csv(dataset_path)
            dataset = dataset.rename(columns={'user': 'UserId', 'item': 'ItemId','time': 'Timestamp'})
            max_seq_length = 200
        else:
            raise BaseException(f"Can't recognize the dataset type for {dataset_path}")
    except BaseException as e:
        print(f"Error: {e}")

    item_stat = dataset['ItemId'].value_counts()
    user_stat = dataset['UserId'].value_counts()

    # filter out cold start items and users
    filtered_item = item_stat[item_stat>10].keys()
    filtered_user = user_stat[user_stat>10].keys()

    filtered_dataset = dataset[dataset['UserId'].isin(filtered_user) & dataset['ItemId'].isin(filtered_item)]
    filtered_dataset = filtered_dataset.sort_values('Timestamp').groupby('UserId').apply(lambda x:x.tail(max_seq_length)).reset_index(drop=True)

    os.mkdir(dataset_saved_path)
    filtered_dataset.to_csv(os.path.join(dataset_saved_path,"processed_data.csv"))

    return filtered_dataset


def label_split(dataset):
    labels = []
    for user_id, group in dataset.groupby('UserId'):
        if len(group) >= 3:  # Ensure there are at least 3 entries
            test_label = group.iloc[-1]  # Last one for test
            validation_label = group.iloc[-2]  # Second last for validation
            training_label = group.iloc[-3]  # Third last for training

            labels.append({
                'userId': user_id,
                'test_label': test_label['value'],
                'validation_label': validation_label['value'],
                'training_label': training_label['value']
            })
    labels = pd.DataFrame(labels)
    return labels


def popularity_stat(dataset,dataset_name):
    # Count the number of interactions per user
    user_interaction_counts = dataset['UserId'].value_counts()

    # Count the number of interactions per item
    item_interaction_counts = dataset['ItemId'].value_counts()

    print(f" {dataset_name} item number: {len(item_interaction_counts.keys())}， user number: {len(user_interaction_counts.keys())}\n")
    # Define bins for popularity levels (using your bin setup)



    bins = [0, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, np.inf]
    labels = ['0-4','5-9', '10-19', '20-49', '50-99', '100-200', '200-500', '500-1000', '1000-2000', '2000-5000', '5000+']

    # Bin the items based on their interaction count
    item_popularity_bins = pd.cut(item_interaction_counts, bins=bins, labels=labels, right=False)

    # Calculate the proportion of items in each bin
    popularity_distribution = item_popularity_bins.value_counts(normalize=True)

    # Calculate the total interactions in each bin
    interaction_totals_per_bin = dataset.groupby(
        pd.cut(dataset['ItemId'].map(item_interaction_counts), bins=bins, labels=labels, right=False),observed=False).size()

    # Calculate the percentage of interactions in each bin
    total_interactions = interaction_totals_per_bin.sum()
    interaction_percentage_per_bin = (interaction_totals_per_bin / total_interactions) * 100

    # Create the figure and the first y-axis for item proportion
    fig, ax1 = plt.subplots(figsize=(10, 7))

    # Plot the proportion of items in each popularity bin
    ax1.bar(np.arange(len(popularity_distribution)), popularity_distribution.sort_index(), color='skyblue',
            label='Proportion of Items', alpha=0.7)
    ax1.set_xlabel('Popularity Bins (Number of Interactions)')
    ax1.set_ylabel('Proportion of Items')
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_title(f'Popularity Distribution {dataset_name}')

    # Annotate the bars for proportion of items
    for i, value in enumerate(popularity_distribution.sort_index()):
        ax1.text(i, value, f'{value:.2%}', ha='center', va='bottom', fontsize=10)

    # Create the second y-axis for percentage of interactions
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(interaction_percentage_per_bin)), interaction_percentage_per_bin.sort_index(), color='green',
             marker='o', label='Percentage of Interactions')
    ax2.set_ylabel('Percentage of Interactions')

    # Annotate the points for percentage of interactions
    for i, value in enumerate(interaction_percentage_per_bin.sort_index()):
        ax2.text(i, value, f'{value:.2f}%', ha='center', va='bottom', fontsize=10, color='green')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # ax1.set_ylim(0, 0.4)
    # ax2.set_ylim(0, 32)
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def popularity_segment(dataset):

    # Count the number of interactions per item
    item_interaction_counts = dataset['ItemId'].value_counts()

    # Sort items by interaction counts in descending order
    item_interaction_counts_sorted = item_interaction_counts.sort_values(ascending=True)
    # Calculate cumulative sum of interactions
    cumulative_interactions = item_interaction_counts.cumsum()

    # Total number of interactions
    total_interactions = cumulative_interactions.iloc[-1]

    # Divide total interactions into equal parts (e.g., 10 equal bins)
    num_bins = 10
    interactions_per_bin = total_interactions / num_bins

    # Create a list to store bin assignment and a tracker for current bin
    bins = []
    current_bin = 1
    current_bin_interactions = 0

    # Assign items to bins based on keeping the number of interactions in each bin equal
    for item_id, interaction_count in item_interaction_counts_sorted.items():
        # Add the current item's interactions to the bin
        if current_bin_interactions + interaction_count > interactions_per_bin and current_bin < num_bins:
            # Move to the next bin if the current bin exceeds the threshold
            current_bin += 1
            current_bin_interactions = 0  # Reset bin interaction counter

        # Assign the current item to the current bin
        # bins.append(f'Pop_{current_bin}')
        bins.append(current_bin)
        current_bin_interactions += interaction_count

        # Create a DataFrame showing item interaction counts and their bin
    item_bins_df = pd.DataFrame({
        'ItemId': item_interaction_counts_sorted.index,
        'InteractionCounts': item_interaction_counts_sorted.values,
        'Bin': bins
    })

    # Convert item_bins_df to a dictionary {ItemId: Bin}
    item_bins_dict = item_bins_df.set_index('ItemId')['Bin'].to_dict()

    # Map the bin information back to the original dataset
    dataset['Popularity'] = dataset['ItemId'].map(item_bins_dict)

    # Calculate the percentage of interactions in each bin
    bin_percentages = item_bins_df.groupby('Bin')['InteractionCounts'].sum() / total_interactions * 100
    item_per_bin = item_bins_df.groupby('Bin')['ItemId'].nunique() 
    # Display the bin thresholds and percentages
    print(item_bins_df)
    print(bin_percentages)
    print(item_per_bin)
    return dataset


def pattern_discover(dataset, column, apply_ipw=True):

    sequences = dataset.groupby('UserId')[column].apply(list).reset_index()
    if apply_ipw:
        sequences['SequenceLength'] = sequences[column].apply(len)
        sequences['weight'] = 1.0 / sequences['SequenceLength'] + 5
    else:
        sequences['weight'] = 1

    sequences = sequences.apply(lambda row: {'sequence': row[column], 'weight': row['weight']}, axis=1).tolist()

    # Initialize PrefixSpan with the user interaction sequences
    total_weight = sum(seq['weight'] for seq in sequences)
    min_support = 0.0001 * total_weight
    print(f"min_support: {min_support}")
    # wfpg = WeightedFPGrowth(sequences, min_support=min_support)
    print("start prefixSpan...")
    wps = WeightedPrefixSpan(sequences, min_support=min_support)

    # Run and get top 10 frequent patterns with minimum length of 5
    # patterns = wfpg.run()
    patterns = wps.run(topk=20, fixed_length=5)
    # Print patterns and their weighted support
    for pattern, support in patterns:
        print(f"Pattern: {pattern}, Weighted Support: {support}")
    print("Finished prefixSpan!")

    # bins = [0, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, np.inf]
    # labels = ['0-4', '5-9', '10-19', '20-49', '50-99', '100-200', '200-500', '500-1000', '1000-2000', '2000-5000',
    #           '5000+']
    #
    # # Bin the items based on their interaction count
    # item_popularity_bins = pd.cut(item_interaction_counts, bins=bins, labels=labels, right=False)
    #
    # # Calculate the proportion of items in each bin
    # popularity_distribution = item_popularity_bins.value_counts(normalize=True)
    #
    # # Calculate the total interactions in each bin
    # interaction_totals_per_bin = dataset.groupby(
    #     pd.cut(dataset['ItemId'].map(item_interaction_counts), bins=bins, labels=labels, right=False),
    #     observed=False).size()
    #
    # # Calculate the percentage of interactions in each bin
    # total_interactions = interaction_totals_per_bin.sum()
    # interaction_percentage_per_bin = (interaction_totals_per_bin / total_interactions) * 100
    #
    # # Create the figure and the first y-axis for item proportion
    # fig, ax1 = plt.subplots(figsize=(10, 7))
    #
    # # Plot the proportion of items in each popularity bin
    # ax1.bar(np.arange(len(popularity_distribution)), popularity_distribution.sort_index(), color='skyblue',
    #         label='Proportion of Items', alpha=0.7)
    # ax1.set_xlabel('Popularity Bins (Number of Interactions)')
    # ax1.set_ylabel('Proportion of Items')
    # ax1.set_xticks(np.arange(len(labels)))
    # ax1.set_xticklabels(labels, rotation=45)
    # ax1.set_title(f'Popularity Distribution {dataset_name}')
    #
    # # Annotate the bars for proportion of items
    # for i, value in enumerate(popularity_distribution.sort_index()):
    #     ax1.text(i, value, f'{value:.2%}', ha='center', va='bottom', fontsize=10)
    #
    # # Create the second y-axis for percentage of interactions
    # ax2 = ax1.twinx()
    # ax2.plot(np.arange(len(interaction_percentage_per_bin)), interaction_percentage_per_bin.sort_index(), color='green',
    #          marker='o', label='Percentage of Interactions')
    # ax2.set_ylabel('Percentage of Interactions')
    #
    # # Annotate the points for percentage of interactions
    # for i, value in enumerate(interaction_percentage_per_bin.sort_index()):
    #     ax2.text(i, value, f'{value:.2f}%', ha='center', va='bottom', fontsize=10, color='green')
    #
    # # Add legends
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    # # ax1.set_ylim(0, 0.4)
    # # ax2.set_ylim(0, 32)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # return patterns
