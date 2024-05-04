from typing import List
import pandas as pd


def concat_dfs(
    dfs: List[pd.DataFrame], dataset_path: str, column_names: List[str], real_class: int
):
    # Creating new df from processed input
    for i in range(len(dfs)):
        if i == 0:
            new_df = pd.DataFrame(dfs[i], columns=column_names)
        else:
            new_df = pd.concat([new_df, dfs[i]], axis=0)
    new_df["label"] = real_class

    # Créer dataset enrichie (csv)
    original_df = pd.read_csv(dataset_path)

    concatened_dataset = pd.concat([original_df, new_df], axis=0)

    concatened_dataset.to_csv(dataset_path, index=False)
