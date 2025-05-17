import os
from datasets import load_dataset, Dataset, DatasetDict
from typing import Union, Optional


def fetch_dataset(
    name: str,
    config: Optional[str] = None,
    split: Optional[Union[str, dict]] = None,
    save_format: str = "csv",
    save_dir: str = "./raw",
):
    """
    Load dataset from Hugging Face Hub.
    Args:
        name (str): Dataset name
        config (str, optional): Dataset configuration
        split (str or dict or None):
            - str: Standard split or slice string (e.g., 'train[:5000]')
            - dict: Mapping split names to sample sizes (e.g., {'train': 5000, 'test': 556})
            - None: Load full dataset
        save_format (str): Save as 'csv' or 'json'
        save_dir (str): Directory to save data files
    Returns:
        Dataset or DatasetDict
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        prefix = name.replace("/", "_")
        if config:
            prefix += f"_{config}"

        if isinstance(split, dict):
            results = {}
            for split_name, size in split.items():
                dataset = load_dataset(name, config, split=split_name, streaming=True)
                dataset = dataset.take(size)
                items = list(dataset)
                dataset = Dataset.from_dict(
                    {k: [example[k] for example in items] for k in items[0].keys()}
                )
                results[split_name] = dataset
                save_path = os.path.join(
                    save_dir, f"{prefix}_{split_name}_sample.{save_format}"
                )
                if save_format == "csv":
                    dataset.to_csv(save_path)
                elif save_format == "json":
                    dataset.to_json(save_path)
                else:
                    raise ValueError("Unsupported save format. Use 'json' or 'csv'.")
                print(f"Saved: {save_path}")
            return DatasetDict(results)
        elif split is None:
            dataset = load_dataset(name, config)
        else:
            if isinstance(split, str):
                if ":" in split:
                    split_name, range_part = split.split("[", 1)
                    dataset = load_dataset(
                        name, config, split=split_name, streaming=True
                    )
                    range_part = range_part.rstrip("]")
                    if ":" in range_part:
                        start, end = range_part.split(":", 1)
                        start = int(start) if start else 0
                        end = int(end) if end else None
                        if end is not None:
                            dataset = dataset.skip(start).take(end - start)
                        else:
                            dataset = dataset.skip(start)
                    items = list(dataset)
                    if items:
                        dataset = Dataset.from_dict(
                            {
                                k: [example[k] for example in items]
                                for k in items[0].keys()
                            }
                        )
                    else:
                        dataset = Dataset.from_dict({})
                else:
                    dataset = load_dataset(name, config, split=split)
            else:
                raise ValueError("Split must be a string, dictionary, or None")

        if isinstance(dataset, Dataset):
            split_name = split if isinstance(split, str) else "dataset"
            save_path = os.path.join(save_dir, f"{prefix}_{split_name}.{save_format}")
            if save_format == "csv":
                dataset.to_csv(save_path)
            elif save_format == "json":
                dataset.to_json(save_path)
            else:
                raise ValueError("Unsupported save format. Use 'json' or 'csv'.")
            print(f"Saved: {save_path}")
        elif isinstance(dataset, DatasetDict):
            for k, subset in dataset.items():
                save_path = os.path.join(save_dir, f"{prefix}_{k}.{save_format}")
                if save_format == "csv":
                    subset.to_csv(save_path)
                elif save_format == "json":
                    subset.to_json(save_path)
                else:
                    raise ValueError("Unsupported save format. Use 'json' or 'csv'.")
                print(f"Saved: {save_path}")

        return dataset
    except Exception as e:
        raise RuntimeError(
            f"Failed to load or save dataset '{name}' (config='{config}') with split='{split}': {e}"
        )


if __name__ == "__main__":
    fetch_dataset(
        name="amazon_polarity",
        split={"train": 5000, "test": 556},
        save_format="csv",
    )
