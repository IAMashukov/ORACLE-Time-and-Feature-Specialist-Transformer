# Written by: Inal A. Mashukov
# MIT License.



import logging
import os 


def setup_logging(dataset_name):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    results_dir = os.path.join("results", dataset_name)
    os.makedirs(results_dir, exist_ok = True)
    log_file = os.path.join(results_dir, "train.log")
    logging.basicConfig( 
        level = logging.INFO, 
        format = "%(asctime)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler(), logging.FileHandler(log_file, mode = "w")]
    )


def log_model_stats(model, device):
    logging.info("=" * 20 + " MODEL ARCHITECTURE " + "=" * 20)
    logging.info(f"\n{str(model)}\n")
    logging.info("="*20 + " PARAMETER SUMMARY " + "="*20)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Parameters:     {total_params:,}")
    logging.info(f"Trainable Parameters: {trainable_params:,}")
    logging.info("=" * 80)



def get_benchmark_indices(dataset_len, data_path: str, lookback: int, horizon: int):
    dataset_name = data_path.lower()
    if "etth" in dataset_name:
        train_raw_end = 8640
        val_raw_end = 11520
        test_raw_end = 14400
    elif "ettm" in dataset_name:
        train_raw_end = 34560
        val_raw_end = 46080
        test_raw_end = 57600
    else:
        train_raw_end = int(dataset_len * 0.7)
        val_raw_end = train_raw_end + int(dataset_len * 0.1)
        test_raw_end = dataset_len

    train_indices = list(range(0, train_raw_end - lookback - horizon + 1))
    val_indices = list(range(train_raw_end - lookback, val_raw_end - lookback - horizon + 1))
    test_indices = list(range(val_raw_end - lookback, min(test_raw_end, dataset_len) - lookback - horizon + 1))

    return train_indices, val_indices, test_indices

