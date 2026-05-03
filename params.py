import attr


@attr.s(auto_attribs = True)
class Params:

    train_dir: str = "train"
    validate_dir: str = "valid"
    test_dir: str = "test"
    model_dir: str = "model_log"

    # make sure to change this to your own path
    # pretrain_model_path: str = ""

    # model hyperparameters
    window_size: int = 120
    label_threshold: int = 120
    pinch_threshold: int = 120
    stride: int = 1
    num_to_ignore = 3
    start_index = 0
    early_stopping_threshold = 85
    selected_channels = (2, 1)

    # training hyperparameters
    lr: float = 0.0001
    batch_size: int = 8
    num_workers: int = 4
    num_epoch: int = 200