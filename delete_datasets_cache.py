if __name__ == '__main__':
    from datasets import MammogramDataset
    from config import MammogramClassifierConfig

    CURRENT_EXP = "delete_cache"

    config = MammogramClassifierConfig(
        exp=CURRENT_EXP,
        dataset_size = -1, # Max = 200
        learning_rate = 0.0001,
        dropout = 0.1,
        cnn_dropout = 0.1,
        feature_dim = 256,
        img_size = 256,
        num_img_channels = 1,
        num_img_init_features = 64,
        batch_size = 20,
        num_epochs = 100,
        )
    
    train_dataset = MammogramDataset(split='train', config=config)
    val_dataset = MammogramDataset(split='val', config=config)
    test_dataset = MammogramDataset(split='test', config=config)

    # Build cache
    train_dataset.delete_cache()
    val_dataset.delete_cache()
    test_dataset.delete_cache()