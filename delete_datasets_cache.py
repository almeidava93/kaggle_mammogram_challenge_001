if __name__ == '__main__':
    from datasets import MammogramDataset
    from config import MammogramClassifierConfig

    CURRENT_EXP = "delete_cache"

    config = MammogramClassifierConfig(exp='delete_cache', cache_data=True)
    
    train_dataset = MammogramDataset(split='train', config=config)
    val_dataset = MammogramDataset(split='val', config=config)
    test_dataset = MammogramDataset(split='test', config=config)

    # Build cache
    train_dataset.delete_cache()
    val_dataset.delete_cache()
    test_dataset.delete_cache()