if __name__ == '__main__':
    from datasets import MammogramDataset
    from config import MammogramClassifierConfig

    config = MammogramClassifierConfig(exp='build_cache')
    
    train_dataset = MammogramDataset(split='train', config=config)
    val_dataset = MammogramDataset(split='val', config=config)
    test_dataset = MammogramDataset(split='test', config=config)

    # Build cache
    train_dataset.build_cache()
    val_dataset.build_cache()
    test_dataset.build_cache()