"""
Dataset loading utilities for Simple DDPM (Fashion-MNIST version, 3-channel, train + test)
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config

def get_transforms():
    """Get data transforms for training & testing"""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # normalize to [-1, 1]
    ])

def create_dataloader(train=True):
    """Create DataLoader for Fashion-MNIST (train or test set)"""
    dataset = datasets.FashionMNIST(
        root=config.DATA_PATH,
        train=train,
        download=True,
        transform=get_transforms()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=train,  # shuffle only for training
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    return dataloader

def test_dataset():
    """Test train & test dataloaders"""
    print("Testing dataset...")

    try:
        train_loader = create_dataloader(train=True)
        test_loader = create_dataloader(train=False)

        print(f"Train size: {len(train_loader.dataset)}")  # type: ignore
        print(f"Test size: {len(test_loader.dataset)}")    # type: ignore
        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")

        # Inspect one batch from train
        train_batch, train_labels = next(iter(train_loader))
        print(f"Train batch shape: {train_batch.shape}")  
        print(f"Train range: [{train_batch.min():.3f}, {train_batch.max():.3f}]")

        # Inspect one batch from test
        test_batch, test_labels = next(iter(test_loader))
        print(f"Test batch shape: {test_batch.shape}")
        print(f"Test range: [{test_batch.min():.3f}, {test_batch.max():.3f}]")

        print("Dataset test passed!")

    except Exception as e:
        print(f"Dataset test failed: {e}")
        raise

if __name__ == "__main__":
    test_dataset()
