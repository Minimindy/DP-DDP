from torchvision import datasets, transforms

train_dataset = datasets.MNIST(root='./data/', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())
