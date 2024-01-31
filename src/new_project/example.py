from ann.ann_classifier import ANNClassifier
from ann.ann_functions import (
    relu,
    relu_derivative,
    softmax,
    cross_entropy_loss_derivative,
    cross_entropy_loss
)


from torch.utils.data import DataLoader
from torchvision import datasets, transforms




if __name__ == '__main__':

# Define the mini-batch size
batch_size = 1000

    # Download the dataset and create the dataloaders
    mnist_train = datasets.MNIST("../artifacts/", train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=False)
    images, labels = next(iter(train_loader))

    mnist_test = datasets.MNIST("../artifacts/", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


    layer_sizes = [784, 100, 10]
    activations = [relu, relu, softmax]
    derivated_activations = [relu_derivative, relu_derivative, cross_entropy_loss_derivative]
    loss_function = cross_entropy_loss
    learning_rate = 0.01

    classifier = ANNClassifier(
        layer_sizes=layer_sizes,
        activations=activations,
        loss_function=loss_function,
        learning_rate=learning_rate,
        derivated_activations=derivated_activations
    )

    classifier.train(train_dataset=mnist_train, epochs=10, percentage_train_size=0.8)


    classifier.plot_train_validation_loss()

    test_accuracy, test_loss = classifier.evaluate(test_or_validation_loader=test_loader)