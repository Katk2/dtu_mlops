import click
import torch
from model import MyAwesomeModel
from torch import nn

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    model.train()
    train_set, _ = mnist()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch = 5
    for epoch in range(epoch):
        running_loss = 0
        for images, labels in train_set:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
        
            optimizer.zero_grad()

            # Forward pass, then backward pass, then update weights
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()

        else:
            print(f"Training loss: {running_loss}")

        torch.save(model, f'checkpoint_{epoch}.pth')

    



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()
    model.eval()
    criterion = nn.NLLLoss()

    accuracies = []
    losses = []
    with torch.no_grad():
        for images, labels in test_set:
            ps = torch.exp(model(images))
            loss = criterion(ps, labels)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            accuracies.append(accuracy.item())
            losses.append(loss.item())
        meanAccuracy = torch.mean(torch.tensor(accuracies))
        print(f'Accuracy: {meanAccuracy.item()*100}%')



cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
