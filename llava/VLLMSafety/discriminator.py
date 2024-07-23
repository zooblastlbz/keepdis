import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


class EasyNeuralNetwork(nn.Module):
    def __init__(self, input_size, classes):
        super().__init__()  # we can add more layers later
        # layer 1
        self.fc1 = nn.Linear(input_size, 50)
        # layer 2
        self.fc2 = nn.Linear(50, classes)

    def forward(self, x):
        # run x through the layers and activation functions
        # (relu activation function is just max(0, x))
        x = F.relu(self.fc1(x))
        # normally there's no activation function on last layer (except softmax etc. when needed)
        x = self.fc2(x)

        return x


def evaluate(model, loss_function, X, y):
    predictions = model(X)  # pass thorugh model
    loss = loss_function(predictions, y)
    predictions = predictions.argmax(dim=1).cpu().numpy()
    acc = (predictions == y.cpu().numpy()).mean()
    return predictions, acc, loss


def train(training_dataloader, IMAGE_SHAPE=1024 * 5, NUM_CLASSES=2, device='cuda', EPOCHS=10):
    model = EasyNeuralNetwork(IMAGE_SHAPE, NUM_CLASSES)
    model.to(device)  # put the model on the device (remember its cuda on workstation)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    training_acc_lst, training_loss_lst = [], []

    epochs_acc = []
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}')
        epoch_acc = []
        training_acc_checkpoint, training_loss_checkpoint = [], []
        for step, (data, labels) in enumerate(training_dataloader):
            data = data.float().unsqueeze(0)
            labels = labels.unsqueeze(0)

            data, labels = data.to(device), labels.to(device)  # Convert labels to tensor if not already

            predictions, acc, loss = evaluate(model, loss_function, data, labels)
            training_acc_checkpoint.append(acc)
            epoch_acc.append(acc)

            # loss already calculated in the evaluate() call. just append it
            training_loss_checkpoint.append(loss.item())

            # back propagation
            loss.backward()

            # gradient descent
            optimizer.step()

            # zero the gradients so they do not accumulate
            optimizer.zero_grad()

            # epoch end
        print("Accuracy: ", np.mean(epoch_acc))
        epochs_acc.append(np.mean(epoch_acc))

        # can do some optimizations here if you want early stopping, right now im not gonna implement this

        model.train(mode=False)  # exit training mode

    return training_acc_lst, training_loss_lst, model


# def test():
#     model.train(False)  # since were testing

#     test_loss = []
#     test_acc = []

#     for X,y in test_loader:
#         with torch.no_grad():
#             X, y = X.to(device), y.to(device)
#             predictions = model(X) #as above: check dimentions

#             loss = loss_function(predictions, y)
#             test_loss.append(loss.item())

#             test_acc.append((predictions.argmax(dim=1).cpu().numpy() == y.cpu().numpy()).mean())

#     print(f'Accuracy: {np.mean(test_acc):.2f}, Loss: {np.mean(test_loss):.2f}')

#     return test_acc #idc about test_loss


def preprocess_and_call_train(get_tkns):
    # set device to cpu
    device = 'mps:0' if torch.backends.mps.is_available() else 'cpu'  # if we are running this on workstation change this to cuda

    # Example data loading (assuming you have loaded im_tok and lang_tok)
    im_tok = [entry['img_tkns'] for entry in get_tkns.items()]
    lang_tok = [entry['lang_tkns'] for entry in get_tkns.items()]

    combined_tokens = [(token, torch.tensor(0)) for token in im_tok] + [(token, torch.tensor(1)) for token in lang_tok]

    # Optionally shuffle the combined list to randomize the order
    random.shuffle(combined_tokens)

    # testing code... if our embeddings are the wrong side we are doing something wrong.
    assert im_tok[0].flatten().size() == [1024*5,
                                          1], ("flattened image tokens fed to discriminator do not match the size of "
                                               "disc first layer")
    assert lang_tok[0].flatten().size() == [1024*5,
                                            1], ("flattened language tokens fed to discriminator do not match the size "
                                                 "of disc first layer")

    # train network
    training_acc_lst, training_loss_lst, model = train(combined_tokens, device=device)

    print("------final training accuracy:", training_acc_lst[-1])

    # not gonna do any eval for now
    # test_acc = test()

    # save the model
    # PATH = 'models/desc_v1_llava.pth'
    # torch.save(model, PATH)

    return model
