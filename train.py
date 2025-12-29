import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import Board2Placement
import os

batchSize = 32 # look at this many games before updating network
learningRate = 0.001
epochs = 10
trueData = "data/trueData"
modelSavePath = "board2placement.pth"

def train():
    # from PyTorch docs
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # load and put data into dataset
    data = torch.load(trueData)
    inputs = torch.stack([x[0] for x in data])
    targets = torch.stack([x[1] for x in data])
    dataset = TensorDataset(inputs, targets)
    print("data loaded")

    # split data into training and validation
    # 80/20 split
    trainingSize = int(0.8 * len(dataset))
    validationSize = len(dataset) - trainingSize
    trainingData, validationData = random_split(dataset, [trainingSize, validationSize])
    trainingLoader = DataLoader(trainingData, batch_size=batchSize, shuffle=True)
    validationLoader = DataLoader(validationData, batch_size=batchSize)
    print("training + validation data made and loaded")

    # initialize model
    model = Board2Placement().to(device)
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    # training loop
    print("training started!")
    for i in range(epochs):
        model.train()
        totalTrainingLoss = 0

        for batchInputs, batchTargets in trainingLoader:
            batchInputs, batchTargets = batchInputs.to(device), batchTargets.to(device)
            # zero gradients (reset from previous batch)
            optimizer.zero_grad()
            # forward pass (make guesses)
            guesses = model(batchInputs)
            # calculate loss
            loss = mse(guesses, batchTargets)
            # backward pass (calculate what to change)
            loss.backward()
            # update network according to previous
            optimizer.step()

            totalTrainingLoss += loss.item()
        
        # validate epoch
        avgTrainingLoss = totalTrainingLoss / len(trainingLoader)
        model.eval()
        totalValidationLoss = 0
        with torch.no_grad():
            for valInput, valTarget in validationLoader:
                valInput, valTarget = valInput.to(device), valTarget.to(device)
                guesses2 = model(valInput)
                loss2 = mse(guesses2, valTarget)
                totalValidationLoss += loss2.item()
        avgValidationLoss = totalValidationLoss / len(validationLoader)
        print(f"Epoch {i + 1} | Training Loss : {avgTrainingLoss} | Validation Loss : {avgValidationLoss}")
    
    # save model
    torch.save(model.state_dict(), modelSavePath)
    print("Training Complete + Model Saved")

train()