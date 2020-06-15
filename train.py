# Main Trainer
def train_model():
    """Generic function to train model"""

    start_time = datetime.now()
    correct = 0
    incorrect = 0
    num_batches = 0
    loss_values = []

    # Epochs
    for epoch in range(num_epochs):
        print("epoch num:", epoch)

        train_sampler = torch.utils.data.RandomSampler(train_set)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   sampler=train_sampler, num_workers=4)

        running_outputs = torch.FloatTensor().cpu()
        running_labels = torch.LongTensor().cpu()
        running_loss = 0.0
        model.train()

        # Batches
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            num_batches += 1
            optimizer.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)

            running_outputs = torch.cat((running_outputs, output.cpu().detach()), 0)
            running_labels = torch.cat((running_labels, labels.cpu().detach()), 0)

            loss = criterion(output, labels)
            loss = Variable(loss, requires_grad=True)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Accuracy
        for idx, emb in enumerate(running_outputs.to(device)):
            pairwise = torch.nn.PairwiseDistance(p=2).to(device)
            dist = pairwise(emb, running_outputs.to(device))
            closest = torch.topk(dist, 2, largest=False).indices[1]
            if running_labels[idx] == running_labels[closest]:
                correct += 1
            else:
                incorrect += 1

        running_outputs = torch.cat((running_outputs, output.cpu().detach()), 0)
        running_labels = torch.cat((running_labels, labels.cpu().detach()), 0)

        print(running_loss / num_batches)
        print("Correct", correct)
        print("Incorrect", incorrect)

        # t-SNE
        map_features(running_outputs, running_labels, "outfile")
        # Loss Plot
        loss_values.append(running_loss / num_batches)

        time_elapsed = datetime.now() - start_time
        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    plt.plot(loss_values)
    return model, running_loss

