# Testing



# def test_model():
#     trained_model.eval()
#     test_running_outputs = torch.FloatTensor().cpu()
#     test_running_labels = torch.LongTensor().cpu()
#     test_running_loss = 0.0
#     correct = 0
#     incorrect = 0

#     test_sampler = torch.utils.data.RandomSampler(test_set)
#     test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
#                                               sampler=test_sampler, num_workers=4)

#     with torch.no_grad():
#         for data, labels in test_loader:
#             data, labels = data.to(device), labels.to(device)
#             output = trained_model(data)

#             test_running_outputs = torch.cat((test_running_outputs, output.cpu().detach()), 0)
#             test_running_labels = torch.cat((test_running_labels, labels.cpu().detach()), 0)

#             test_loss = criterion(output, labels)
#             test_running_loss += loss.item()

#         # Accuracy
#         for idx, emb in enumerate(running_outputs.to(device)):
#             pairwise = torch.nn.PairwiseDistance(p=2).to(device)
#             dist = pairwise(emb, running_outputs.to(device))
#             closest = torch.topk(dist, 2, largest=False).indices[1]
#             if running_labels[idx] == running_labels[closest]:
#                 correct += 1
#             else:
#                 incorrect += 1

#         map_features(test_running_outputs, test_running_labels, "test_outfile")
#         print("correct", correct)
#         print("incorrect", incorrect)