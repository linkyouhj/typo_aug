import torch
from tqdm import tqdm, tqdm_notebook


def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def train_epoch(model, optimizer, loss_fn, scheduler, train_dataloader, test_dataloader,
                num_epochs, device, max_grad_norm):
    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.to(device)
            segment_ids = segment_ids.to(device)
            valid_length = valid_length
            label = label.to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            train_acc += calc_accuracy(out, label)
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
        model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
            token_ids = token_ids.to(device)
            segment_ids = segment_ids.to(device)
            valid_length= valid_length
            label = label.to(device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
        print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))