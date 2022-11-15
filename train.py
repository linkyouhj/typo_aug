import torch
from tqdm import tqdm, tqdm_notebook


def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def train_epoch(model, optimizer, loss_fn, scheduler, train_dataloader, test_dataloader,
                num_epochs, device, max_grad_norm,early_stopping_callback):
    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
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
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
            token_ids = token_ids.to(device)
            segment_ids = segment_ids.to(device)
            valid_length= valid_length
            label = label.to(device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
        if early_stopping_callback(test_acc / (batch_id+1), model):
            print("[Early Stopping] - at Epoch " + str(e+1))
            break
        print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
    return (e+1)

def predict(path,device,pred_dataloader,df_pred,best_epoch,augmented_size,augmented_ratio):
    best_model = torch.load(path+"model.pt")
    best_model.eval()
    test_result = []

    for batch_id, (token_ids, valid_length, segment_ids) in enumerate(tqdm(pred_dataloader)):
        token_ids = token_ids.to(device)
        segment_ids = segment_ids.to(device)
        valid_length= valid_length
        with torch.no_grad():
            predictions = best_model(token_ids, valid_length, segment_ids)
        predictions = predictions.argmax(dim=-1)
        test_result.append(predictions.cpu().numpy())

    pred_label = []
    for i in test_result:
        for label in i:
            pred_label.append(label)
    df_pred["label"] = pred_label
    df_pred.drop("tokenized",axis=1,inplace=True)
    df_pred.to_csv(path+"/predictions/pred_korean_hate"+"_epoch_"+str(best_epoch)+"__"+str(augmented_size)+"_"+str(augmented_ratio)+".csv",sep=",",index=False)
