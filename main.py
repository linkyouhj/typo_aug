import torch
import os
import get_dataset
import train
import model


CFG = {
    "DEVICE" : torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
    "BATCH_SIZE" : 16,
    "MAX_LENGTH" : 32,
    "NUM_EPOCHS" : 5,
    "LEARNING_RATE" : 3e-6,
    "WARMUP_RATIO" : 0.1,
    "MAX_GRAD_NORM" : 1,
    "LOG_INTERVAL" : 200,
    "PATH" : os.getcwd(),
    "TRAIN_NAME" : "/ratings_train.txt",
    "TEST_NAME" : "/ratings_test.txt",
    "SEP" : "\t",
    "MODEL" : "monologg/kobert"
}

train_dataloader, test_dataloader = get_dataset.get_dataset(CFG["MODEL"],
                                                            CFG["SEP"],
                                                            CFG["PATH"],
                                                            CFG["TRAIN_NAME"],
                                                            CFG["TEST_NAME"],
                                                            CFG["MAX_LENGTH"],
                                                            CFG["BATCH_SIZE"])

classification_model, optimizer, loss_fn, scheduler = model.get_model(train_dataloader,
                                                    CFG["MODEL"],
                                                    CFG["DEVICE"],
                                                    CFG["LEARNING_RATE"],
                                                    CFG["NUM_EPOCHS"],
                                                    CFG["WARMUP_RATIO"]
                                                        )

train.train_epoch(classification_model, 
                optimizer, 
                loss_fn, 
                scheduler, 
                train_dataloader, 
                test_dataloader,
                CFG["NUM_EPOCHS"],
                CFG["DEVICE"],
                CFG["MAX_GRAD_NORM"]
                )