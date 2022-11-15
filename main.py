import torch
import os
import get_dataset
import train
import model
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hyper Parameter, Model, Dataset path etc.')

    parser.add_argument(
        "--device", 
        type = str,
        default=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        )
    parser.add_argument(
        "--batch_size",
        type = int,
        default = 16
        )
    parser.add_argument(
        "--max_length", 
        type = int,
        default = 64
        )
    parser.add_argument(
        "--num_epochs",
        type = int,
        default = 20
        )
    parser.add_argument(
        "--learning_rate",
        type = float,
        default = 3e-6
        )
    parser.add_argument(
        "--warmup_ratio",
        type = float,
        default = 0.1
        )
    parser.add_argument(
        "--max_grad_norm",
        type = int,
        default = 5
        )
    parser.add_argument(
        "--path", 
        type = str,
        default = os.getcwd()
        )
    parser.add_argument(
        "--train_path", 
        default = "/data/train.hate.csv"
        )
    parser.add_argument(
        "--test_path",
        default = "/data/dev.hate.csv"
        )
    parser.add_argument(
        "--sep",
        type = str,
        default = ","
        )
    parser.add_argument(
        "--model", 
        type = str,
        default = "monologg/kobert"
        )
    parser.add_argument(
        "--num_classes",
        type = int,
        default = 3
        )
    parser.add_argument(
        "--sentence_column",
        type = str,
        default="comments"
        )
    parser.add_argument(
        "--early_stop_patience",
        type = int,
        default = 4)
    parser.add_argument(
        "--data_type",
        type = str,
        required = True,
        default = 'korean_hate',
        choices = ['nsmc','korean_hate']
        )
    parser.add_argument(
        "--predict",
        type = bool,
        default = False
    )
    parser.add_argument(
        "--augmented_size",
        type = int,
        default = 0
    )
    parser.add_argument(
        "--augmented_ratio",
        type = int,
        default = 0
    )
    parser.add_argument(
        "--augment_all",
        type=bool,
        default = False
    )

    args = parser.parse_args()
    path = {
        "nsmc" : {
            "train": "/data/ratings_train.txt",
            "test": "/data/ratings_test.txt"
        },
        "korean_hate" : {
            "train": "/data/train.hate.csv",
            "test": "/data/test.hate.csv"
        }
    }
    # CFG = {
    #     "DEVICE" : torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
    #     "BATCH_SIZE" : 16,
    #     "MAX_LENGTH" : 64,
    #     "NUM_EPOCHS" : 5,
    #     "LEARNING_RATE" : 3e-6,
    #     "WARMUP_RATIO" : 0.1,
    #     "MAX_GRAD_NORM" : 1,
    #     "LOG_INTERVAL" : 200,
    #     "PATH" : os.getcwd(),
    #     "TRAIN_NAME" : "/data/train.hate.csv",
    #     "TEST_NAME" : "/data/dev.hate.csv",
    #     "SEP" : ",",
    #     "MODEL" : "monologg/kobert",
    #     "NUM_CLASSES" : 3,
    #     "SENTENCE_COLUMN" : "comments",
    #     "CLASSES" : {"none": 0,
    #         "offensive": 1,
    #         "hate": 2}
    # }

    classes = {
        'korean_hate' :  {"none": 0,
                        "offensive": 1,
                        "hate": 2},
        'nsmc'  : {0 : 0,
                    1 : 1
                    }
    }

    size = [2,4,6,8]
    ratio = [0.2,0.4,0.6,0.8]
    train_names = ["" for i in range(5)]
    test_names = ["" for i in range(5)]
    ratio_names = ["" for i in range(5)]
    for s in size:
        train_names[int(s/2)] = '/data/augment_size_'+str(s)+'/train.hate_'+str(s)
        test_names[int(s/2)] = '/data/augment_size_'+str(s)+'/dev.hate_'+str(s)
    for r in ratio:
        ratio_names[int(r*5)] ='_'+str(r)+'_default.csv'

    if args.augmented_size != 0 and args.augmented_ratio != 0:
        args.train_path = train_names[args.augmented_size]+ratio_names[args.augmented_ratio]
        args.test_path = test_names[args.augmented_size]+ratio_names[args.augmented_ratio]

    # train_dataloader, test_dataloader = get_dataset.get_dataset(CFG["MODEL"],
    #                                                             CFG["SEP"],
    #                                                             CFG["PATH"],
    #                                                             CFG["TRAIN_NAME"],
    #                                                             CFG["TEST_NAME"],
    #                                                             CFG["MAX_LENGTH"],
    #                                                             CFG["BATCH_SIZE"],
    #                                                             CFG["SENTENCE_COLUMN"],
    #                                                             CFG["CLASSES"]
    #                                                             )

    def train_all(train_path, test_path, augmented_size, augmented_ratio):                    
        train_dataloader, test_dataloader = get_dataset.get_dataset(args.model,
                                                                    args.sep,
                                                                    args.path,
                                                                    train_path,
                                                                    test_path,
                                                                    args.max_length,
                                                                    args.batch_size,
                                                                    args.sentence_column,
                                                                    classes[args.data_type]
                                                                    )

        # classification_model, optimizer, loss_fn, scheduler = model.get_model(train_dataloader,
        #                                                     CFG["MODEL"],
        #                                                     CFG["DEVICE"],
        #                                                     CFG["LEARNING_RATE"],
        #                                                     CFG["NUM_EPOCHS"],
        #                                                     CFG["WARMUP_RATIO"],
        #                                                     CFG["NUM_CLASSES"]
        #                                                         )
        classification_model, optimizer, loss_fn, scheduler, early_stopping_callback = model.get_model(train_dataloader,
                                                                            args.model,
                                                                            args.device,
                                                                            args.learning_rate,
                                                                            args.num_epochs,
                                                                            args.warmup_ratio,
                                                                            args.num_classes,
                                                                            args.early_stop_patience,
                                                                            args.path
                                                                                )
        
        best_epoch = train.train_epoch(classification_model, 
                        optimizer, 
                        loss_fn, 
                        scheduler, 
                        train_dataloader, 
                        test_dataloader,
                        args.num_epochs,
                        args.device,
                        args.max_grad_norm,
                        early_stopping_callback
                        )   
                        # CFG["NUM_EPOCHS"],
                        # CFG["DEVICE"],
                        # CFG["MAX_GRAD_NORM"]
                        # )
        if args.predict == True :
            df_pred, pred_dataloader = get_dataset.get_testset(args.path, args.sep, args.max_length, args.model, args.batch_size)
            train.predict(args.path, args.device, pred_dataloader, df_pred, best_epoch,augmented_size,augmented_ratio)

    if args.augment_all == True:
        for size in range(1,5):
            for ratio in range(1,5):
                train_path = train_names[size]+ratio_names[ratio]
                test_path = test_names[size]+ratio_names[ratio]
                train_all(train_path,test_path,size*2, ratio*2)

        train_all(args.train_path, args.test_path,0,0)
    
    else :
        train_all(args.train_path, args.test_path,0,0)
