# main.py
import sys
import os
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import timm
from sklearn.model_selection import train_test_split
import uuid

from src.lr_schedulers import init_lr_scheduler
from src.optimizers import init_optimizer

from data import knifeDataset
from utils import *
from args import argument_parser, optimizer_kwargs, lr_scheduler_kwargs

# global variables
args = argument_parser().parse_args()

warnings.filterwarnings('ignore')

def evaluate(test_loader, model, start, log):
    model.eval()
    map = AverageMeter()
    with torch.no_grad():
        for i, (images, target, fnames) in enumerate(test_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = model(images)
                preds = logits.softmax(1)
            
            test_map5, test_acc1, test_acc5 = map_accuracy(preds, target)
            map.update(test_map5, images.size(0))
        
        message_test_epoch = format_test_message("test", map.avg, time_to_str(timer() - start, "sec"))
        log.write("\n")  
        log.write(message_test_epoch)
    
    return [map.avg]

def main():
    # Set the seed for reproducibility
    seed_value = args.seed
    set_seed(seed_value)
    
    test_imlist = pd.read_csv(args.test_datacsv)
    test_gen = knifeDataset(test_imlist, mode="val")
    test_loader = DataLoader(test_gen, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ## Loading the model to run
    model = timm.create_model(args.model_mode, pretrained=True, num_classes=args.n_classes)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    print("Model loaded for evaluation.")
    model.to(device)
    

    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    log = Logger()
    # log.open("logs/%s_log_train.txt")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log.open(f"logs/log_test_{timestamp}.txt")
    student_id = os.environ.get('STUDENT_ID', 'your_id')
    student_name = os.environ.get('STUDENT_NAME', 'your_name')
    log.write(f"Student ID:{student_id}\n")
    log.write(f"Student name:{student_name}\n")
    log.write(f"UUID:{uuid.uuid4()}\n")

    log.write(f"==========\nArgs:{args}\n==========")

    log.write("\n" + "-" * 25 + f" [START {timestamp}] " + "-" * 25 + "\n\n")
    log.write('         |   Test  |              |\n')
    log.write(' | Mode  |    mAP   |     Time     |\n')
    log.write('-' * 79 + '\n')

    test_metrics = evaluate(test_loader, model, timer(), log)
    print("Evaluation complete.")
    

if __name__ == "__main__":
    main()
