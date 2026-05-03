import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import os
import csv
import copy

# Imports for my model, dataset, and parameters
from ultralight_model import UltraLightCNN1D
from preprocessing_train import SensorDataset
from params import Params



def train (device, epoch, model, optimizer, train_dataloader, criterion):
    model.train ()

    real_epoch = epoch + 1
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm (train_dataloader)

    for _, batch in enumerate (pbar):
        resistance = batch["sensors"]

        if isinstance (resistance, list):
            resistance = torch.stack (resistance).permute (1, 0, 2).to (device)
        else:
            resistance = resistance.permute (1, 0, 2).to (device)
        resistance = resistance.float ()
        target = batch["label"].long ().to (device)

        output = model (resistance)

        loss = criterion (output, target)
        total_loss += loss.item () * target.size (0)
        pbar.set_postfix({'loss': loss.item ()})

        optimizer.zero_grad ()
        loss.backward ()
        optimizer.step ()

        total_samples += target.size(0)

    average_loss = total_loss / total_samples
    print("train loss: ", average_loss, "epoch: ", real_epoch)



def validate(device, model, validate_sensor_dataLoader, criterion):
    global accs
    model.eval ()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad ():
        for _, batch in enumerate (validate_sensor_dataLoader):
            resistance = batch["sensors"]

            if isinstance (resistance, list):
                resistance = torch.stack (resistance).permute (1, 0, 2).to (device)
            else:
                resistance = resistance.permute (1, 0, 2).to (device)
            resistance = resistance.float ()
            target = batch["label"].long ().to (device)

            output = model (resistance)

            predicted_labels = output.argmax (dim = 1)

            loss = criterion (output, target)
            total_loss += loss.item () * target.size (0)

            total_correct += (predicted_labels == target).sum ().item ()
            total_samples += target.size (0)

    average_loss = total_loss / total_samples
    overall_accuracy = 100.0 * total_correct / total_samples
    accs.append (overall_accuracy)

    return overall_accuracy, average_loss


def main ():
    global accs
    accs = []

    device = torch.device ('cuda' if torch.cuda.is_available () else 'cpu')
    print (device)
    torch.cuda.empty_cache ()

    params = Params()

    model = UltraLightCNN1D ().to (device)


    train_sensor_dataset = SensorDataset (params)
    for file_name in os.listdir (params.train_dir):
        file = os.path.join (params.train_dir, file_name)
        if os.path.isfile (file):
            train_sensor_dataset.load_file (file)
    train_sensor_dataset.parsing ()

    validate_sensor_dataset = SensorDataset (params)
    for file_name in os.listdir (params.validate_dir):
        file = os.path.join (params.validate_dir, file_name)
        if os.path.isfile (file):
            validate_sensor_dataset.load_file (file)
    validate_sensor_dataset.parsing ()


    train_sensor_dataLoader = DataLoader (train_sensor_dataset,
                                          batch_size = params.batch_size,
                                          shuffle = True,
                                          num_workers = params.num_workers
                                          )
    
    validate_sensor_dataLoader = DataLoader (validate_sensor_dataset,
                                             batch_size = params.batch_size,
                                             shuffle = False,
                                             num_workers = params.num_workers
                                             )

    optimizer = optim.Adam (model.parameters (), lr = params.lr)
    criterion = nn.CrossEntropyLoss ()
    
    best_accuracy = -1
    best_loss = 0
    best_epoch = 0

    top_five_acc_models = []

    start_time = time.time ()

    try:
        for epoch in range (params.num_epoch):
            train (device, epoch, model, optimizer, train_sensor_dataLoader, criterion)
            acc, loss = validate (device, model, validate_sensor_dataLoader, criterion)

            print("validation accuracy= {}".format(acc))

            with open ("accuracy_0405data_ws90_for_demo_0416_2ch_4out_until_85acc.csv", 'w') as file:
                writer = csv.writer (file)
                writer.writerow (accs)

            if acc >= best_accuracy:
                torch.save (model.state_dict (), 'model_log/App2_model_0405data_ws90_for_demo_0416_2ch_4out_until_85acc.pth')
                best_accuracy = acc
                best_loss = loss
                best_epoch = epoch + 1
            
            if acc >= params.early_stopping_threshold:
                deep_copied_model = copy.deepcopy (model)
                top_five_acc_models.append ((deep_copied_model, acc, epoch))
                if len (top_five_acc_models) >= 1:
                    break
        
        # top_five_acc_models.sort (key = lambda x: x[1])
        for (deep_copied_model, acc, epoch) in top_five_acc_models:
            torch.save (deep_copied_model.state_dict (), 'model_log/App2_model_{}_{}_0405data_ws90_for_demo_0416_2ch_4out_until_85acc.pth'.format (acc, epoch + 1))

        print ("Total training time = {}s".format ((time.time () - start_time)))
        print ("best accracy = {} at {} epochs with loss = {}".format (best_accuracy, best_epoch, best_loss))
        
    except KeyboardInterrupt:
        torch.save (model.state_dict(), 'model_log/App2_model_0405data_ws90_for_demo_0416_2ch_4out_until_85acc_epoch_{}.pth'.format (best_epoch))
        print ("Total training time = {}s".format ((time.time () - start_time)))
        print ("best accracy = {} at {} epochs with loss = {}".format (best_accuracy, best_epoch, best_loss))



if __name__ == "__main__":
    main ()