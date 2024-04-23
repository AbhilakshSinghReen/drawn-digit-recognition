from datetime import datetime
from os import makedirs
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim

from .config import config, models_dir
from .dataset import data_loaders
from .model import CNN


class ModelTrainer:
    def __init__(self):
        self.training_id = "torch---" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")    

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CNN()
        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.loss_fn = nn.CrossEntropyLoss()

        self.models_dir = join(models_dir, self.training_id)
        makedirs(self.models_dir, exist_ok=False)
    
    def train_single_epoch(self):
        self.model.train()

        for batch_index, (data, target) in enumerate(data_loaders["train"]):
            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)
            
            loss = self.loss_fn(output, target)
            loss.backward()

            self.optimizer.step()

            if batch_index % config['log_batch_interval'] == 0:
                print(f"    {batch_index} / {len(data_loaders['train'])}")

    def test(self):
        self.model.eval()

        sum_test_loss = 0
        num_correct = 0

        with torch.no_grad():
            for data, target in data_loaders["test"]:
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)

                sum_test_loss += self.loss_fn(output, target).item()

                prediction = output.argmax(dim=1, keepdim=True)

                num_correct += prediction.eq(target.view_as(prediction)).sum().item()
        
        average_test_loss = sum_test_loss / len(data_loaders['test'].dataset)
        accuracy = num_correct / len(data_loaders['test'].dataset)

        print(f"    Average loss: {average_test_loss}")
        print(f"    Accuracy: {accuracy}")
    
    def save_weights(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def train(self):
        for epoch_index in range(config['training_num_epochs']):
            print(f"Epoch: {epoch_index}")
            self.train_single_epoch()

            model_save_file_path = join(self.models_dir, f"epoch-{epoch_index}.pt")
            self.save_weights(model_save_file_path)
            
            self.test()


if __name__ == "__main__":
    model_trainer = ModelTrainer()
    model_trainer.train()
