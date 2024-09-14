import sys
import os
import shutil
from os import X_OK
from collections import OrderedDict
import time
import numpy as np
from numpy.lib.twodim_base import triu_indices
import pandas as pd
import matplotlib.pyplot as plt
import monai
from monai.data import DataLoader
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DeepPETModelManager:
    def __init__(self, model, odir) -> None:

        print(f"GPU available: {torch.cuda.is_available()}")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.odir = odir
        if not os.path.isdir(self.odir):
            os.mkdir(self.odir)
        self.checkpoint = os.path.join(self.odir, "model.pth")
        self.history = os.path.join(self.odir, "history.csv")

        self.optimizer = None
        self.current_epoch = None

        pass

    def create_loader(self, ds, batch_size, shuffle=True):

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=monai.data.utils.pad_list_data_collate,
            persistent_workers=False,
        )

        return loader

    def load_checkpoint(self):

        print(f"loading checkpoint from {self.load_checkpoint}")

        if not torch.cuda.is_available():
            model_checkpoint = torch.load(
                self.checkpoint, map_location=torch.device("cpu")
            )
        else:
            model_checkpoint = torch.load(self.checkpoint)

        self.model.load_state_dict(model_checkpoint["model"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(model_checkpoint["optimizer"])
        self.current_epoch = model_checkpoint["epoch"]

        return self.model

    def train_model(self, train_ds, val_ds, loss_function, num_epochs, optimizer, batch_size):

        self.loss_function = loss_function
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.batch_size = batch_size
        print(f"batch_size: {self.batch_size}")

        # scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=10, factor=0.5)

        train_steps = np.floor(len(train_ds) / self.batch_size)
        val_steps = np.floor(len(val_ds) / self.batch_size)
        train_loader = self.create_loader(train_ds, batch_size=self.batch_size)
        val_loader = self.create_loader(val_ds, batch_size=self.batch_size)

        epoch_train_loss_values = list()
        epoch_train_accuracy_values = list()
        epoch_val_loss_values = list()
        epoch_val_accuracy_values = list()
        for epoch in range(self.num_epochs + 1):

            start_time = time.time()
            print("-" * 10)
            print(f"epoch {epoch}/{self.num_epochs}")
            epoch_loss = 0
            epoch_accuracy = 0

            # initialize train and valdiation and iterator 
            train_loader_iter = iter(train_loader)
            val_loader_iter = iter(val_loader)

            self.model.train()
            for step in np.arange(train_steps):
                batch_data = next(train_loader_iter)
                inputs, labels = batch_data["img"].to(self.device), batch_data["label"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs.float(), labels.unsqueeze(1).float())
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                print(f"{int(step+1)}/{int(train_steps)} step train loss: {loss.item():.4f}")

                # positivity threshold = 0 due to lack of final sigmoid layer
                pred = outputs.float() > 0 
                epoch_accuracy += (pred == labels.unsqueeze(1).float()).sum().item() / labels.shape[0]

            epoch_accuracy /= train_steps
            epoch_loss /= train_steps
            epoch_train_loss_values.append(epoch_loss)
            epoch_train_accuracy_values.append(epoch_accuracy)
            print(f"\n{epoch}/{self.num_epochs}, train loss: {epoch_loss:.4f}, train accuracy: {epoch_accuracy:.4f}\n")
            print(f"epoch time elapsed: {time.time() - start_time}")

            self.model.eval()
            with torch.no_grad():
                epoch_val_loss = 0
                epoch_val_accuracy = 0
                
                for step in np.arange(val_steps):
                    val_data = next(val_loader_iter)
                    val_inputs, val_labels = val_data["img"].to(self.device), val_data["label"].to(self.device)
                    val_outputs = self.model(val_inputs)
                    val_loss = self.loss_function(val_outputs.float(), val_labels.unsqueeze(1).float())
                    epoch_val_loss += val_loss.item()

                    val_pred = val_outputs.float() > 0
                    epoch_val_accuracy += (val_pred == val_labels.unsqueeze(1).float()).sum().item() / val_labels.shape[0] 

                epoch_val_accuracy /= val_steps
                epoch_val_loss /= val_steps
                print(f"\n{epoch}/{self.num_epochs}, validation loss: {epoch_val_loss:.4f}, validation accuracy: {epoch_val_accuracy:.4f}\n")
                epoch_val_loss_values.append(epoch_val_loss)
                epoch_val_accuracy_values.append(epoch_val_accuracy)

                # scheduler.step(epoch_val_loss)

            # save models
            torch.save({"epoch": epoch + 1, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, self.checkpoint)

            # save training log
            log_dict = {
                "train_loss": epoch_train_loss_values,
                "train_accuracy": epoch_train_accuracy_values,
                "val_loss": epoch_val_loss_values,
                "val_accuracy": epoch_val_accuracy_values,
            }

            history_df = pd.DataFrame(log_dict)
            history_df.to_csv(self.history, index=False)

            # early stopping
            epoch_val_lss_values_arr = np.array(epoch_val_loss_values)
            min_idx = np.where(epoch_val_lss_values_arr == epoch_val_lss_values_arr.min())[0][0]
            if epoch >= 25 and min_idx < len(epoch_val_lss_values_arr) - 25:
                return self.model

        return self.model

    def plot_history(self):

        hist_df = pd.read_csv(self.history)

        plt.figure()
        plt.plot(range(len(hist_df)), hist_df["train_loss"], label="training")
        plt.plot(range(len(hist_df)), hist_df["val_loss"], label="validation")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(os.path.join(self.odir, "loss.png"), dpi=300)
        plt.close()

        plt.figure()
        plt.plot(range(len(hist_df)), hist_df["train_accuracy"], label="training")
        plt.plot(range(len(hist_df)), hist_df["val_accuracy"], label="validation")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig(os.path.join(self.odir, "accuracy.png"), dpi=300)
        plt.close()

        return None

    def predict(self, test_ds):

        self.load_checkpoint()
        self.model.eval()
        test_loader = self.create_loader(test_ds, batch_size=1, shuffle=False)

        outputs = np.empty((len(test_ds),))
        outputs[:] = np.NaN

        try:
            for i, test_batch in enumerate(test_loader):
                try:
                    test_img = test_batch["img"].to(self.device)
                    test_outputs = self.model(test_img)
                    test_outputs = test_outputs.flatten().detach()
                    outputs[i] = test_outputs.flatten().detach()
                    print(f"making predictions: {i+1}/{len(test_ds)} –  {outputs[i]}")

                except:
                    print(f"making predictions: {i+1}/{len(test_ds)} –  failed")
            return outputs
        except:
            return outputs

    def generate_saliency_maps(self, img):

        self.model.eval()
        self.model.zero_grad()

        img = img.to(self.device)
        print(f"input image shape: {img.shape}")

        img = Variable(img, requires_grad=True)
        suvr = self.model(img)
        suvr.backward()

        gradients = self.model.get_input_gradient()

        return gradients

    def generate_smooth_grad(self, img, n=50):

        smooth_grad = np.zeros(img.shape)
        img_np = img.detach().cpu().numpy()
        stdev = 0.10 * (np.max(img_np) - np.min(img_np))
        print(stdev)

        for i in np.arange(n):

            print(img.shape)
            img_w_noise = img + np.random.normal(0, stdev, img.shape).astype(np.float32)
            print(img_w_noise.shape)
            grad = self.generate_saliency_maps(img_w_noise)
            print(grad.shape)
            smooth_grad += grad
            print(smooth_grad.shape)

        smooth_grad = smooth_grad / n

        return smooth_grad

    def generate_grad_cam(self, img):
        """
        compute GradCAM for ONLY ONE image at different depths of the model
        """

        self.load_checkpoint()

        self.model.eval()
        self.model.zero_grad()
        img = img.to(self.device)
        print(f"input image shape: {img.shape}")
        output = self.model(img)
        output.backward()

        gradient_maps = self.model.get_activation_gradients()
        print(f"dimension of gradient maps: {gradient_maps.shape}")
        activation_maps = self.model.get_activation_maps(img)
        print(f"dimension of activation maps: {activation_maps.shape}")
        grad_cam = self.compute_gradcam(activation_maps, gradient_maps)
        print(f"dimension of GradCAM before resizing: {grad_cam.shape}")

        return activation_maps, gradient_maps, grad_cam

    def compute_gradcam(self, activation_maps, gradient_maps):

        averaged_gradients = torch.mean(gradient_maps, dim=[0, 2, 3, 4])
        # print(averaged_gradients.shape)
        for i in range(activation_maps.shape[1]):
            activation_maps[:, i, :, :, :] *= averaged_gradients[i]
        grad_cam = torch.mean(activation_maps, dim=1).squeeze().detach().cpu()
        # ReLU
        grad_cam = np.maximum(grad_cam, 0)
        grad_cam /= torch.max(grad_cam)

        return grad_cam
    
    def fetch_model(self):

        self.load_checkpoint()
        self.model.eval()
        
        return self.model

    def summary(self, input_shape):

        # Saving the reference of the standard output
        original_stdout = sys.stdout

        with open(os.path.join(self.odir, "model.txt"), "w") as f:
            sys.stdout = f
            summary(self.model, input_shape)
            # Reset the standard output
            sys.stdout = original_stdout

        return None


