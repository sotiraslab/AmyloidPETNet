import argparse
from datetime import datetime
import torch
from DeepPET.data import *
from DeepPET.architecture import *
from DeepPET.model import *

# initialize parser
parser = argparse.ArgumentParser(description='DeepPET model training')
parser.add_argument('--train', help='path to training dataset')
parser.add_argument('--val', help='path to validation dataset')
parser.add_argument(
    '--lr',
    default=2.5e-5,
    help="learning rate"
    )
parser.add_argument(
    '--reg',
    default=7.5e-6, 
    help="regularizer"
    )
parser.add_argument(
    '--cdir', 
    default="/tmp",
    help='temporary directory for storing cached files'
    )
parser.add_argument(
    '--odir', 
    default=f"./log/{datetime.now().strftime('%d%m%Y%H%M%S%f')}",
    help='output directory'
    )
args = parser.parse_args()

# parse arguments 
train_path = str(args.train)
print(f"training dataset path: {train_path}")
val_path = str(args.val)
print(f"training dataset path: {val_path}")
lr = float(args.lr)
reg = float(args.reg)

# temporary file directory that you have write access to
cdir = str(args.cdir)

# set random seed
seed = 42
torch.manual_seed(seed)

# specify output directory
odir = args.odir

try:

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # create training and validation data sets
    train_gen = DeepPETDataGenerator(
        fpaths=train_df["img_path"].values.flatten(),
        labels=train_df["suvr_positivity"].values.flatten(),
    )
    val_gen = DeepPETDataGenerator(
        fpaths=val_df["img_path"].values.flatten(),
        labels=val_df["suvr_positivity"].values.flatten(),
    )

    train_ds = train_gen.create_dataset(idx=np.arange(len(train_df)), cache_dir=cdir, mode="training")
    val_ds = val_gen.create_dataset(idx=np.arange(len(val_df)), cache_dir=cdir, mode="validation")

    # initialize model and manager
    model = DeepPETEncoderGradCAM()
    model_manager = DeepPETModelManager(model=model, odir=odir)
    model_manager.summary(input_shape=(1, 120, 120, 80))

    # train model
    training_config = {
        "loss_function": torch.nn.BCEWithLogitsLoss(),
        "optimizer": torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg),
        "num_epochs": 100,
        "batch_size": 10,
    }   

    with open(f'{odir}/hyper.txt', 'w') as f:
        f.write(f'lr: {lr}, reg: {reg}')

    model_manager.train_model(train_ds=train_ds, val_ds=val_ds, **training_config)
    model_manager.plot_history()

    # predict on ADNI
    train_ds = train_gen.create_dataset(idx=np.arange(len(train_df)), cache_dir=cdir, mode="prediction")
    train_df["y_score"] = model_manager.predict(test_ds=train_ds)
    train_df.to_csv(f"{odir}/train.csv")

    val_ds = val_gen.create_dataset(idx=np.arange(len(val_df)), cache_dir=cdir, mode="prediction")
    val_df["y_score"] = model_manager.predict(test_ds=val_ds)
    val_df.to_csv(f"{odir}/val.csv")

finally:
    # clear cache
    pt_files = os.listdir(cdir)
    filtered_files = [file for file in pt_files if file.endswith(".pt")]
    print(f"removing {len(filtered_files)} files")
    for file in filtered_files:
        path_to_file = os.path.join(cdir, file)
        os.remove(path_to_file)
    print(f"removed {len(filtered_files)} files")
    print(f"clean-up complete")
