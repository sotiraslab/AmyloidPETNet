import argparse
from DeepPET.data import *
from DeepPET.architecture import *
from DeepPET.model import *

# initialize parser
parser = argparse.ArgumentParser(description='DeepPET model testing')
parser.add_argument('--odir', help='model directory')
parser.add_argument('--dataset', help='path to testing dataset')
args = parser.parse_args()

# parse arguments 
odir = str(args.odir)
print(f"model directory: {odir}")
ds_path = str(args.dataset)
print(f"path to testing dataset: {ds_path}")

# temporary file directory
cdir = "/tmp"

# initialize model and manager
model = DeepPETEncoderGradCAM()
model_manager = DeepPETModelManager(model=model, odir=odir)

try:
    # predict
    test_df = pd.read_csv(ds_path)
    # QC check
    test_gen = DeepPETDataGenerator(
        fpaths=test_df["img_path"].values.flatten(),
    )
    test_ds = test_gen.create_dataset(cache_dir=cdir, mode="prediction")

    outputs = model_manager.predict(test_ds=test_ds)
    test_df["y_score"] = outputs
    test_df.to_csv(os.path.join(odir, os.path.basename(ds_path)), index=False)

finally:
    # clear cache 
    pt_files = os.listdir(cdir)
    filtered_files = [file for file in pt_files if file.endswith(".pt")]
    print(f"removing: {filtered_files}")
    for file in filtered_files:
        path_to_file = os.path.join(cdir, file)
        os.remove(path_to_file)
    print(f"clean-up complete")
