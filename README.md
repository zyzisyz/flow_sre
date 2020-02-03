# flow_sre

## Dependency and work ENV

1. install `pytorch-gpu` 

2. `pip3 install -r requrements.txt`

## Data Preparation

`tools/ark2npz.py` convert kaldi `feats.ark` to numpy `feats.npz` (`utt2spk` file is also required)

```bash
# convert ark to npz
python -u tools/ark2npz.py \
    --src_file data/feats.ark \
    --dest_file data/feats.npz \
    --utt2spk data/utt2spk

```

When we finish data preparation, we can get a `feats.npz` file, which contain audio Mel-features and labels(speakers)

What's more, `ark2npz.py` also convert the string label to int label, and the label of the first speaker is 0.

```
.
├── feats.ark
├── feats.scp
├── spk2utt
├── utt2num_frames
├── utt2spk
└── feats.npz (new file)
```
`data_loader.py` will load the `feats.npz` while pytorch training

## Train a model and infer data from x space to z space

### Train-scheme and Loss function

`class_mean` and `all_mean` will be updata at the begining of each training epoch.

`var_j` and `var_0` is fixed

Loss: Gaussion Log-likehood (HDA, `class_meean` and `all_mean` matrix are recombination ordered by `c_dim`)

```python
# convert data from x space to z space
z, logdet = self.model(data)

mean_j = torch.index_select(self.class_mean, 0, label)

# compute hda Gaussion log-likehood
log_det_sigma = torch.log(
		var_global+1e-15).sum(-1, keepdim=True).to(self.device)

log_probs = -0.5 * ((torch.pow((z-mean_j), 2) / (var_global+1e-15) + torch.log(
		2 * pi)).sum(-1, keepdim=True) + log_det_sigma).to(self.device)

loss = -(log_probs + logdet).mean()
```

### Run

```bash
# train a model
python -u main.py \
	   --flow maf \
	   --epochs 2 \
	   --batch_size 20000 \
	   --train_data_npz ./data/feats.npz \
	   --lr 0.001 \
	   --num_blocks 10 \
	   --num_hidden 256 \
	   --device 0 \
	   --ckpt_dir ckpt \
	   --v_c 0.1 \
	   --v_0 1.0 \
	   --c_dim 36 \
	   --ckpt_save_interval 1

# infer data from x space to z space and store infered_data to npz file
python -u main.py \
	   --eval \
	   --infer_epoch 1 \
	   --test_data_npz ./data/feats.npz \
	   --infer_data_store_path ./infered.npz
```

## tools

### Data Visualization

Using Embedding Projector for data visualization

1. sample same index(label) data from x space and z space, make tsv

```bash
# sample same index from x and z, make tsv
python -u comp_tsv_data_prepare.py \
    --pre_npz ../data/feats.npz \
    --infered_npz ../test.npz \
    --class_num 30 \
    --sample_num 300 \
    --tsv_dir ./tsv
```

2. download Embedding Projector in you onw computer

```bash
git clone https://github.com/tensorflow/embedding-projector-standalone.git
```

3. open `index.html`, upload the tsv file to the web page

```bash
├ label.tsv
├ x_data.tsv
└ z_data.tsv
```
