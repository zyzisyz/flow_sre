# flow_sre

## Dependency and work ENV

1. install `pytorch-gpu` 

2. `pip3 install -r requrements.txt`

## Data Preparation

`tools/ark2npz.py` transfers kaldi `feats.ark` to numpy `feats.npz` (`utt2spk` file is also required)

```bash
# convert ark to npz
python -u tools/ark2npz.py \
    --src_file data/feats.ark \
    --dest_file data/feats.npz \
    --utt2spk data/utt2spk
```

When we finish data preparation process, we can get a `feats.npz` file, which contain audio Mel-features and its labels

Each frame has 72 dimension vector and two labels: `spker_label` and `utt_label`.

`spker_label` is for pytorch `HDA_Gaussion_log_likehood` training and `utt_label` is for kaldi ark data infering

What's more, `ark2npz.py` also convert the string label to int label, and the label of the first speaker is 0.

```bash
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

### Train-scheme

`class_mean` will be initialized or updated at the begining of each training epoch.

**epoch_0**: initialize `class_mean` from x space (orignal dataset)

**epoch_n**: (n>0) update `class_meean` from epoch_n-1 's `class_mean` and epoch_n z space `class_mean`, `args.u_shift` determine the `class_mean` update shift weight.

```python
self.class_mean = class_mean.to(self.device)*args.u_shift + self.class_mean*(1.0-args.u_shift)
```

`var_j` `var_0` `u_0` are fixed

### Loss function

Loss: HDA Gaussion Log-likehood (HDA, `class_meean` and `all_mean` matrix are recombination ordered by `c_dim`)

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
# check and modify shell config
vim run.sh

# run
bash run.sh
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

### numpy npz & kaldi ark

1. `tools/ark2npz.py` transfers kaldi `feats.ark` to numpy `feats.npz` (`utt2spk` file is also required)
2. `tools/npz2ark.py` transfers numpy `feats.npz` to numpy `feats.ark` 

### Tensorboard

```bash
tensorboard --logdir runs/*
```
