# flow_sre

## Dependency

0. find a computer and install python3

1. install `pytorch-gpu` 

2. `pip3 install -r requrements.txt`

## data preparation

`tools/ark2npz.py` convert feats.ark to feats.npz (`utt2spk` file is also required)

```bash
# convert ark to npz
python -u tools/ark2npz.py \
    --src_file data/feats.ark \
    --dest_file data/feats.npz \
    --utt2spk data/utt2spk
```

## train a model

```bash
# modify config
vim run.sh

# start to trian and infer
bash run.sh
```

## tools

Using Embedding Projector for data visualization

1. sample from original train_data and infered_data, make tsv file

```bash
# sample same index from x and z, make tsv
python -u comp_tsv_data_prepare.py \
    --pre_npz ../data/feats.npz \
    --infered_npz ../test.npz \
    --class_num 30 \
    --sample_num 300 \
    --tsv_dir ./tsv
```

2. download Embedding Projector in you onw computer and open `index.html`.

```bash
git clone https://github.com/tensorflow/embedding-projector-standalone.git
```

3. upload the tsv file to the web page

```bash
├ label.tsv
├ x_data.tsv
└ z_data.tsv
```
