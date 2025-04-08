# DCCM-MSIF
**Prediction of miRNA-disease Associations via a dual-channel contrastive model based on multi-source information fusion**

## Requirements
- python==3.8
- numpy==1.22.5
- scikit-learn==1.3.2
- pytorch==2.0.0+cu118
- tqdm==4.66.4

## File Structure

### data
- `all_association`: Contains all associations needed to construct the heterogeneous graph.
- `node`: Stores all node information for the heterogeneous graph.
- `HMDD v3.2`: Contains data required for training.


### code
- `eval.py`: The startup code of the program
- `train.py`: Train the model
- `model.py`: Structure of the model
- `utils.py`: Methods of data processing
- `embedding.py`:Generate miRNA and disease embedding features

## Usage
1. Download code and data
2. Execute `python eval.py`
