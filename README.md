# scHERMES
The source code and input data of scHERMES

## Requirement
- Pytorch --- 1.12.1
- Python --- 3.8.16
- Numpy --- 1.24.3
- Scipy --- 1.10.1
- Sklearn --- 1.2.2
- Munkres --- 1.1.4
- tqdm --- 4.65.0
- Matplotlib --- 3.7.1

## Usage
#### Clone this repo.
```bash
git clone [https://github.com/your-username/scHERMES.git](https://github.com/your-username/scHERMES.git)
cd scHERMES

#### Code structure
data_loader.py: loads the dataset and constructs the cell graph.
opt.py: defines parameters and hyper-parameters.
utils.py: defines the utility functions (clustering, metrics, etc.).
encoder.py: defines the AE, IGAE (Improved Graph Autoencoder), and q_distribution.
scHERMES.py: defines the architecture of the whole network (including Cross-Modal Attention, Feature Recalibration, and Dynamic Gating mechanisms).
main.py: run the model (supports both pre-training and formal clustering training).
run.py: conduct parameter analysis and ablation studies.

