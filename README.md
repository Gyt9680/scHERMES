# scHERMES
The source code and input data of scHERMES
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

