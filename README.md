# BNS-LQTY
Blockchain Network Study on LQTY

## How to use

1. Create a [conda](https://docs.conda.io/en/latest/) environment with Python>3.7

```bash
conda create --name bns python=3.8
conda activate bns
```

2. Install required packages

```bash
pip install -r requirements.txt
cd Main
```

3. Query token transaction records via [Kaggle Integration of BigQuery](https://www.kaggle.com/datasets/bigquery/ethereum-blockchain)

- Run this notebook: https://www.kaggle.com/code/bruceyufanzhang/blockchain
- Download and put the queried CSV file under `./Data/`
- **Note**: You must use Kaggle to run the notebook. It won't work otherwise.

4. Construct network features

```bash
nohup python 01_Network_Features.py --token-name LQTY >> ./logs/LQTY.txt
nohup python 01_Network_Features.py --token-name LUSD >> ./logs/LUSD.txt
```

- **Note**: If you are using ssh, you might need to use `nohup` instead of `python` to run the Python since it might takes **hours** for the [core-periphery test](https://github.com/skojaku/core-periphery-detection).
- The output data will be saved to `./Data/{token_name}_{data_collected_date}/`

5. Register a [infura project](https://infura.io/) for Ethereum API use, and copy the ENDPOINTS

- The infura ENDPOINTS will be used to detect whether an address is CA or EOA, vis [Web3.py](https://web3py.readthedocs.io/en/stable/quickstart.html)
   
6. Run the blockchian network analysis and get the visualizations

```bash
python 02_Feature_Analysis.py --token-name LQTY --infura-url <infura ENDPOINTS> --start-date 2021-04-05 >> ./logs/LQTY_analysis.txt
python 02_Feature_Analysis.py --token-name LUSD --infura-url <infura ENDPOINTS> --start-date 2021-04-15 >> ./logs/LQTY_analysis.txt
```

- **FYI**: Token genesis date (LQTY: 2021-04-05, LUSD: 2021-04-15)
- The output figures will be saved to `./Figure/{token_name}_{start_date}_{end_date}`

## Acknowledgements

Code derived and reshaped from: [BNS](https://github.com/Blockchain-Network-Studies/BNS)
