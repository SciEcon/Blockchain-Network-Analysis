# BNS-Liquity
Blockchain network study on two tokens (LUSD & LQTY) of [Liquity](https://www.liquity.org/)

## Contract address

| Token | Protocol                              | Contract Address                           | Start Date |
|-------|---------------------------------------|--------------------------------------------|------------|
| LUSD  | [Liquity](https://www.liquity.org/)   | 0x5f98805A4E8be255a32880FDeC7F6728C6568bA0 | 2021-04-05 |
| LQTY  | [Liquity](https://www.liquity.org/)   | 0x6DEA81C8171D0bA574754EF6F8b412F2Ed88c54D | 2021-04-05 |
| AAVE  | [Aave](https://aave.com/)             | 0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9 | 2020-10-02 |
| COMP  | [Compound](https://compound.finance/) | 0xc00e94Cb662C3520282E6f5717214004A7f26888 | 2020-03-04 |
| Dai   | [MakerDAO](https://makerdao.com/)     | 0x6B175474E89094C44Da98b954EedeAC495271d0F | 2019-11-13 |

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
- Download the queried CSV files and put them file under `./Data/`
- **Note**: You must use Kaggle to run the notebook. It won't work otherwise.

4. Construct network features

```bash
nohup python 01_Network_Features.py --token-name LQTY >> ./logs/LQTY.txt
nohup python 01_Network_Features.py --token-name LUSD >> ./logs/LUSD.txt
```

- **Note**: If you are using ssh, you might need to use `nohup` to run the Python since it might takes **hours** for the [core-periphery test](https://github.com/skojaku/core-periphery-detection).
- The output data will be saved to `./Data/{token_name}_{data_collected_date}/`

5. Register a [infura project](https://infura.io/) for Ethereum API use, and remember the ENDPOINTS

- The infura ENDPOINTS will be used to detect whether an address is CA or EOA, through [Web3.py](https://web3py.readthedocs.io/en/stable/quickstart.html)
   
6. Run the blockchian network analysis and get the visualizations

```bash
python 02_Feature_Analysis.py --token-name LQTY --infura-url <infura ENDPOINTS> --start-date 2021-04-05 >> ./logs/LQTY_analysis.txt
python 02_Feature_Analysis.py --token-name LUSD --infura-url <infura ENDPOINTS> --start-date 2021-04-15 >> ./logs/LQTY_analysis.txt
```

- **FYI**: Token genesis date (LQTY: 2021-04-05, LUSD: 2021-04-15)
- The output figures will be saved to `./Figure/{token_name}_{start_date}_{end_date}`

## Acknowledgements

Code derived and reshaped from: [BNS](https://github.com/Blockchain-Network-Studies/BNS)
