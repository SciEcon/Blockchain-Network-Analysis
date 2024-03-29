# Blockchain Network Analysis

[![DOI](https://zenodo.org/badge/503755689.svg)](https://zenodo.org/badge/latestdoi/503755689)

## Project information

**The Portfolio for [Blockchain Network Analysis: A Comparative Study of Decentralized Banks](https://arxiv.org/abs/2212.05632)**

- Paper accepted by [SAI Computing Conference 2023](https://saiconference.com/Computing)
- by **[Yufan Zhang](https://yufanz.xyz/), Zichao Chen, Yutong Sun, Yulin Liu\*, and [Luyao Zhang](http://scholars.duke.edu/person/luyao.zhang)\***

\* *corresponding authors*

The flow chart of the blockchain network analysis methodology

![Flow Chart](https://raw.githubusercontent.com/SciEcon/Blockchain-Network-Analysis/main/flowchart.png)
## CC-BY 4.0 Datasets on Harvard Dataverse
Yufan Zhang; Zichao Chen; Yutong Sun; Yulin Liu; Luyao Zhang, 2023, "Replication Data for: Blockchain Network Analysis: A Comparative Study of Decentralized Banks", https://doi.org/10.7910/DVN/CZSB6C, Harvard Dataverse, V2, UNF:6:0afchMq55rHG86p/ikY02g== [fileUNF]
```
@data{DVN/CZSB6C_2023,
author = {Yufan Zhang and Zichao Chen and Yutong Sun and Yulin Liu and Luyao Zhang},
publisher = {Harvard Dataverse},
title = {{Replication Data for: Blockchain Network Analysis: A Comparative Study of Decentralized Banks}},
UNF = {UNF:6:0afchMq55rHG86p/ikY02g==},
year = {2023},
version = {V2},
doi = {10.7910/DVN/CZSB6C},
url = {https://doi.org/10.7910/DVN/CZSB6C}
}

```

## Repository structure

This repository includes supplementary resources, data, and code. 

*The Structure is generated by the [ASCII Tree Generator](https://ascii-tree-generator.com/)

```
.
├── Code
│   ├── analysis.ipynb
│   ├── extract_feature.py
│   ├── query_tx.ipynb
│   └── utils
│       ├── cp_test.py
│       └── network_fea.py
├── Data
│   ├── processedData
│   │   ├── AAVE_2022-07-13
│   │   ├── COMP_2022-07-13
│   │   ├── Dai_2022-07-13
│   │   ├── LQTY_2022-07-13
│   │   └── LUSD_2022-07-13
│   └── queriedData
│       ├── AAVE_2022-07-13.csv
│       ├── COMP_2022-07-13.csv
│       ├── Dai_2022-07-13.csv
│       ├── LQTY_2022-07-13.csv
│       └── LUSD_2022-07-13.csv
├── Figure
│   ├── AAVE_2020-10-02-2022-07-12
│   ├── BoxPlots
│   ├── COMP_2020-03-06-2022-07-12
│   ├── Dai_2019-11-18-2022-07-12
│   ├── LQTY_2021-04-05-2022-07-12
│   └── LUSD_2021-04-05-2022-07-12
├── README.md
└── requirements.txt
```

### Data

| Token Name | Queried Data (need to download by yourself) | Processed Data                         |
|------------|---------------------------------------------|----------------------------------------|
| AAVE       | ./Data/queriedData/AAVE_2022-07-13.csv      | ./Data/processedData/AAVE_2022-07-13/* |
| COMP       | ./Data/queriedData/COMP_2022-07-13.csv      | ./Data/processedData/COMP_2022-07-13/* |
| Dai        | ./Data/queriedData/Dai_2022-07-13.csv       | ./Data/processedData/Dai_2022-07-13/*  |
| LQTY       | ./Data/queriedData/LQTY_2022-07-13.csv      | ./Data/processedData/LQTY_2022-07-13/* |
| LUSD       | ./Data/queriedData/LUSD_2022-07-13.csv      | ./Data/processedData/LUSD_2022-07-13/* |

### Code

| Content                                         | File                         |
|-------------------------------------------------|------------------------------|
| Query the transaction data in Kaggle kernel     | ./Code/query_tx.ipynb        |
| Extract the network features                    | ./Code/extract_feature.ipynb |
| Analysis the features and perform visualization | ./Code/analysis.py           |


## How to use

**Contract address**

| Token | Protocol                              | Contract Address                           | Start Date |
|-------|---------------------------------------|--------------------------------------------|------------|
| LUSD  | [Liquity](https://www.liquity.org/)   | 0x5f98805A4E8be255a32880FDeC7F6728C6568bA0 | 2021-04-05 |
| LQTY  | [Liquity](https://www.liquity.org/)   | 0x6DEA81C8171D0bA574754EF6F8b412F2Ed88c54D | 2021-04-05 |
| AAVE  | [Aave](https://aave.com/)             | 0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9 | 2020-10-02 |
| COMP  | [Compound](https://compound.finance/) | 0xc00e94Cb662C3520282E6f5717214004A7f26888 | 2020-03-04 |
| Dai   | [MakerDAO](https://makerdao.com/)     | 0x6B175474E89094C44Da98b954EedeAC495271d0F | 2019-11-13 |


1. Create a [conda](https://docs.conda.io/en/latest/) environment with Python>=3.8

    ```bash
    conda create --name bna python=3.8
    conda activate bna
    ```

2. Install required packages

    ```bash
    pip install -r requirements.txt
    ```

3. Query token transaction records via [Kaggle Integration of BigQuery](https://www.kaggle.com/datasets/bigquery/ethereum-blockchain)

   - Since the data is too big to commit to the GitHub, query by yourself by running this Kaggle notebook: [https://www.kaggle.com/bruceyufanzhang/query-defi-token-transaction-records](https://www.kaggle.com/bruceyufanzhang/query-defi-token-transaction-records) (The same code can be found at [./Code/query_tx.ipynb](./Code/query_tx.ipynb))
     - Download the queried CSV files and put them file under `./Data/queriedData`
     - **Note**: You must use Kaggle to run the notebook. It won't work elsewhere.
  <!-- - **[Method 2]** Download our queried data from Google Drive: [https://drive.google.com/file/d/1ROOx5Cg3PrFcxG_VgRkcBV-NVwkV81qg/view?usp=share_link](https://drive.google.com/file/d/1ROOx5Cg3PrFcxG_VgRkcBV-NVwkV81qg/view?usp=share_link) -->

4. Extract network features and the core-periphery test results

    ```bash
    cd ./Code
    nohup python extract_feature.py --token-name LQTY >> ./logs/LQTY.txt
    nohup python extract_feature.py --token-name LUSD >> ./logs/LUSD.txt
    nohup python extract_feature.py --token-name AAVE >> ./logs/AAVE.txt
    nohup python extract_feature.py --token-name COMP >> ./logs/COMP.txt
    nohup python extract_feature.py --token-name Dai >> ./logs/Dai.txt
    ```

   - **Note**: If you are using ssh, you might need to use `nohup` to run the Python since it might takes **hours** for the [core-periphery test](https://github.com/skojaku/core-periphery-detection).
   - The output data will be saved to `./Data/processedData/{token_name}_{data_collected_date}/`

5. Register a [infura project](https://infura.io/) for Ethereum API use, and remember the ENDPOINTS

   - The infura ENDPOINTS will be used to detect whether an address is CA or EOA, through [Web3.py](https://web3py.readthedocs.io/en/stable/quickstart.html)
   
6. Run `analysis.ipynb` and get the visualization results
   - The output figures will be saved to `./Figure/{token_name}_{start_date}_{end_date}`

## Acknowledgements

Code derived and reshaped from: [BNS](https://github.com/Blockchain-Network-Studies/BNS)
