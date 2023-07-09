# Pneumonia Diagnosis Experiment

## Adversarial Example Matrix
- - -
We show the adversarial example matrix in this experiment below.
| Method |Perturbation Visibility|Perturbation Scope|Attack Computation|Attacker's Knowledge  |
|:-------|:---------------------:|:----------------:|:----------------:|:--------------------:|
| Adversarial Patch | Physical              | Individual       | Iterative           | WhiteBox/Proxy       |
| BIM               | Digital               | Individual       | Iterative           | WhiteBox/Proxy       |
| Boundary Attack   | Digital               | Individual       | Iterative           | Query                |
| CW                | Digital               | Individual       | Iterative           | WhiteBox/Proxy       |
| DeepFool          | Digital               | Individual       | Iterative           | WhiteBox/Proxy       |
| FGSM              | Digital               | Individual       | 1-Step              | WhiteBox/Proxy       |
| GeoDA             | Digital               | Individual       | Iterative           | Query                |
| HopSkipJump Attack| Digital               | Individual       | Iterative           | Query                |
| PGD               | Digital               | Individual       | Iterative           | WhiteBox/Proxy       |
| SimpleBA          | Digital               | Individual       | Iterative           | Query                |
| UAP               | Digital               | Universal        | Iterative           | WhiteBox/Proxy       |

## Evasion Attack Scenario
- - -
We show the evasion attack matrix in this experiment below.
| Scenario | PV | PS | AC | AK |Conventional Attack|Available Methods|
|:---------|:--:|:--:|:--:|:--:|:------------------|:----------------|
| Man In The Middle (MITM) - RealTime(*W) | Digital   | Individual | 1-Step    | WhiteBox |Full Model Access|FGSM|
| | | | | | Get Account||
| | | | | | Inject AE||
| MITM - RealTime(P)                     | Digital   | Individual | 1-Step    | Proxy    |Get Account|FGSM|
| | | | | | Model Query Access||
| | | | | | Collect Public Dataset||
| | | | | | Confirm Model Output||
| | | | | | Inject AE||
| Put AE into DB (PAEDB) - (*W)           | Digital   | Individual | Iterative | WhiteBox |Full Model Access|BIM|
| | | | | | Get Account|CW|
| | | | | | |DeepFool|
| | | | | | |PGD|
| PAEDB - (*P)                            | Digital   | Individual | Iterative | Proxy    |Get Account|BIM|
| | | | | | Model Query Access|CW|
| | | | | | Collect Public Dataset|DeepFool|
| | | | | | Confirm Model Output|PGD|
| MITM - Advance(W)                      | Digital   | Universal  | Iterative | WhiteBox |Model Full Access|UAP|
| | | | | | Get Account||
| | | | | | Inject AE||
| PAEDB - Advance(*W)                     | Digital   | Universal  | Iterative | WhiteBox |Model Full Access|UAP|
| | | | | | Get Account||
| MITM - Advance(P)                      | Digital   | Universal  | Iterative | Proxy    |Get Account|UAP|
| | | | | | Model Query Access||
| | | | | | Collect Public Dataset||
| | | | | | Confirm Model Output||
| | | | | | Inject AE||
| PAEDB - Advance(*P)                     | Digital   | Universal  | Iterative | Proxy    |Get Account|UAP|
| | | | | | Model Query Access||
| | | | | | Collect Public Dataset||
| | | | | | Confirm Model Output||


## AT4EA Calculation
- - -
We show the AT4EA Calculation setting below.
### Experimental Error Rates and Queries Setting in AEM Nodes.
We show the AEM node parameters in this experiment.
| AEM node         |error rate|query|freq. value|
|:-----------------|:--------:|:---:|:---------:|
|FGSM(MIIM-R(W))   |0.72      |0    |0.9        |
|FGSM(MIIM-R(P))   |0.0       |0    |0.9        |
|BIM(PAEDB(W))     |0.78      |0    |0.9        |
|CW(PAEDB(W))      |0.0       |0    |0.9        |
|DeepFool(PAEDB(W))|0.08      |0    |0.9        |
|PGD(PAEDB(W))     |0.86      |0    |0.9        |
|BIM(PAEDB(P))     |0.42      |0    |0.9        |
|CW(PAEDB(P))      |0.0       |0    |0.9        |
|DeepFool(PAEDB(P))|0.14      |0    |0.9        |
|PGD(PAEDB(P))     |0.02      |0    |0.9        |
|UAP(MIIM-A(W))    |0.0       |0    |0.9        |
|UAP(PAEDB-A(W))   |0.0       |0    |0.9        |
|UAP(MIIM-A(P))    |0.0       |0    |0.9        |
|UAP(PAEDB-A(P))   |0.0       |0    |0.9        |

### Attack Probabilities and Query Setting in CA Nodes.
We show the CA node parameters in this experiment.
| CA node              |prob|query|
|:---------------------|:--:|:---:|
|Model Full Access     |0.01|0    |
|InjectAE              |0.01|0    |
|Get Account           |0.03|0    |
|Model Query Access    |0.06|0    |
|Collect Public Dataset|0.02|0    |
|Confirm Model Output  |0.09|10   |

### Edge weights of Root and AEA Nodes.
We show the edge weights of Root and AEA node in this experiment.
| Root node              | w1 | w2 |
|:-----------------------|:--:|:--:|
|Fool Pneumonia Diagnosis|1.0 |    |

| AEA node               | w1 | w2 |
|:-----------------------|:--:|:--:|
|[MIIM-R-W]  Digital     |0.7 |0.3 |
|            Individual  |0.5 |0.5 |
|            1-Step      |0.1 |0.9 |
|            WhiteBox    |1.0 |    |
|[MIIM-R-P]  Proxy       |1.0 |    |
|[PAEDB-W]   Iterative   |0.1 |0.9 |
|            WhiteBox    |1.0 |    |
|[PAEDB-P]   Proxy       |1.0 |    |
|[MIIM-A-W]  Universal   |    |1.0 |
|[PAEDB-A-W] Iterative   |0.1 |0.9 |
|            WhiteBox    |0.3 |0.7 |
|[MIIM-A-P]              |    |    |
|[PAEDB-A-P] Proxy       |0.3 |0.7 |


### Results of AT4EA Calculation.
We show the attack probability and minimum query for evasion attack below.

|Calculation | Value|
| :---  | :---: |
|Attack Probability|$8.58 \times 10^{-6}$|
|Minimum Query|$10$|
