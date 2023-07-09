# Item Classification System Experiment

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
| Send AE by API (SAEAPI) - (W)          | Digital   | Individual | Iterative | WhiteBox |Full Model Access|BIM|
| | | | | | Get Account|CW|
| | | | | | |DeepFool|
| | | | | | |PGD|
| SAEAPI - (*P)                           | Digital   | Individual | Iterative | Proxy    |Reconnaissance|BIM|
| | | | | | Get Account|CW|
| | | | | | Model Query Access|DeepFool|
| | | | | | Collect Public Dataset|PGD|
| | | | | | Confirm Model Output||
| SAEAPI - (*Q)                           | Digital   | Individual | Iterative | Query    |Get Account|Boundary Attack|
| | | | | | Model Query Access|GeoDA|
| | | | | | |HopSkipJump Attack|
| | | | | | |SimpleBA|
| Send AE as Stickers (SAES) - (W)       | Physical  | Individual | Iterative | WhiteBox |Model Full Access|Adversarial Patch|
| | | | | | Get Account||
| SAES - (P)                             | Physical  | Individual | Iterative | Proxy    |Reconnaissance|Adversarial Patch|
| | | | | | Get Account||
| | | | | | Model Query Access||
| | | | | | Collect Public Dataset||
| | | | | | Confirm Model Output||


## AT4EA Calculation
- - -
We show the AT4EA Calculation setting below.
### Experimental Error Rates and Queries Setting in AEM Nodes.
We show the AEM node parameters in this experiment.
| AEM node                    |error rate|query|freq. value|
|:----------------------------|:--------:|:---:|:---------:|
|BIM(SAEAPI(W))               |0.26      |0    |0.9        |
|CW(SAEAPI(W))                |0.32      |0    |0.9        |
|DeepFool(SAEAPI(W))          |0.22      |0    |0.9        |
|PGD(SAEAPI(W))               |0.52      |0    |0.9        |
|BIM(SAEAPI(P))               |0.36      |0    |0.9        |
|CW(SAEAPI(P))                |0.22      |0    |0.9        |
|DeepFool(SAEAPI(P))          |0.08      |0    |0.9        |
|PGD(SAEAPI(P))               |0.78      |0    |0.9        |
|Boundary Attack(SAEAPI(Q))   |0.6       |1000 |0.9        |
|GeoDA(SAEAPI(Q))             |0.4       |300  |0.9        |
|HopSkipJump Attack(SAEAPI(Q))|0.32      |500  |0.9        |
|SimpleBA(SAEAPI(Q))          |0.56      |100  |0.9        |
|Adversarial Patch(SAES(W))   |0.16      |0    |0.9        |
|Adversarial Patch(SAES(P))   |0.14      |0    |0.9        |

### Attack Probabilities and Query Setting in CA Nodes.
We show the CA node parameters in this experiment.
| CA node              |prob|query|
|:---------------------|:--:|:---:|
|Model Full Access     |0.01|0    |
|Get Account           |0.09|0    |
|Reconnaissance        |0.05|0    |
|Model Query Access    |0.07|0    |
|Collect Public Dataset|0.08|0    |
|Confirm Model Output  |0.08|50   |

### Edge weights of Root and AEA Nodes.
We show the edge weights of Root and AEA node in this experiment.
| Root node              | w1 | w2 |
|:-----------------------|:--:|:--:|
|Fool Item Classification|0.5 |0.5 |

| AEA node               | w1 | w2 | w3 |
|:-----------------------|:--:|:--:|:--:|
|[SAEAPI-W]  Digital     |1.0 |    |    |
|            Individual  |    |1.0 |    |
|            Iterative   |0.1 |0.3 |0.6 |
|            WhiteBox    |1.0 |    |    |
|[SAEAPI-P]  Proxy       |1.0 |    |    |
|[SAEAPI-Q]  Query       |1.0 |    |    |
|            WhiteBox    |1.0 |    |    |
|[SAES-W]    Physical    |1.0 |    |    |
|            Individual  |    |1.0 |    |
|            Iterative   |0.1 |0.9 |    |
|            WhiteBox    |1.0 |    |    |
|[SAES-P]    Proxy       |1.0 |    |    |


### Results of AT4EA Calculation.
We show the attack probability and minimum query for evasion attack below.

|Calculation | Value|
| :---  | :---: |
|Attack Probability|$1.05 \times 10^{-3}$|
|Minimum Query|$50$|


### Mitigation Trade-off.
We show the list of mitigation for the attack.
- Adversarial Training (AT)
- Difficult Proxy training (DP)
- Complex Query access (CQ)
- Query Restriction (QR)

We calculated the error rates against AT, DP, and AT+DP.
| AEM node                    |AT        |DP   |AT + DP    |
|:----------------------------|:--------:|:---:|:---------:|
|BIM(SAEAPI(W))               |0.0       |0.26 |0.0        |
|CW(SAEAPI(W))                |0.0       |0.32 |0.0        |
|DeepFool(SAEAPI(W))          |0.08      |0.22 |0.08       |
|PGD(SAEAPI(W))               |0.08      |0.52 |0.08       |
|BIM(SAEAPI(P))               |0.0       |0.34 |0.0        |
|CW(SAEAPI(P))                |0.0       |0.02 |0.0        |
|DeepFool(SAEAPI(P))          |0.0       |0.04 |0.0        |
|PGD(SAEAPI(P))               |0.02      |0.34 |0.06       |
|Boundary Attack(SAEAPI(Q))   |0.92      |0.6  |0.92       |
|GeoDA(SAEAPI(Q))             |0.18      |0.4  |0.18       |
|HopSkipJump Attack(SAEAPI(Q))|0.08      |0.32 |0.08       |
|SimpleBA(SAEAPI(Q))          |0.12      |0.56 |0.12       |
|Adversarial Patch(SAES(W))   |0.02      |0.16 |0.02       |
|Adversarial Patch(SAES(P))   |0.04      |0.06 |0.04       |

Then, we show the each attack probability of mitigation and their combination.
| System | Attack Probability |
| :--- | :---: |
|Plain Item Classification System | $1.05 \times 10{-3}$ |
|AT| $1.58 \times 10{-3}$ |
|DP| $1.04 \times 10{-3}$ |
|CQ| $5.38 \times 10{-4}$ |
|QR| $2.79 \times 10{-5}$ |
|AT+DP| $1.58 \times 10{-3}$ |
|AT+CQ| $7.95 \times 10{-4}$ |
|AT+QR| $1.31 \times 10{-5}$ |
|DP+CQ| $5.38 \times 10{-4}$ |
|DP+QR| $2.77 \times 10{-5}$ |
|CQ+QR| $2.77 \times 10{-5}$ |
