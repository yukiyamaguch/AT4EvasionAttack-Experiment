# Road Sign Classification Experiment

## System
- - -

## Adversarial Example Matrix
- - -
We show the adversarial example matrix in this experiment below.
| Method |Perturbation Visibility|Perturbation Scope|Attack Computation|Attacker's Knowledge  |
|:-------|:---------------------:|:----------------:|:----------------:|:--------------------:|
| FGSM   | Digital               | Individual       | 1-Step           | WhiteBox/Proxy       |
| RP2    | Physical              | Individual       | Iterative        | WhiteBox/Proxy       |
| UAP    | Digital               | Universal        | Iterative        | WhiteBox/Proxy       |
| AdvGAN | Digital               | Individual       | 1-Step           | WhiteBox/Proxy/Query |

## Evasion Attack Scenario
- - -
We show the evasion attack matrix in this experiment below.
| Scenario | PV | PS | AC | AK |Conventional Attack|Available Methods|
|:---------|:--:|:--:|:--:|:--:|:------------------|:----------------|
| Man In The Middle (MITM) - RealTime(*W) | Digital   | Individual | 1-Step    | WhiteBox |Full Model Access|AdvGAN|
| | | | | | Inject AE|FGSM|
| MITM - RealTime(P)                     | Physical  | Individual | 1-Step    | Proxy    |Reconnaissance|AdvGAN|
| | | | | | Model Query Access|FGSM|
| | | | | | Collect Public Dataset||
| | | | | | Confirm Model Output||
| | | | | | Inject AE||
| MITM - RealTime(Q)                     | Digital   | Individual | 1-Step    | Query    |Model Query Access|AdvGAN|
| | | | | | Inject AE||
| MITM - Advance(W)                      | Digital   | Universal  | Iterative | WhiteBox |Model Full Access|UAP|
| | | | | | Inject AE||
| MITM - Advance(P)                      | Digital   | Universal  | Iterative | Proxy    |Reconnaissance|UAP|
| | | | | | Model Query Access||
| | | | | | Collect Public Dataset||
| | | | | | Confirm Model Output||
| | | | | | Inject AE||
| Sticker Attack(W)                      | Physical  | Individual | Iterative | WhiteBox |Model Full Access|RP2|
| | | | | | Put Stickers||
| Sticker Attack(P)                      | Physical  | Individual | Iterative | Proxy    |Reconnaissance|RP2|
| | | | | | Model Query Access||
| | | | | | Collect Public Dataset||
| | | | | | Confirm Model Output||
| | | | | | Inject AE||
| | | | | | Put Stickers||


## AT4EA Calculation
- - -
We show the AT4EA Calculation setting below.
### Experimental Error Rates and Queries Setting in AEM Nodes.
We show the AEM node parameters in this experiment.
| AEM node        |error rate|query|freq. value|
|:----------------|:--------:|:---:|:---------:|
|FGSM(MIIM-R(W))  |0.675     |0    |0.9        |
|AdvGAN(MIIM-R(W))|1.0       |0    |0.7        |
|FGSM(MIIM-R(P))  |0.25      |0    |0.9        |
|AdvGAN(MIIM-R(P))|0.174     |0    |0.7        |
|AdvGAN(MIIM-R(Q))|0.375     |27446|0.7        |
|UAP(MIIM-A(W))   |0.9       |0    |0.9        |
|UAP(MIIM-A(P))   |0.525     |0    |0.9        |
|RP2(SA(W))       |1.0       |0    |0.7        |
|RP2(SA(P))       |0.5       |0    |0.7        |

### Attack Probabilities and Query Setting in CA Nodes.
We show the CA node parameters in this experiment.
| CA node              |prob|query|
|:---------------------|:--:|:---:|
|Model Full Access     |0.01|0    |
|InjectAE              |0.01|0    |
|Reconnaissance        |0.09|0    |
|Model Query Access    |0.05|0    |
|Collect Public Dataset|0.09|0    |
|Confirm Model Output  |0.09|215  |
|Put Stickers          |0.08|0    |

### Edge weights of Root and AEA Nodes.
We show the edge weights of Root and AEA node in this experiment.
| Root node              | w1 | w2 |
|:-----------------------|:--:|:--:|
|Misclassify Stop Sign   |0.3 |0.7 |

| AEA node               | w1 | w2 | w3 |
|:-----------------------|:--:|:--:|:--:|
|[MIIM-R-W]  Digital     |0.5 |0.5 |    |
|            Individual  |1.0 |    |    |
|            1-Step      |0.1 |0.3 |0.6 |
|            WhiteBox    |1.0 |    |    |
|[MIIM-R-P]  Proxy       |1.0 |    |    |
|[MIIM-R-Q]  Query       |1.0 |    |    |
|[MIIM-A-W]  Universal   |    |1.0 |    |
|            Iterative   |0.1 |0.9 |    |
|            WhiteBox    |1.0 |    |    |
|[MIIM-A-P]  Proxy       |1.0 |    |    |
|[SA-W]      Physical    |1.0 |    |    |
|            Individual  |    |1.0 |    |
|            Iterative   |0.1 |0.9 |    |
|            WhiteBox    |1.0 |    |    |
|[SA-P]      Proxy       |1.0 |    |    |


### Results of AT4EA Calculation.
We show the attack probability and minimum query for evasion attack below.

|Attack Probability|$5.39 \times 10^{-5}$|
|Minimum Query|$215$|
