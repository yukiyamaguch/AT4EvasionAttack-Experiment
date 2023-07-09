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
