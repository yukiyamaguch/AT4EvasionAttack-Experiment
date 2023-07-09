# Attack Tree for Evasion Attack
- - -
This repository is the experimental codes of the paper: Attack Tree for Evasion Attack.
We conducted the experiments to evaluate the versatality and effectiveness of our methods.
We made 3 image classification models in different domains and conducted evasion attacks to calculate model error rates.
Then, we calculate the root value of attack trees using them.


## Environment
- - -
We use anaconda to make the experiment environment.
```
conda create -n name python=3.8
```
Then, you can install nesesary libraries.
```
pip install -r requirements.txt
```

## Experiments
- - -
We show the table of experiment systems and using datasets.

| system                    | dataset               |
|:--------------------------|:---------------------:|
| Road Sign Classification  | GTSRB[1]              |
| Pneumonia Classification  | Pneumonia dataset[2]  |
| Item Classification       | Fashion MNIST[3]      |

It is nesesary to conduct these experiments.
You can download these dataset easiliy.
The root directory has 3 directories to related to each experiment.
Each directory includes the experiment codes.
We show the experiment settings and results into each directory.


## Run
- - -
### Training model
Each experiment directory has model.py and train.py.

You can train the model of each system by train.py.

### Calculate error rates
Each experiment directory has "sh" directory.

You can run the experiment to use each code.

### Calculate attack probability by AT4EA
Each experiment directory has AT4EA yaml file and calc_attack_prob.py.

You can calculate attack probability of the Root node of AT4EA.

## Reference
- - -
[1] J. Stalkamp, et al. Man vs Computer: Benchmarking machine learning algorithm for traffic sign recognition Neural network 32:323-332, 2012.

[2] D.S. Kermany, et al. Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell, 172(5):1122-1131, 2018.

[3] H. Xiao, et al. Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithm. arXiv Preprint arXiv:1708:07747, 2017.

