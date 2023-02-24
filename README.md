# Attack Tree for Evasion Attack
This repository is the experimental codes of this thesis.
We made 3 image classification models in different domains and conducted evasion attacks to calculate model error ratio.
Then, we calculate the root value of attack trees using them.


## Environment
We use anaconda to make the experiment environment.
```
conda create -n name python=3.8
```
Then, you can install nesesary libraries.
```
pip install -r requirements.txt
```

## Experiments
We show the table of experiment systems and using datasets.

| taxk                      | datase                |
|:--------------------------|:---------------------:|
| Road Sign Classification  | GTSRB[1]              |
| Pneumonia Classification  | Pneumonia dataset[2]  |
| Item Classification       | Fashion MNIST[3]      | 

It is nesesary to conduct these experiments.
You can download these dataset easiliy.
The root directory has 3 directories to related to each experiment.
Each directory includes the experiment codes.


## Run
Each experiment directory has "sh" directory.
You can run the experiment to use shell codes they have.

## Reference
[1] J. Stalkamp, et al. Man vs Computer: Benchmarking machine learning algorithm for traffic sign recognition Neural network 32:323-332, 2012.

[2] D.S. Kermany, et al. Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell, 172(5):1122-1131, 2018.

[3] H. Xiao, et al. Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithm. arXiv Preprint arXiv:1708:07747, 2017.

