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

## Settings
We trained the classifiers on these datasets using the Keras framework.
The evasion attacks on these systems are shown below.
- The intentional error of stop sign recognition: the attack on human safety during the road sign recognition task.
- Wrong diagnosis versus normal diagnosis: a menace to the reliability of an ML provider in the pneumonia diagnosis task.
- The intentional misclassification of an item: the mischief to inject poisoned data into the item classification task.
In this experiment, the adversarial example matrix consists of methods with unique attributes.

The evasion attacks are identified, including all methods in the matrix.
Then, our method translates the scenarios to the AT4EA using the pattern.
We train the classifiers to calculate their error rates using adversarial examples in the three different systems.
We calculate the error rates of the classifiers to assess the risk of each system.
The test dataset consists of 50 images.
Each method generates adversarial examples with noise sizes less than the constants.
The codes of the methods consist of 11 programs from the ART library and two programs we wrote.
Furthermore, the methods in the black-box setting assign appropriate query parameters to restrict the size of the adversarial example below a certain value.
The attack probabilities of CA nodes have values between 0.1 and 0.01.
We set the weights of edges to the attacker's knowledge nodes to be white-box: 0.1, black-box (proxy): 0.3, black-box (query): 0.6 or white-box: 0.1, and black-box (proxy): 0.9.
Other weights depend on each system.
The frequency of AEM nodes packaged in the ART library is 0.9, and that of the others is 0.7.



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

