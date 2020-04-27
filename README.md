In this assignment you will implement a YOLO-like object detector on the PASCAL VOC dataset. You will mainly focus on imlementing the yolo loss function and experiment with different base architectures and hyperparameters. Starter kit is provided to help you guide though the steps. 

**PS : Refer to lecture slides for the exact loss function (YOLO V1) to implement.**

# General Instructions
1. **Bonus Assignment**: This assignment is not cumpolsary and its aminly for those who want to improve their grade. That said, I **HIGHLY** encourage for everyone to do. I myself learnt a lot from it. 

2. **Group Assignment**: This can be done in groups of at most 3. All members of a group will be graded equally.

3. **Two stage submission** : As mentioned in class, submission would be done in two stages. After the end of first stage (due 11:59PM,May 4), you need to submit your best perfroming MAP scores. We will release, the scores of the top three performers after which you will have another week (Stage 2, due 11:59PM, May 11) to improve your model performance. Both stages will have indivdual scores.

**Important Note:** You are only allowed to use your late days for Stage 1 and NOT for stage 2.
# Data Setup

Once you have cloned this folder, execute the download_data script provided:
```
./download_data.sh
```
You will be using the train+val split (`voc2007.txt`) as the training dataset and the test dataset (`voc2007test.txt`) as your validation or local test dataset. The actual test dataset would be released shortly.

Instructions are provided in the `yolo_loss.py` file to help you guide through this assignment. Once you have implemented the loss function, feel free to change other parts of the network to boost the performance of your model.





# Training and Testing 
First and foremost, install all required packages by running the comand:

```
pip install -r requirements.txt
```
Once installed and you have implemented the loss function, you can train using the command:

```
python main.py --name="exp_name" --B=2 --S=14 --learning-rate=0.001 --num-epochs=50 --batch-size=24 --lambda-coord=5 --lambda-noobj=0.5

```

where


| Arguments        | Description |
| :------------- |:----------|
| `--name`     | Experiment name |
| `--B`     | Number of bounding box predictions per cell |
| `--S`     | Width/height of network output grid |
| `--learning-rate`     | Inital learning rate for the model |
| `--num-epochs`     | Number of epochs you want to train for |
| `--lambda-coord`     | Yolo loss component coefficient: λ in order to focus more on detection |
| `--lambda-noobj`     | Yolo loss component coefficient: Down-weight loss from Class probability boxes that don’t contain objects |

This will create model file `<exp_name>_best_detector.pth` which will store weights of the model. Tensorboard logging is also interated, so while training you can check your loss plots through tensorboard by running the command on a different terminal window:

```
tensorboard --logdir=tb_log/ --bind_all
```
Go to the link that appears to visualise your plots. 

Once trained, you can evaluate the performance by running the command:
```
python main.py --eval --model-path="/path/to/model"
```
where 

| Arguments        | Description |
| :------------- |:----------|
| `--eval`     | To indicate that only evaluation needs to be perfromed |
| `--model-path`     | Relative path to \<exp_name>_best_detector.pth |

This will create a `my_solution.csv` which will contain MAP scores for each of the class. 

Once you have implemented the loss and successfully finish one iteration of training, feel free to change other parts of the network to boost the performance of your model.

# Submission Instructions

For each of the stage you need to upload three things on ELMS:
1. A ZIP of all code files including model file.
2. `my_solution.csv` - This needs to be uploaded **separately**.
3. A _brief_ report on the changes you made to boost performance.This needs to be uploaded **separately**.

Please folow all the instructions, failiure to do so will result in deduction of points. 

Due dates:
1. **Stage 1** - 11:59PM, May 4 (May the force be with you)
2. **Stage 2** - 11:59PM, May 11

## Ofice hours

I will be holding a single **combined** office hours on May 4 to clear all doubts. I will have a poll on piazza for the exact timing.

# Acknowledgements

The assignment is inspired from [Assignment 3 Part 2 of CS498](http://slazebni.cs.illinois.edu/fall18/assignment3_part2.html).