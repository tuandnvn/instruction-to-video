This repository is used to store code for sequence to sequence model 
used to turn textual instructions into execution to navigate a block to traverse
a maze created with structures of different shapes and colors.

## Task description

Following is a *Maze traversal video*
![Sample video](miscelanous/sample_video.gif)

Samples of instructions corresponding to that video:

```
move the purple square down on the right side of the blue L and red L then move the purple square left between red L and purple L and it ends at the left side of the purple L

move the purple block down then left so that's directly between the blue L shape and the red L shape then move it down again until it reaches the top of the red L shape then move it straight left until it reaches 1 cell beyond the purple L shape

move the purple block down until it is in position between the 2 red L shapes and move it to the left until it is in front of purple L shape then move it down 1 block space

move the purple block down on the right of the blue L left and between the 2 red Ls to the left of the purple L
```

The learning objective is: given textual inputs describing actions (*instructions*), plan actions (sequence of action steps) on the grounded visual environment to match the instruction.

In training phase, a machine learning model is learned by feeding into it a parallel corpus of instructions and corresponding sequences of actions as video captures. In evaluating phase, the machine learning model is given a textual instruction and the starting configuration of visual environment (\textit{maze puzzle}), and it needs to direct a selected block to traverse the maze toward some final target. The evaluation objective is for the planned trajectory to follow as close as possible to the intended trajectory. 

## Needed library

```
pip install opencv-python

pip install tensorflow==1.8.0
```

## Models

Baseline model (Bahdanau attention model)

<img src="miscelanous/attention.png" width="400">

Improved model (Attention with feedback loop)

<img src="miscelanous/attention_image.png" width="550">

## Datasets

Videos from 0 to 299 are divided into 3 directories

```
target/0/*.mp4
target/1/*.mp4
target/2/*.mp4
```

Annotations used for training + validating (videos 0 to 199)

```
annotation.csv
```

Annotations used for testing (videos 200 to 299)

```
annotation2.csv
```

Annotations that have been turned into TRAIN/VALIDATE/TEST

Inputs:

```
data/instructions.txt
data/eval_instructions.txt
data/test_instructions.txt
```

Outputs

```
data/commands.txt
data/eval_commands.txt
data/test_commands.txt
```
