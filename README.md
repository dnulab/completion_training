# Onboarding Exercise
In this module, you will learn how to train your first completion model, test its accuracy, and visualize how models learn without ever seeing all of the data.
# Prerequisites
To start, make sure you have Python 3.12 or higher installed. There are two packages that must be installed for this code to work; `torch` and `numpy`. To do this, run the following in the terminal:
```
pip install torch numpy
```
Notes: 

1. If you want to use a CUDA GPU (NVIDIA), check your cuda driver version and install the corresponding version of torch.

2. It's often preferable to use a virtual environment (`venv`) for experiments of this kind.

# First Experiment: Train a memorization model

### 1. Generate the data.
To start your first experiment, you must first generate and store some values in a .txt file using one of the `make_inputs_*.py` files. We will start with `make_inputs_capital.py`. First, run the following command in your terminal:
```
python make_inputs_capital.py 3 26 1000
```
- 3: The maximum length of the input string.

- 26: The character set size (utilizing the slice of the lowercase English alphabet from a to z).

- 1000: The number of unique dataset lines to generate.

We'll be training a model of that learns how to capitalize short input strings. The output will be stored in a file called `inputcapital.txt`. You can open this file to see the generated data. Each line contains a string like `drb=DRB` or `md=MD`. The left side of the `=` is the input string, and the right side is the expected output. The model will learn to map the input to the output.

### 2. Prepare the data for training.
To transform the raw text into binary tokens (`train.bin`, `val.bin`) and a vocabulary mapping (`meta.pkl`) required by the transformer, run the preparation script:
```
cd 1-Char
python prepare_1char.py ../inputcapital.txt
```
Note: This partitions your data into a 90% training split and 10% validation split.
### 3. Train the model.
Now that the tokens are prepped, you can kick off the training routine. Run:
```
cd ..
python train_completions.py config/config_1char.py
```
By default, the model will run for 100 epochs (complete passes through the data) to learn the underlying sequence pattern. This may take about two minutes on a standard laptop.
- Train Loss: Represents how well the model is fitting the data it is actively studying.
- Val Loss: Represents how well the model generalizes to unseen validation data.
### 4. Test for accuracy.
Once training concludes, a model checkpoint named `completion_model.pth` will be saved inside the `out_1char/` directory. To evaluate its structural accuracy against your generated text, run:
```
python generate.py inputcapital.txt
```
At the bottom, it will output the accuracy, split between total processed, total correct, and a final accuracy.

You can also test individual input strings, for example using `python generate_one.py rg` to see if the model correctly outputs `RG`. You can test any string of length 1-3 using this method.

## Running the New Test Suite

The project now includes a small, refactor-focused pytest suite with fast and integration layers.

### Fast tests (recommended during active refactoring)
Runs CLI smoke checks and tiny data-logic checks only:

```
./venv/Scripts/python.exe -m pytest -q -m "not integration"
```

### Integration tests (end-to-end tiny pipeline)
Runs only integration checks (including tiny training/inference flow):

```
./venv/Scripts/python.exe -m pytest -q -m integration
```

### Full suite
Runs everything:

```
./venv/Scripts/python.exe -m pytest -q
```

# Second Experiment: Out-of-Distribution Generalization
How does an AI learn to solve problems it was never explicitly shown? In this experiment, you will test a model's ability to achieve true mathematical generalization, but this time, you'll need to apply the syntax you learned in the first module.
### 1. The Challenge: Generate a Sparse Math Dataset
Your goal is to generate 3,000 lines of addition problems modulo 100 using the make_inputs_add.py script.
Because a complete $100 \times 100$ addition table contains 10,000 total permutations, your model will only see 30% of the possible data during training.
Your Task: Note that `make_inputs_add.py` takes arguments `V` and `N`, where `V` is the modulus and `N` is the number of lines generated. Using this, create and run a command to generate the `inputadd.txt` based on the data presented earlier.
### 2. Prepare and Train the Data
Now, prepare your newly generated inputadd.txt file for training and kick off the training routine just like you did in the first experiment.
Pro-Tip: Mathematical patterns take longer to learn than simple memorization. Before running the training script, open config_1char.py and locate the epochs variable, and increase it (e.g., set epochs = 200 or higher) to give the network enough time to discover the underlying arithmetic logic.
### 3. Exhaustive Evaluation
While you can test accuracy using generate.py on your input file, we want to see if the model actually understands addition globally. We can test its conceptual understanding by sweeping every single possible combination from $0+0$ to $99+99$. To do this, input the following command into console:
```
python generate_all.py
```
Note that the accuracy is higher than 30%. That is because the model is learning (some of) the underlying pattern and not just memorizing the data.
