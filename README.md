# March Madness Predictor #
### Author: Rylan Paul
#### April 26, 2025

## Bio & Info ##

I am a sophomore class of 2027 at the University of Notre Dame. Originally a CPEG, I am now an electrical engineer. The only incentive for creating this project was because I think it's cool. I had previously made one my junior year of highschool and reformatted it for the March Madness data during my freshman spring at Notre Dame. It performed slightly worse than my own bracket, but it was still a joy to create and watch unfold. This is my new and improved version! Given 2 teams, it will predict how many points each team will score both if they were home then away, then average their respective points. By default, the network uses LeakyReLU activation functions for everything. The default loss function is Mean Squared Error. For understanding, start in main.py, then work your way to neuralnetwork.py, objects.py, then all the other stuff is extra. Enjoy!  
##### - Rylan Paul

## Files ##

> ### bracket.py
> - This displays a bracket using a pipline like:$ cat bracket_display.txt | bracket.py

> ### dataX.csv
> - These are the csv data files from game to player stats that were scraped using the Python CbbPy library

> ### fetchdata.py
> - This uses teh CbbPy library to download the training data

> ### progressbar.py
> - This enables me to show a progress bar for various things while the user waits

> ### mathfunctions.py
> - This contains various options for activation functions and loss functions as well as there derivatives

> ### objects.py
> - This contains many nice data structres that allow convenient use of moving all of the data around

> ### neuralnetwork.py
> - This has the main neural network class structure from layers to neurons to backpropagation

> ### main.py
> - Run this to train the network, save weights and biases, and then draw a bracket from bracket_teams.txt

> ### usage.txt
> - This contains all the functions for cbbpy

> ### weights-biases.csv
> - This stores all of the saved weights and biases

> ### weights-biases-RP.csv (readonly)
> - This stores all of the saved weights and biases from Rylan's final training


## See the Output ##

### 1. Run bracket as I finished it

```sh
$ python3 main.py
```

> this will print out the bracket to bracket_display.txt


### 2. Display bracket

```sh
$ cat bracket_display.txt | python3 bracket.py
```

> this will take the resulting bracket and display it

### 3. Alternatively combine this into one command
    
```sh
$ python3 main.py; cat bracket_display.txt | python3 bracket.py
```

### 4. Main.py main function example:
```python
def main():
    # USER PARAMETERS
    network = MarchMadnessNetwork.loadMadness()
    # DO NOT EDIT BELOW
    ...
```

## Run the Project ##

> Note: This can all be done in main.py in the main function
1. Uncomment the network intialization (sometimes)
    ```python
    network = MarchMadnessNetwork([19, 64, 32, 16, 2])
    ```
    - the array is the size: [inputs, layer1, ... , layerN, output]
    - This creates a brand new model with randomized weights
    - You'll want to do this for a new model
2. Comment out network load (sometimes)
    ```python
    network = MarchMadnessNetwork.loadMadness()
    ```
    - this loads the saved network from weights-biases.csv
    - (Comment out the first time, then uncomment to load your saved model)
3. Train Model
    ```python
    network.TrainMadness(trials=10000, learn_rate=1e-7)
    ```

    - change trials to how many games to train on
    - change learning rate to affect the weight of backpropagation
    - if you keeping reaching infinity with nan weights, then lower your learning rate
4. Save Model
    ```python
    network.save()
    ```

    - writes the size, weights, and biases to weights-biases.csv
    - must call this to load it next time
5. Creaing a New Model Example
    ```python
    def main():
        // USER PARAMETERS
        network = MarchMadnessNetwork([19, 64, 32, 16, 2])
        network.TrainMadness(trials=10000, learn_rate=1e-7)
        network.save()
        // DO NOT EDIT BELOW
        ...
    ```
6. Loading a Saved Model Example
    ```python
    def main():
        // USER PARAMETERS
        network = MarchMadnessNetwork.loadMadness()
        network.TrainMadness(trials=10000, learn_rate=1e-7)
        network.save()
        // DO NOT EDIT BELOW
        ...
    ```