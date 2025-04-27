# March Madness Predictor #
### Author: Rylan Paul
#### April 26, 2025
#### [Rylan's Portfolio](https://rylanpaul.com)


## Bio & Info ##

I am a sophomore class of 2027 at the University of Notre Dame. Originally a CPEG, I am now an electrical engineer. The only incentive for creating this project was because I think it's cool. I had previously made one my junior year of highschool and reformatted it for the March Madness data during my freshman spring at Notre Dame. It performed slightly worse than my own bracket, but it was still a joy to create and watch unfold. This is my new and improved version! Given 2 teams, it will predict how many points each team will score both if they were home then away, then average their respective points. By default, the network uses LeakyReLU activation functions for everything. The default loss function is Mean Squared Error. For understanding, start in main.py, then work your way to neuralnetwork.py, objects.py, then all the other stuff is extra. Enjoy!  

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

## Summary ##
For the past two years now, I have created a small feedforward neural network to predict a perfect March Madness bracket. Although they were each far from perfect, I enjoyed developing them in Python. My 2024-2025 bracket picked Gonzaga beating Maryland in the final. While this was not very close, it did correctly have both Auburn and Duke losing in the semifinals. I would not say that this model was particularly great at predicting the winner, but rather it was excellent at predicting if a game was going to be very close or a blowout. The ESPN bracket achieved 23.8% correct with 670 points. Comparing this model with my personal 4 other brackets, this model placed exactly in the middle at third. This time around I created several Python data structures to assist in developing the network. The first object I created was one for the player to hold all of a player’s stats. This was useful as I could load data into a player object then utilize Python’s magic methods to get a player average with
```python
(Player1 + Player 2) / 2 = PlayerAverage
```
Additionally, I used Cbbpy to fetch data with Pandas to read the csv. Most math was done with Numpy, and its various functions. The network was trained with a size of **19** inputs, **3** layers of **64**, **32**, and **16** neurons each respectively, and an output layer of **2** neurons. The training data was from all games since 2021. Given two teams, the model will create the teams’ average players by averaging the top 5 players from every game this season. The inputs are the differences between the home team and player stats with the away team and player stats. The model then predicts how many points each team will score both if they were home then away, then averages their respective points. By default, the network uses LeakyReLU activation function, but there are plenty of included options to change that. The default loss function is Mean Squared Error, but again there are many different options. I also have scripts for displaying an in-line progress bar and one to draw the predicted bracket graphic.

## Developed by Rylan Paul
### [Rylan's Portfolio](https://rylanpaul.com)
[https://rylanpaul.com](https://rylanpaul.com)