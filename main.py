# main.py
# Author: Rylan Paul
# Data: Februrary 27, 2025

import pandas as pd
import matplotlib.pyplot as plt
from fetchdata import DataCollector
from progressbar import ProgressBar
from objects import *
import networkx as nx
from neuralnetwork import RylansNeuralNetwork
from typing import Tuple
from mathfunctions import Activations, LossFunctions
from random import randint as rand
import csv
from copy import deepcopy
class MarchMadnessNetwork(RylansNeuralNetwork):
    def fetch_games(self, seasons=[2021, 2022,2023,2024,2025]):
        if not hasattr(self, 'collector'):
            self.collector = DataCollector("data.csv")
        self.collector.scrape(seasons)
    def read_csvs(self):
        if not hasattr(self, 'collector'):
            self.collector = DataCollector("data.csv")
        self.df_games = self.collector.read_csv(0)
        self.df_players = self.collector.read_csv(1)
        self.players = Player.getPlayers(self.df_players)
    def read_teams(self, file='bracket_teams.txt'):
        if not hasattr(self, 'df_games'):
            self.read_csvs()
        teams = []
        with open(file, 'r') as txt:
            for line in list(txt)[1:]:
                parts = line.rstrip().lstrip().split(',')
                if len(parts) < 3:
                    continue
                teamname = parts[1]
                rank = int(parts[0])
                record = parts[2]
                index = rand(0,1)
                if '/' in teamname:
                    teamname = teamname.split('/')[index].strip()
                if '/' in record:
                    record = record.split('/')[index].strip()
                teams.append( (teamname.strip(), rank, record) )
        self.teams = teams
        return teams
    def loadMadness(file: str = 'weights-biases.csv') -> 'MarchMadnessNetwork':
        with open(file, 'r') as f:
            reader = csv.reader(f)
            weights_and_biases = list(reader)
        
        # The first row contains the layer sizes
        layer_sizes = list(map(int, weights_and_biases[0]))
        
        # Create a new network with the loaded layer sizes
        network = MarchMadnessNetwork(layer_sizes)
        
        # Load the weights and biases into the newly created network
        index = 1  # Start from the second row (first row is the layer sizes)
        for layer_index, layer in enumerate(network.layers):
            for neuron_index, neuron in enumerate(layer.neurons):
                # Load the weights and bias for each neuron
                row = np.array(weights_and_biases[index], dtype=float)
                neuron.weights = row[:-1]  # All except the last value are weights
                neuron.bias = row[-1]  # Last value is the bias
                index += 1
        return network
    def TrainMadness(self, trials=10000, learn_rate=1e-7):
        if not hasattr(self, 'df_games') or not hasattr(self, 'df_players'):
            self.read_csvs()
        self.train(self.df_games, self.df_players, trials, learn_rate)

def main():
    ### USER PARAMETERS ###
    # network = MarchMadnessNetwork([19, 64, 32, 16, 2])
    network = MarchMadnessNetwork.loadMadness()
    # network.TrainMadness(trials=100, learn_rate=1e-7)
    #network.save()

    ### DO NOT EDIT BELOW ###
    network.read_teams()
    bracket = [[]]
    n = 64 + 32 + 16 + 8 + 4 + 2 + 1
    pb = ProgressBar(total=n, desc="Running...", unit="", color=3, length=50, other="Loss")        # progress bar
    n_round = 0
    for team in network.teams:
        (name, rank, record) = team
        bracket[0].append(DataFormater.makeTeam(network.df_games, team=name, isHome=False, record=record, rank=rank))
    while n_round < 7:
        n_teams = 2 ** (6-n_round)
        bracket.append([])
        if n_teams == 1:
            pb.update(1)                                                                           # progress bar
            bracket[n_round][0].score = 0
            break
        for i in range(0,n_teams,2):
            home = bracket[n_round][i]
            away = bracket[n_round][i+1]
            game_1 = DataFormater.teamsToGame(network.df_games, network.df_players, home, away, season=2025, topPlayers=4)
            game_2 = DataFormater.teamsToGame(network.df_games, network.df_players, away, home, season=2025, topPlayers=4)
            (scorehome1, scoreaway1) = network.predict(np.array(DataFormater.gameToInputs(game_1)))
            (scorehome2, scoreaway2) = network.predict(np.array(DataFormater.gameToInputs(game_2)))
            # print(f'{home.team} {scorehome1:.2f} vs. {away.team} {scoreaway1:.2f}')  # prints out resluts
            # print(f'{away.team} {scorehome2:.2f} vs. {home.team} {scoreaway2:.2f}\n')
            scorehome = (scorehome1 + scoreaway2) / 2.0
            scoreaway = (scoreaway1 + scorehome2) / 2.0
            home.score = scorehome
            away.score = scoreaway
            bracket[n_round][i] = deepcopy(home)
            bracket[n_round][i+1] = deepcopy(away)
            home.score = -1
            away.score = -1
            bracket[n_round+1].append(home if scorehome >= scoreaway else away)
            pb.update(2)                                                                           # progress bar
        n_round += 1
    with open("bracket_display.txt", 'w') as display:
        first = True
        for round in bracket:
            if not first:
                display.write('\n')
            first = False
            for team in round:
                display.write(str(team))
                display.write('\n')

if __name__ == "__main__":
    main()


