import pandas as pd
from copy import deepcopy
from collections.abc import Iterator
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from difflib import SequenceMatcher
from typing import Tuple

class Game: # represents a given game
    def __init__(self, df_games:pd.DataFrame, df_players:pd.DataFrame, gameid:str='-1', column='game_id', manual:tuple=None):
        if manual != None:
            self._manualsetup(*manual)
            return
        row = None
        if gameid == '-1': row = df_games.sample(n=1)
        else: row = df_games[df_games[column] == gameid]
        self.id = row[column].iloc[0]
        self.status = row['game_status'].iloc[0]

        self.home_team = Team(df_games, df_players, True, self.id, column)
        self.away_team = Team(df_games, df_players, False, self.id, column)
        # home_point_spread, 10 means that they are +10pts, AKA underdogs
        keys = ['home_point_spread', 'home_win', 'num_ots', 'is_conference', 'is_neutral', 'is_postseason', 'tournament', 'game_day', 'game_time', 'game_loc', 'arena', 'arena_capacity', 'attendance', 'tv_network', 'referee_1', 'referee_2', 'referee_3']
        self.stats = {}
        for key in keys:
            if key == "game_day":
                value = datetime.strptime(str(row[key].iloc[0]), "%B %d, %Y")
                self.stats[key] = value
            try:
                value = int(row[key].iloc[0])
                self.stats[key] = value
            except:
                try:
                    value = float(row[key].iloc[0])
                    self.stats[key] = value
                except:
                    try:
                        value = str(row[key].iloc[0])
                        self.stats[key] = value
                    except:
                        self.stats[key] = row[key].iloc[0]
        try:
            self.stats['home_win'] = True if self.stats['home_win'] == 1 else False
            self.stats['is_neutral'] = True if self.stats['is_neutral'] == 1 else False
            self.stats['is_neutral'] = True if self.stats['is_postseason'] == 1 else False
            self.stats['is_neutral'] = True if self.stats['tournament'] == 1 else False
        except:
            pass
    def _manualsetup(self, hometeam:'Team', awayteam: 'Team'):
        self.home_team = hometeam
        self.away_team = awayteam
        self.stats = None
    def __str__(self):
        if self.stats != None:
            stats = '\n'.join( list(f'{key}: {value}' for key, value in self.stats.items()) )
            return f'{self.home_team}\n{self.away_team}\n\n{stats}'
        return f'{self.home_team}\n{self.away_team}\n'
    def getHome(self) -> 'Team':
        return self.home_team
    def getAway(self) -> 'Team':
        return self.away_team
    def getTeamSeasonGames(df_games:pd.DataFrame, df_players:pd.DataFrame, season:int=2025, teamname:str="Notre Dame Fighting Irish", teamid:str='-1') -> Iterator['Game']:
        seasonstart = datetime(season-1, 10, 20) # mid october, because games start in November
        games = None
        if teamid != '-1':
            games = df_games[(df_games['home_id'] == teamid) | (df_games['away_id'] == teamid)]
        else:
            games = df_games[(df_games['home_team'] == teamname) | (df_games['away_team'] == teamname)]
        for index, row in games.iterrows():
            gameday = datetime(2020, 1, 1)
            try:
                gameday = datetime.strptime(str(row['game_day']), "%B %d, %Y")
            except:
                pass
            if gameday > seasonstart:
                yield Game(df_games, df_players, row['game_id'])
    def RandomGame(df_games: pd.DataFrame, df_players: pd.DataFrame) -> 'Game':
        game = None
        while game == None:
            game = Game(df_games, df_players)
            if game.home_team == None or game.away_team == None or game.status.lower() != 'final':
                game = None
            elif len(DataFormater.gameToInputs(game)) < 1:
                game = None
        return game
    def GamesGenerator(df_games: pd.DataFrame, df_players: pd.DataFrame, numberGames:int=6000) -> Iterator['Game']:
        for _ in range(numberGames):
            yield Game.RandomGame(df_games, df_players)
class Player:
    def __init__(self, df: pd.Series):
        # Assuming df is a Series representing a single player's data
        self.game_id = df['game_id']
        self.team = df['team']
        self.player = df['player']
        self.player_id = df['player_id']
        self.position = df['position'] if df['position'] != "TOTAL" else 'T'
        self.starter = df['starter']
        self.min = float(df['min'])
        self.fgm = float(df['fgm'])
        self.fga = float(df['fga'])
        self.two_pm = float(df['2pm'])
        self.two_pa = float(df['2pa'])
        self.three_pm = float(df['3pm'])
        self.three_pa = float(df['3pa'])
        self.ftm = float(df['ftm'])
        self.fta = float(df['fta'])
        self.oreb = float(df['oreb'])
        self.dreb = float(df['dreb'])
        self.reb = float(df['reb'])
        self.ast = float(df['ast'])
        self.stl = float(df['stl'])
        self.blk = float(df['blk'])
        self.to = float(df['to'])
        self.pf = float(df['pf'])
        self.pts = float(df['pts'])
    def __str__(self):
        return (
            f"{self.player[:20]:<20} "  # Left-aligned player name, 15 characters wide
            f"[{self.position:^3}] "  # Centered position, 3 characters wide
            f"{self.team[:20]:<20} "  # Left-aligned team name, 5 characters wide
            f"{str(self.pts)[:4]:>4} pts "  # Right-aligned points, 3 characters wide
            f"{str(self.min)[:5]:>5} min "  # Right-aligned minutes, 3 characters wide
            f"{str(self.fgm)[:4]:>4} fgm "  # Right-aligned field goals made, 2 characters wide
            f"{str(self.reb)[:4]:>4} reb"  # Right-aligned rebounds, 3 characters wide
        )
    def __repr__(self):
        return f"Player(player={self.player}, team={self.team}, pts={self.pts})"
    def _add_number(self, number) -> 'Player':
        p = deepcopy(self)
        p.min = self.min + number
        p.fgm = self.fgm + number
        p.fga = self.fga + number
        p.two_pm = self.two_pm + number
        p.two_pa = self.two_pa + number
        p.three_pm = self.three_pm + number
        p.three_pa = self.three_pa + number
        p.ftm = self.ftm + number
        p.fta = self.fta + number
        p.oreb = self.oreb + number
        p.dreb = self.dreb + number
        p.reb = self.reb + number
        p.ast = self.ast + number
        p.stl = self.stl + number
        p.blk = self.blk + number
        p.to = self.to + number
        p.pf = self.pf + number
        p.pts = self.pts + number
        return p
    def _sub_number(self, number) -> 'Player':
        p = deepcopy(self)
        p.min = self.min - number
        p.fgm = self.fgm - number
        p.fga = self.fga - number
        p.two_pm = self.two_pm - number
        p.two_pa = self.two_pa - number
        p.three_pm = self.three_pm - number
        p.three_pa = self.three_pa - number
        p.ftm = self.ftm - number
        p.fta = self.fta - number
        p.oreb = self.oreb - number
        p.dreb = self.dreb - number
        p.reb = self.reb - number
        p.ast = self.ast - number
        p.stl = self.stl - number
        p.blk = self.blk - number
        p.to = self.to - number
        p.pf = self.pf - number
        p.pts = self.pts - number
        return p
    def __add__(self, other) -> 'Player':
        if isinstance(other, (int, float)):
            return self._add_number(other)
        if not isinstance(other, Player):
            return NotImplemented
        p = deepcopy(self)
        p.min = self.min + other.min
        p.fgm = self.fgm + other.fgm
        p.fga = self.fga + other.fga
        p.two_pm = self.two_pm + other.two_pm
        p.two_pa = self.two_pa + other.two_pa
        p.three_pm = self.three_pm + other.three_pm
        p.three_pa = self.three_pa + other.three_pa
        p.ftm = self.ftm + other.ftm
        p.fta = self.fta + other.fta
        p.oreb = self.oreb + other.oreb
        p.dreb = self.dreb + other.dreb
        p.reb = self.reb + other.reb
        p.ast = self.ast + other.ast
        p.stl = self.stl + other.stl
        p.blk = self.blk + other.blk
        p.to = self.to + other.to
        p.pf = self.pf + other.pf
        p.pts = self.pts + other.pts
        return p
    def __sub__(self, other) -> 'Player':
        if isinstance(other, (int, float)):
            return self._sub_number(other)
        if not isinstance(other, Player):
            return NotImplemented
        p = deepcopy(self)
        p.min = self.min - other.min
        p.fgm = self.fgm - other.fgm
        p.fga = self.fga - other.fga
        p.two_pm = self.two_pm - other.two_pm
        p.two_pa = self.two_pa - other.two_pa
        p.three_pm = self.three_pm - other.three_pm
        p.three_pa = self.three_pa - other.three_pa
        p.ftm = self.ftm - other.ftm
        p.fta = self.fta - other.fta
        p.oreb = self.oreb - other.oreb
        p.dreb = self.dreb - other.dreb
        p.reb = self.reb - other.reb
        p.ast = self.ast - other.ast
        p.stl = self.stl - other.stl
        p.blk = self.blk - other.blk
        p.to = self.to - other.to
        p.pf = self.pf - other.pf
        p.pts = self.pts - other.pts
        return p
    def _div_number(self, number:float) -> 'Player':
        if number == 0:
            number = 1e-6
        p = deepcopy(self)
        p.min = self.min / number
        p.fgm = self.fgm / number
        p.fga = self.fga / number
        p.two_pm = self.two_pm / number
        p.two_pa = self.two_pa / number
        p.three_pm = self.three_pm / number
        p.three_pa = self.three_pa / number
        p.ftm = self.ftm / number
        p.fta = self.fta / number
        p.oreb = self.oreb / number
        p.dreb = self.dreb / number
        p.reb = self.reb / number
        p.ast = self.ast / number
        p.stl = self.stl / number
        p.blk = self.blk / number
        p.to = self.to / number
        p.pf = self.pf / number
        p.pts = self.pts / number
        return p
    def __truediv__(self, other) -> 'Player':
        if isinstance(other, (int, float)):
            return self._div_number(other)
        if not isinstance(other, Player):
            return NotImplemented
        p = deepcopy(self)
        min_val = 1e-6
        def safedivide(a,b):
            return a / (b if b != 0 else min_val)
        p.min = safedivide(self.min, other.min)
        p.fgm = safedivide(self.fgm, other.fgm)
        p.fga = safedivide(self.fga, other.fga)
        p.two_pm = safedivide(self.two_pm, other.two_pm)
        p.two_pa = safedivide(self.two_pa, other.two_pa)
        p.three_pm = safedivide(self.three_pm, other.three_pm)
        p.three_pa = safedivide(self.three_pa, other.three_pa)
        p.ftm = safedivide(self.ftm, other.ftm)
        p.fta = safedivide(self.fta, other.fta)
        p.oreb = safedivide(self.oreb, other.oreb)
        p.dreb = safedivide(self.dreb, other.dreb)
        p.reb = safedivide(self.reb, other.reb)
        p.ast = safedivide(self.ast, other.ast)
        p.stl = safedivide(self.stl, other.stl)
        p.blk = safedivide(self.blk, other.blk)
        p.to = safedivide(self.to, other.to)
        p.pf = safedivide(self.pf, other.pf)
        p.pts = safedivide(self.pts, other.pts)
        return p
    def _mul_number(self, number:float) -> 'Player':
        p = deepcopy(self)
        p.min = self.min * number
        p.fgm = self.fgm * number
        p.fga = self.fga * number
        p.two_pm = self.two_pm * number
        p.two_pa = self.two_pa * number
        p.three_pm = self.three_pm * number
        p.three_pa = self.three_pa * number
        p.ftm = self.ftm * number
        p.fta = self.fta * number
        p.oreb = self.oreb * number
        p.dreb = self.dreb * number
        p.reb = self.reb * number
        p.ast = self.ast * number
        p.stl = self.stl * number
        p.blk = self.blk * number
        p.to = self.to * number
        p.pf = self.pf * number
        p.pts = self.pts * number
        return p
    def __mul__(self, other) -> 'Player':
        if isinstance(other, (int, float)):
            return self._mul_number(other)
        if not isinstance(other, Player):
            return NotImplemented
        p = deepcopy(self)
        p.min = self.min * other.min
        p.fgm = self.fgm * other.fgm
        p.fga = self.fga * other.fga
        p.two_pm = self.two_pm * other.two_pm
        p.two_pa = self.two_pa * other.two_pa
        p.three_pm = self.three_pm * other.three_pm
        p.three_pa = self.three_pa * other.three_pa
        p.ftm = self.ftm * other.ftm
        p.fta = self.fta * other.fta
        p.oreb = self.oreb * other.oreb
        p.dreb = self.dreb * other.dreb
        p.reb = self.reb * other.reb
        p.ast = self.ast * other.ast
        p.stl = self.stl * other.stl
        p.blk = self.blk * other.blk
        p.to = self.to * other.to
        p.pf = self.pf * other.pf
        p.pts = self.pts * other.pts
        return p
    def plot_stats(players):
        players = list(players)
        player_names = [player.player for player in players]
        
        pts = [player.pts for player in players]
        fgm = [player.fgm for player in players]
        reb = [player.reb for player in players]
        fga = [player.fga for player in players]

        # Creating a bar chart
        bar_width = 0.25
        n = len(players)
        x = np.arange(n)
        plt.bar(x - bar_width / 1.5, fgm, width=bar_width, label="FGM", align="center", color='g')
        plt.bar(x, pts, width=bar_width, label="PTS", align="center", color='b')
        plt.bar(x + bar_width / 1.5, fga, width=bar_width, label="FGA", align="center", color='orange')
        plt.bar(x + bar_width, reb, width=bar_width, label="REB", align="center", color='red')
        

        # Adding labels and title
        plt.xlabel('Players')
        plt.ylabel('Stats')
        plt.title('Comparison of Player Stats')
        plt.xticks(x, player_names, rotation=45)
        plt.legend()

        # Display the plot
        plt.tight_layout()
        plt.show()
    def getPlayers( df:pd.DataFrame, gameid:str='-1', teamname:str='') -> Iterator['Player']:
        if gameid < 0:
            gameid = df['game_id'].sample(1).iloc[0]
        players = df[df['game_id'] == gameid]
        if teamname == '':
            teamname = players['team'].sample(1).iloc[0]
        players = df[(df['team'] == teamname) & (df['game_id'] == gameid)]
        for _, row in players.iterrows():
            if row['player'].lower().strip() == 'team':
                continue
            yield Player(row)
    def averagePlayer(players) -> 'Player':
        players = list(players)
        if len(players) == 1:
            return players[0]
        elif len(players) == 0:
            return None
        p = players[0]
        for player in players[1:]:
            p = p + player
        p = p / len(players)
        p.player = 'Team'
        return p
class Team:
    def __init__(self, df_games:pd.DataFrame, df_players:pd.DataFrame, isHomeTeam:bool=True, gameid:str='-1', column='game_id', manual:tuple=None):
        if manual != None:
            self._manualsetup(*manual)
            return
        row = None
        if gameid == '-1': row = df_games.sample(n=1)
        else: row = df_games[df_games[column] == gameid]
        home = 'home' if isHomeTeam else 'away'
        self.home = isHomeTeam
        self.team = row[f'{home}_team'].iloc[0]
        self.id = row[f'{home}_id'].iloc[0]
        self.rank = int(row[f'{home}_rank'].iloc[0])
        self.record = str(row[f'{home}_record'].iloc[0])
        self.score = row[f'{home}_score'].iloc[0]

        if (self.rank < 1): # unranked
            self.rank = 50
        self.players = np.array(list(Player.getPlayers(df_players, gameid, self.team)))
    def _manualsetup(self, isHome: bool, teamname: str, teamid: str, rank:int, record: str, players: np.ndarray, score:int=-1):
        self.home = isHome
        self.team = teamname
        self.id = teamid
        self.rank = rank
        self.record = record
        self.score = score

        if (self.rank < 1): # unranked
            self.rank = 50
        self.players = players
        return
    def MakeTeam(isHome: bool, teamname: str, teamid: str, rank:int, record: str, players: np.ndarray=[], score:int=-1) -> 'Team':
        return Team(None, None, manual=(isHome, teamname, teamid, rank, record, players, score))
    def __str__(self):
        return f"{self.rank:<2} {self.team} {str(int(round(self.score))):<3}"
class DataFormater:
    def getInputs(df:pd.DataFrame, gameid:int=-1) -> np.ndarray:
        game = Game(df, gameid)
        return DataFormater.gameToInputs(game)
    def closest_key(search_string, keys):
        def similarity(a, b):
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()
        return max(keys, key=lambda k: similarity(search_string, k))
    def batchGameToInputs(df:pd.DataFrame, gameid:int=-1, numbergames: int=1) -> np.ndarray:
        return [ DataFormater.getInputs(df, gameid) for _ in range(numbergames) ]
    def gameToInputs(game: Game, topPlayers: int=5) -> np.ndarray:
        home = game.home_team
        away = game.away_team
        rank = float(away.rank - home.rank)
        record1 = home.record.split('-')
        record2 = away.record.split('-')
        wins = 0
        losses = 0
        if len(record1) > 1 and len(record2) > 1:
            wins =  float(int(record1[0]) - int(record2[0]))
            losses = float(int(record1[1]) - int(record2[1]))
        homeplayer = Player.averagePlayer(sorted(home.players, key=lambda p: p.pts, reverse=True)[:topPlayers])
        awayplayer = Player.averagePlayer(sorted(away.players, key=lambda p: p.pts, reverse=True)[:topPlayers])
        if homeplayer == None or awayplayer == None:
            return []
        player = homeplayer - awayplayer
        return [rank, wins, losses, player.pts, player.fgm, player.fga, player.two_pm, player.two_pa, player.three_pm, player.three_pa, player.ftm, player.fta, player.oreb, player.dreb, player.ast, player.stl, player.blk, player.to, player.pf]
    def gameToOutputs(game: Game) -> np.ndarray:
        if hasattr(game,'home_team'):
            if hasattr(game.home_team,'score'):
                if hasattr(game,'away_team'):
                    if hasattr(game.away_team,'score'):
                        return [int(game.home_team.score), int(game.away_team.score)]
        return [0]
    def searchTeam(df_games:pd.DataFrame, team: str) -> Tuple[str, str]:
        if team.lower() == 'alabama':
            team += ' crimson tide'
        elif team.lower() == 'iu':
            team = 'indiana hoosiers'
        elif team.lower() == 'duke':
            team += ' blue devils'
        elif team.lower() == 'st. johns' or team.lower() == 'st johns' or team.lower() == 'saint johns':
            team += ' red storm'
        if 'st.' in team.lower() or 'st' in team.lower():
            words = team.split()
            for index, word in enumerate(words):
                if word.lower() == 'st' or word.lower() == 'st.':
                    if index == 0:
                        words[index] = 'saint'
                    else:
                        words[index] = 'state'
            team = ' '.join(words)
        teams = {}
        # seach through team names
        df_selected = df_games[["home_team", "home_id"]]
        rows = df_selected.iterrows()
        t = next(rows)[1]
        for row in rows:
            t = row[1]
            name = t['home_team']
            id = t['home_id']
            if name not in teams:
                teams[name] = id
        name = DataFormater.closest_key(team, teams) # get closest match
        return (name, teams[name])

        pass
    def teamsToGame(df_games:pd.DataFrame, df_players:pd.DataFrame, home:Team, away:Team, season:int=2025, topPlayers: int=5) -> Game:
        # get games from October 2024 to today, then get top 5 players of this team for every game, then average then
        home_games = Game.getTeamSeasonGames(df_games=df_games, df_players=df_players, season=season, teamname=home.team, teamid=home.id)
        away_games = Game.getTeamSeasonGames(df_games=df_games, df_players=df_players, season=season, teamname=away.team, teamid=away.id)
        # Calculate average players, then create a game object
        home_players = []
        away_players = []
        for game in home_games:
            players = sorted(Player.getPlayers(df_players, game.id, home.team), key=lambda p: p.pts, reverse=True)[:topPlayers]
            for player in players:
                home_players.append(player)
        for game in away_games:
            players = sorted(Player.getPlayers(df_players, game.id, away.team), key=lambda p: p.pts, reverse=True)[:topPlayers]
            for player in players:
                away_players.append(player)
        if len(home_players) < 1 or len(away_players) < 1:
            return None
        home_player = Player.averagePlayer(home_players)
        away_player = Player.averagePlayer(away_players)
        home.players = [home_player]
        away.players = [away_player]
        return Game(df_games, df_players, manual=(home, away))
    def makeTeam(df_games:pd.DataFrame, team: str, isHome: bool, record: str, rank:int=-1) -> Team:
        (name, id) = DataFormater.searchTeam(df_games, team)
        return Team.MakeTeam(isHome, name, id, rank, record)
if __name__ == "__main__":
    from fetchdata import DataCollector
    collector = DataCollector("data.csv")
    df_games = collector.read_csv(0)
    df_players = collector.read_csv(1)
    home = DataFormater.makeTeam(df_games, team='auburn', isHome=True, record='27-3', rank=1)
    away = DataFormater.makeTeam(df_games, team='purdue', isHome=False, record='27-3', rank=2)
    game = DataFormater.teamsToGame(df_games, df_players, home, away, season=2025, topPlayers=5)
    print(game.home_team.players[0])