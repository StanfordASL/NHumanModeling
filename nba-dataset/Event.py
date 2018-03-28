from Constant import Constant
from Moment import Moment
from Team import Team

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle, Arc
import numpy as np

import sys
sys.path.append("../code")
from stg_node import STGNode

class Event:
    """A class for handling and showing events"""

    def __init__(self, event, positions_map=None):
        self.id = event['eventId']
        moments = event['moments']
        self.moments = [Moment(moment) for moment in moments]
        home_players = event['home']['players']
        guest_players = event['visitor']['players']
        players = home_players + guest_players
        player_teams = ['Home' for player in home_players] + ['Away' for player in guest_players]
        player_ids = [player['playerid'] for player in players]
        player_names = [" ".join([player['firstname'],
                        player['lastname']]) for player in players]
        player_jerseys = [player['jersey'] for player in players]
        
        if positions_map is not None:
            player_positions = [positions_map[player_name] for player_name in player_names]
            
            # Fall back to the positions in this dataset if the player is not present in the map.
            for i in xrange(len(player_positions)):
                if len(player_positions[i]) == 0:
                    player_positions[i] = players[i]['position']
        else:
            player_positions = [player['position'] for player in players]
            
        values = list(zip(player_names, player_jerseys, player_positions))
        # Example: 101108: ['Chris Paul', '3', 'F-G']
        self.player_ids_dict = dict(zip(player_ids, values))
        
        # Example: 'Chris Paul': 'Home'
        self.player_teams_dict = dict(zip(player_names, player_teams))
        
        self.player_names = {key: value[0] for key, value in self.player_ids_dict.iteritems()}
        self.player_types = {value[0]: self.player_teams_dict[value[0]] + ''.join(value[2].split('-'))
                                 for key, value in self.player_ids_dict.iteritems()}

    def update_radius(self, i, player_circles, ball_circle, annotations, clock_info):
        moment = self.moments[i]
        for j, circle in enumerate(player_circles):
            circle.center = moment.players[j].x, moment.players[j].y
            annotations[j].set_position(circle.center)
            clock_test = 'Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}'.format(
                         moment.quarter,
                         int(moment.game_clock) % 3600 // 60,
                         int(moment.game_clock) % 60,
                         moment.shot_clock)
            clock_info.set_text(clock_test)
        ball_circle.center = moment.ball.x, moment.ball.y
        ball_circle.radius = moment.ball.radius / Constant.NORMALIZATION_COEF
        return player_circles, ball_circle

    def show(self):
        # Leave some space for inbound passes
        ax = plt.axes(xlim=(Constant.X_MIN,
                            Constant.X_MAX),
                      ylim=(Constant.Y_MIN,
                            Constant.Y_MAX))
        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)  # Remove grid
        start_moment = self.moments[0]
        player_dict = self.player_ids_dict

        clock_info = ax.annotate('', xy=[Constant.X_CENTER, Constant.Y_CENTER],
                                 color='black', horizontalalignment='center',
                                   verticalalignment='center')

        annotations = [ax.annotate(self.player_ids_dict[player.id][1], xy=[0, 0], color='w',
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold')
                       for player in start_moment.players]

        # Prepare table
        sorted_players = sorted(start_moment.players, key=lambda player: player.team.id)
        
        home_player = sorted_players[0]
        guest_player = sorted_players[5]
        column_labels = tuple([home_player.team.name, guest_player.team.name])
        column_colours = tuple([home_player.team.color, guest_player.team.color])
        cell_colours = [column_colours for _ in range(5)]
        
        home_players = [' #'.join([player_dict[player.id][0], player_dict[player.id][1]]) for player in sorted_players[:5]]
        guest_players = [' #'.join([player_dict[player.id][0], player_dict[player.id][1]]) for player in sorted_players[5:]]
        players_data = list(zip(home_players, guest_players))

        table = plt.table(cellText=players_data,
                              colLabels=column_labels,
                              colColours=column_colours,
                              colWidths=[Constant.COL_WIDTH, Constant.COL_WIDTH],
                              loc='bottom',
                              cellColours=cell_colours,
                              fontsize=Constant.FONTSIZE,
                              cellLoc='center')
        table.scale(1, Constant.SCALE)
        table_cells = table.properties()['child_artists']
        for cell in table_cells:
            cell._text.set_color('white')

        player_circles = [plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=player.color)
                          for player in start_moment.players]
        ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE,
                                 color=start_moment.ball.color)
        for circle in player_circles:
            ax.add_patch(circle)
        ax.add_patch(ball_circle)

        anim = animation.FuncAnimation(
                         fig, self.update_radius,
                         fargs=(player_circles, ball_circle, annotations, clock_info),
                         frames=len(self.moments), interval=Constant.INTERVAL)
        court = plt.imread("court.png")
        plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN])
        plt.show()

        
    def get_node_features_dict(self, features):
        node_features = dict()
        for player_name in features:
            player_node = STGNode(player_name, self.player_types[player_name])
            node_features[player_node] = features[player_name]
            
        return node_features
        
        
    def get_features_dict(self, only_initial=False, return_avg_time_diff=False):
        all_players = set()
        for moment in self.moments:
            all_players.update(moment.players)
        
        all_players = list(all_players)
        T = len(self.moments)
        
        # 6 = x, y, xd, yd, xdd, ydd
        features = {self.player_names[player.id]: np.zeros((T, 6)) for player in all_players}
        if len(features) == 0:
            if return_avg_time_diff:
                return dict(), 0.0
            else:
                return dict()
        
        moment_timestamps = np.zeros((len(self.moments), ))
        for t, moment in enumerate(self.moments):
            moment_timestamps[t] = moment.game_clock
            for i, player in enumerate(moment.players):
                player_name = self.player_names[player.id]
                features[player_name][t, 0] = player.x
                features[player_name][t, 1] = player.y
                
            if only_initial:
                # STGNode-ifying the features dict
                if return_avg_time_diff:
                    return self.get_node_features_dict(features), 0.0
                else:
                    return self.get_node_features_dict(features)
                
        # negative because game time decreases as real time advances
        avg_time_diff = -np.mean(np.ediff1d(moment_timestamps))
        if np.isclose(avg_time_diff, 0.0):
            if return_avg_time_diff:
                return dict(), 0.0
            else:
                return dict()
        
        for player in all_players:
            player_name = self.player_names[player.id]
            # speed
            features[player_name][:, 2] = np.gradient(features[player_name][:, 0], avg_time_diff)
            features[player_name][:, 3] = np.gradient(features[player_name][:, 1], avg_time_diff)
            
            # acceleration
            features[player_name][:, 4] = np.gradient(features[player_name][:, 2], avg_time_diff)
            features[player_name][:, 5] = np.gradient(features[player_name][:, 3], avg_time_diff)
                
        # STGNode-ifying the features dict
        if return_avg_time_diff:
            return self.get_node_features_dict(features), avg_time_diff
        else:
            return self.get_node_features_dict(features)
