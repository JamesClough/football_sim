""" Using odds for each game, perform Monte-Carlo simulation
of season outcomes"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from operator import itemgetter
import sys

team_colours = {'Arsenal':       ['#ef0107', '#023474'],
                'Aston Villa':   ['#83001f', '#8ccce5'],
                'Blackburn':     ['#1f618d', '#d00027'],
                'Blackpool':     ['#ef6c00', '#000000'],
                'Birmingham':    ['#42a5f5', '#42a5f5'],
                'Bolton':        ['#ffffff', '#1f618d'],
                'Bournemouth':   ['#e62333', '#000000'],
                'Burnley':       ['#8ccce5', '#53162f'],
                'Cardiff':       ['#005ca9', '#ffffff'],
                'Charlton':      ['#d00027', '#000000'],
                'Chelsea':       ['#034694', '#dba111'],
                'Crystal Palace':['#1b458f', '#c4122e'],
                'Derby':         ['#ffffff', '#000000'],
                'Everton':       ['#1E88E5', '#274488'],
                'Fulham':        ['#000000', '#000000'],
                'Hull':          ['#f5a12d', '#000000'],
                'Leeds':         ['#ffffff', '#1E88E5'],
                'Leicester':     ['#0053a0', '#0053a0'],
                'Liverpool':     ['#d00027', '#d00027'],
                'Man City':      ['#98c5e9', '#00285e'],
                'Man United':    ['#da020e', '#ffe500'],
                'Middlesbrough': ['#ff0000', '#000000'],
                'Middlesboro':   ['#ff0000', '#000000'],
                'Newcastle':     ['#000000', '#ffffff'],
                'Norwich':       ['#ffe600', '#00a850'],
                'Portsmouth':    ['#274488', '#E3F2FD'],
                'QPR':           ['#1d5ba4', '#e62333'],
                'Reading':       ['#ffffff', '#0000ff'],
                'Sheffield United':['#f44336', '#000000'],
                'Southampton':   ['#ed1a3b', '#ffffff'],
                'Stoke':         ['#e03a3e', '#1b449c'],
                'Sunderland':    ['#eb172b', '#ffffff'],
                'Swansea':       ['#ffffff', '#000000'],
                'Tottenham':     ['#ffffff', '#001c58'],
                'Watford':       ['#fbee23', '#ed2127'],
                'West Brom':     ['#091453', '#ffffff'],
                'West Ham':      ['#60223b', '#f7c240'],
                'Wigan':         ['#0033ff', '#ffffff'],
                'Wolves':        ['#D4AF37', '#000000']}

points_lookup = {'H':(3,0),
                 'A':(0,3),
                 'D':(1,1)}

def load_season_data(year, division=1, country='EN'):
    """

    NB: seasons are indexed by their starting year
    eg. year=2005 means the 2005-2006 season
    """

    data_filepath = os.path.join('data',
                    '%s%s_%s.csv' % (country, division, year))
    df = pd.read_csv(data_filepath)
    df.dropna(how='all', inplace=True)
    bookie = 'B365'
    if year < 2005:
        bookie = 'WH'
        # try to minimise missing data as much as possible
    for r in ['H', 'A', 'D']:
        df['p_%s' % r] = 1./df['%s%s' % (bookie, r)]

    # these probabilities won't be normalised as the bookies odds
    # don't add up precisely
    df['raw_sum_p'] = df['p_H'] + df['p_A'] + df['p_D']
    for r in ['H', 'A', 'D']:
        df['p_%s' % r] /= df['raw_sum_p']
    return df

def season_simulation(df, N):
    teams = df['HomeTeam'].unique()
    sim_df = pd.DataFrame(index=range(N), columns=teams, data=0)
    for match in df.iterrows():
        home = match[1]['HomeTeam']
        away = match[1]['AwayTeam']
        probs = [match[1]['p_H'], match[1]['p_A'], match[1]['p_D']]
        match_results = np.random.choice(['H', 'A', 'D'],
                                         size=N,
                                         p=probs)
        home_points = [points_lookup[x][0] for x in match_results]
        away_points = [points_lookup[x][1] for x in match_results]
        sim_df[home] += home_points
        sim_df[away] += away_points
    return sim_df

def season_ranking_probabilities(sim_df):
    N = len(sim_df)
    t_sim_df = sim_df.transpose()
    teams = t_sim_df.index
    team_dict = {team:{i:0 for i in range(1,len(teams)+1)} for team in teams}
    for i in range(N):
        season_points = t_sim_df.loc[:, i]
        season_points = season_points.sort_values(ascending=False)
        for j, team in enumerate(season_points.index):
            team_dict[team][j+1] += 1
    return team_dict

def get_order_by_sum(sim_df):
    teams = sim_df.columns
    ordered_teams = []
    for team in teams:
        sum_points = sim_df[team].sum()
        ordered_teams.append([team, sum_points])
    return sorted(ordered_teams, key=itemgetter(1))[::-1]

def plot_season_bars(ranks, team_order, year):
    fig = plt.figure()
    ax = plt.subplot(111)
    N = len(team_order)
    ind = np.arange(N)
    width = 0.35
    p_list = []
    bottom = np.array([0 for i in range(N)])
    for team in team_order:
        this_team_values = np.array([ranks[team][i] for i in range(1, N+1)])
        p = plt.bar(ind, this_team_values, width, bottom=bottom, color=team_colours[team][0],
                    edgecolor=team_colours[team][1], #hatch='//', hatchcolor='r',
                    lw=3., zorder=0)

        bottom += this_team_values
        p_list.append(p)
    #plt.legend(p_list, team_order)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(p_list, team_order, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_facecolor('#d5d8dc')
    plt.xticks([])
    plt.xlabel('Position')
    plt.ylabel('Probability')
    plt.title('Premier League %s-%s' % (year, year+1))
    plt.show()


if __name__ == "__main__":
    num_args = len(sys.argv)
    year = 2015
    N = 10000
    if num_args > 1:
        year = int(sys.argv[1])
    if num_args > 2:
        N = int(sys.argv[2])
    assert 2000 <= year <= 2015, 'Year out of range. 2000-2015 only.'
    df = load_season_data(year)
    sim_df = season_simulation(df, N)
    order_by_average = [t[0] for t in get_order_by_sum(sim_df)]
    ranks = season_ranking_probabilities(sim_df)
    plot_season_bars(ranks, order_by_average, year)
