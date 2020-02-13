#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 21:42:13 2020

@author: adamgosztolai
"""

G = { 1: [2, 3, 5], 2: [1], 3: [1], 4: [2], 5: [2] }

#generate a dictionary of neighbours
#G = {}
#for i in graph.nodes:
#    G[i] = [n for n in graph.neighbors(i)]

#find all cycles
#cycles = [[node]+path for node in G for path in dfs(G, node, node)]


def dfs(graph, start, end, limit=2):
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        if len(path)<limit:
            for next_state in graph[state]:
                if next_state in path:
                    continue
                fringe.append((next_state, path+[next_state]))
       
from scipy import stats
import numpy as np       
a = np.array([0.08010061, 0.0306243,  0.03732923])   
pdf = stats.gaussian_kde(a)
            
cycles = []
for node in G:
    for path in dfs(G, node, node):
        cycles.append([node]+path)
        

print(cycles)
            