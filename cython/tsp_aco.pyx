# cython: language_level=3

import copy
import random
from math import *

class Agent:
    def __init__(self, towns, roads, start, pheromone):
        # value
        self.current = start
        self.alpha = 1
        self.beta = 5
        self.select_prob = 1
        self.candidate_size = 20

        # list
        self.whole = copy.deepcopy(towns)
        self.notVisited = copy.deepcopy(towns)
        self.notVisited.remove(start)
        self.visited = [towns[start]]
        self.candidate = []

        # dict
        self.roads = roads
        self.pheromone = copy.deepcopy(pheromone)
        self.way = {}
        for i in towns:
            for j in towns:
                if i is j:
                    continue
                self.way[(i, j)] = False

    def assessment(self, j):

        # denom = {l: (self.pheromone[(self.current, l)]**self.alpha) * ((1/self.roads[(self.current, l)]) ** self.beta) for l in self.notVisited}

        denom = {l: (self.pheromone[(self.current, l)]**self.alpha) * ((1/self.roads[(self.current, l)]) ** self.beta) for l in self.candidate}
        denominator = sum(denom.values())
        numerator = denom[j]
        assess = numerator / denominator

        return assess

    def probability(self):

        # Candidate Listを作る版
        # 選択可能な経路のリストを作成
        if len(self.notVisited) > self.candidate_size:
            croads = {(self.current, j): copy.deepcopy(self.roads[(self.current, j)]) for j in self.notVisited}
            croads = sorted(croads.items(), key= lambda x:x[1])

            self.candidate = [(croads[i][0])[1] for i in range(self.candidate_size)]
            del croads
        else:
            self.candidate = copy.deepcopy(self.notVisited)
        assesslist = {m: self.assessment(m) for m in self.candidate}
        # cdef double sum_of_assess = sum(assesslist.values())
        # p = {m: assesslist[m]/sum_of_assess for m in self.candidate}

        self.candidate.clear()

        # 全部評価する版
        # assesslist = {m: self.assessment(m) for m in self.notVisited}
        # sum_of_assess = sum(assesslist.values())
        # p = {m: assesslist[m]/sum_of_assess for m in self.notVisited}

        # return p
        return assesslist

    def agentwalk(self):
        cdef double choice
        for i in self.whole:
            prob = self.probability()
            prob_s = prob.items()
            choice = random.random()
            for i in prob_s:
                choice = choice - i[1]
                if choice < 0:
                    nextT = i
                    self.select_prob *= i[1]
                    break
            self.way[(self.current, nextT[0])] = True
            self.current = nextT[0]
            self.visited.append(nextT[0])
            self.notVisited.remove(nextT[0])
            if len(self.notVisited) == 0:
                return self.visited
        return self.visited

    def get_deltaphero(self):
        cdef double sum_of_len = self.get_length()
        pheromone_delta = {}
        for i in self.pheromone:
            if self.way[i]:
                pheromone_delta[i] = 1 / sum_of_len
            else:
                pheromone_delta[i] = 0
        return pheromone_delta

    def get_phero(self, pheromones_origin, pheromones_delta, double rho):
        for i in pheromones_origin:
            pheromones_origin[i] = (1-rho) * pheromones_origin[i] + pheromones_delta[i]
        return pheromones_origin

    def get_length(self):
        cdef double sum_of_len = 0.0
        for i in range(1, len(self.visited)):
            sum_of_len += self.roads[(self.visited[i-1], self.visited[i])]
        sum_of_len += self.roads[(self.visited[-1], self.visited[0])]
        return sum_of_len

    def get_way(self):
        return self.visited

    def get_select_prob(self):
        return self.select_prob


# NN法で初期解を求める
def nn(towns, roads):
    notvisited = copy.deepcopy(towns)
    whole = copy.deepcopy(towns)
    start = 0
    result = []
    current = start
    notvisited.remove(start)
    way = {}
    value = None

    for i in whole:
        minlength = inf
        for j in notvisited:
            if current is j:
                continue
            if roads[(current, j)] < minlength:
                minlength = roads[(current, j)]
                nextT = j
        result.append(minlength)
        current = nextT
        notvisited.remove(nextT)
        if len(notvisited) == 0:
            value = sum(result)
            value += roads[(current, start)]
            return value
    return value

def tsp():
    cdef int num_of_town = 127
    cdef int num_of_agent = 20
    cdef int num_of_solve = 1500
    cdef int num_of_exam = 10
    cdef double rho = 0.02
    random.seed()

    towns = []
    positions = []


    # 都市の情報を入力
    file = open('../problem/bier127.tsp')
    data1 = file.read()
    file.close()
    lines = data1.split('\n')
    for line in lines:
        cityinfo = line.split()
        towns.append(int(cityinfo[0])-1)
        positions.append((float(cityinfo[1]), float(cityinfo[2])))

    roads = {}
    for i in towns:
        for j in towns:
            if i is j:
                continue
            roads[(i, j)] = sqrt((positions[i][0] - positions[j][0])**2 +
                                 (positions[i][1] - positions[j][1])**2)

    # NN報で初期解を求める
    cdef double ans_primary = nn(towns, roads)

    pheromone = {}
    for i in towns:
        for j in towns:
            if i is j:
                continue
            else:
                pheromone[(i, j)] = 1 / (rho * ans_primary)

    cdef double min_length = inf
    best_agent = None

    cdef double phero_max = 1 / (rho * ans_primary)
    cdef double phero_min = 0.0

    cdef double inv_town = 1 / num_of_town
    cdef double half_town = num_of_town / 2.0

    result_log = []

    cdef double zantei_length = 0.0
    cdef double length = 0.0
    for j in range(num_of_exam):

        for i in range(num_of_solve):
            delta_pheros = []
            for m in range(num_of_agent):
                k = Agent(towns=towns, start=0, roads=roads, pheromone=pheromone)
                k.agentwalk()
                length = k.get_length()
                if (min_length is inf) or (min_length > length):
                    min_length = length
                    best_agent = copy.deepcopy(k)

            print(str(j) + "-" + str(i) + '番目の結果')
            zantei_length = best_agent.get_length()
            print(zantei_length)

            result_log.append(zantei_length)

            # 現状の最適解をもつエージェントがフェロモンを出す
            delta_pheros = best_agent.get_deltaphero()
            pheromone = best_agent.get_phero(pheromone, delta_pheros, rho)

            # フェロモンが制限を超えていないかどうかを判定
            phero_max = 1 / (rho * best_agent.get_length())
            p_best = best_agent.get_select_prob()
            phero_min = ((1 - pow(p_best, inv_town)) / (half_town - 1)) * phero_max

            for p in pheromone.values():
                if p > phero_max:
                    p = phero_max
                elif p < phero_min:
                    p = phero_min

        print('最終結果')
        print(best_agent.get_way())
        print(best_agent.get_length())
        print('履歴')
        for res in result_log:
            print(str(res))

        with open('../results/result' + str(j) + '.txt', mode='w') as f:
            for res in result_log:
                f.write(str(res) + '\n')

