import pandas as pd
import numpy as np
import random

from random import randrange
from numpy.random import rand

class AlgoritmoGeneticoBin:
    def __init__(self, probc, probm):
        self.probc = probc
        self.probm = probm

    # Função de Geração da População Inicial
    def populacaoBinaria(self, L, bits, xmin, xmax):
        Po_Binario = np.array(
            [np.array([randrange(0, 2) for p in range(0, bits)]) for ind in range(0, L)])  # Gera Po aleatório (binário)
        Po_Binario_Str = [''.join(item) for item in Po_Binario.astype(str)]  # Converte o array binário em string

        Po = np.array(list(map(lambda x: int(x, 2), Po_Binario_Str)))  # Converte binário em inteiro
        Po = xmin + Po * (xmax - xmin) / (2 ** bits - 1)  # Correspondente do binário em real (entre xmin e xmax)

        return Po, Po_Binario

    # Função do Fitness
    def fitness(self, x):
        fit = -x ** 4 + x ** 3 + 7 * x ** 2 - x - 6
        # fit = 1/((x-3)**2 + 0.1) + 1/((x-2)**2 + 0.05) +2
        return fit

    # Avaliação do Fitnes da População
    def avaliacao(self, Po):
        return np.array(list(map(lambda x: self.fitness(x), Po)))

    # Seleção dos Individuos para Reprodução
    def selecao_roleta(self, Po, Po_Binario, bits, Fitness):
        # Seleção usando o método da roleta
        L = len(Po)
        Probabilidade = Fitness / sum(Fitness)
        Prob_Acumulativa = Probabilidade.cumsum()

        Ps_Binario = np.zeros((L, bits))
        Ps = np.zeros(L)

        for i in range(0, L):
            roleta = rand()  # valor aleatório da roleta
            indice_Sel = min(
                np.where(Prob_Acumulativa >= roleta)[0])  # menor índice após corte na probabilidade cumulativa
            Ps[i] = Po[indice_Sel]
            Ps_Binario[i, :] = Po_Binario[indice_Sel, :]

        return Ps, Ps_Binario.astype(int)

    # Seleção dos Individuos para Reprodução
    def selecao_torneio(self, Po, Po_Binario, Fitness):
        # Seleção usando o método de torneio
        L = len(Po)
        indice_Sel = [i if Fitness_Inicial[i] > Fitness_Inicial[i + 1] else i + 1 for i in
                      range(0, L, 2)]  # checa dois a dois o maior fitness e retorna o índice
        Ps = Po[indice_Sel]
        Ps_Binario = Po_Binario[indice_Sel, :]

        return Ps, Ps_Binario

    # Operação de Cruzamento
    def cruzamento(self, Ps_Binario, bits):
        L = len(Ps_Binario)
        indice_Sel = [i for i in range(L) if np.array(rand()) <= self.probc]  # Índices que atendem r <= self.probc
        Ps_Binario_c = Ps_Binario[
            indice_Sel]  # Apenas os indivíduos que atendem a condição r <= self.probc é que serão usados no cruzamento

        Pc_Binario = Ps_Binario.copy()

        for i in range(0, int(len(indice_Sel) - len(indice_Sel) % 2), 2):
            alfa = np.random.randint(1,
                                     bits - 1)  # Alfa entre 1 e bits -1, pois 0 e bits indicam que não haverá alteração

            # Cruzamento
            Pc_Binario[indice_Sel[i], :] = np.concatenate(
                [Ps_Binario_c[i, :alfa], Ps_Binario_c[i + 1, alfa:]])  # Filho 1
            Pc_Binario[indice_Sel[i + 1], :] = np.concatenate(
                [Ps_Binario_c[i + 1, :alfa], Ps_Binario_c[i, alfa:]])  # Filho 2

        return Pc_Binario

        # Operador de Mutação

    def mutacao(self, Pc_Binario, bits, xmin, xmax):
        # Muta se número aleatório for menor do que self.probm. Função aplicada para cada indivíduo de Pc_Binario
        Pm_Binario = np.array(
            list(map(lambda indv: [int(not i) if np.array(rand()) <= self.probm else i for i in indv], Pc_Binario)))
        Pm_Binario_Str = [''.join(item) for item in Pm_Binario.astype(str)]

        Pfinal = np.array(list(map(lambda x: int(x, 2), Pm_Binario_Str)))
        Pfinal = xmin + Pfinal * (xmax - xmin) / (2 ** bits - 1)

        Fitness_Final = self.avaliacao(Pfinal)

        return Pm_Binario, Fitness_Final, Pfinal

    # Operador de Elitismo
    def elitismo(self, Fitness_Inicial, Fitness_Final, Po, Po_Binario, Pm_Binario, Pfinal):
        # Substituindo o pior da População Inicial com o Melhor da População Final
        Bad_Inicial = np.argmin(Fitness_Inicial)
        Best_Final = np.argmax(Fitness_Final)

        if Fitness_Final[Best_Final] > Fitness_Inicial[Bad_Inicial]:
            Po[Bad_Inicial] = Pfinal[Best_Final]
            Po_Binario[Bad_Inicial, :] = Pm_Binario[Best_Final, :]
            Fitness_Inicial = self.avaliacao(Po)

        return Po, Po_Binario, Fitness_Inicial