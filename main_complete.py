###############################################################################
# main_minimal.py: minimal script for calling BRKGA algorithms to solve
#                  instances of the Traveling Salesman Problem.
#
# (c) Copyright 2019, Carlos Eduardo de Andrade.
# All Rights Reserved.
#
# This code is released under LICENSE.md.
#
# Created on:  Nov 18, 2019 by ceandrade
# Last update: Nov 18, 2019 by ceandrade
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import sys

from brkga_mp_ipr.enums import Sense
from brkga_mp_ipr.types_io import load_configuration
from brkga_mp_ipr.algorithm import BrkgaMpIpr

from tsp_instance import TSPInstance
from tsp_decoder import TSPDecoder

from copy import copy, deepcopy
import datetime
import sys
from copy import deepcopy
from datetime import datetime
from os.path import basename
import random
import time


###############################################################################


def custo_inicial(cromossomos, instance_file, qtd_grupos, qtd_variaveis, qtd_min_por_grupo):

    instance = TSPInstance(instance_file)

    #remova essas 3 linhas caso queira o dataset com todas as colunas
    colunas_selecionadas = ['Compr_Renda', 'Nivel_Escolaridade', 'Taxa', 'Estado_Civil', 'Regiao', 'Flag_Efet', 'Nivel_Risco_Novo']
    colunas_removidas = [col for col in instance.df.columns if col not in colunas_selecionadas]
    instance.df.drop(colunas_removidas, axis=1, inplace=True)

    instance.tratamento_dados()
    instance.df_original = instance.df.copy()
    decoder = TSPDecoder(instance, qtd_grupos, qtd_variaveis, qtd_min_por_grupo)
    result = decoder.decode(cromossomos, False)
    return result
    

def imprimir_saida(cromossomos, instance_file, qtd_grupos, qtd_variaveis, qtd_min_por_grupo):

    instance = TSPInstance(instance_file)

    #remova essas 3 linhas caso queira o dataset com todas as colunas
    colunas_selecionadas = ['Compr_Renda', 'Nivel_Escolaridade', 'Taxa', 'Estado_Civil', 'Regiao', 'Flag_Efet', 'Nivel_Risco_Novo']
    colunas_removidas = [col for col in instance.df.columns if col not in colunas_selecionadas]
    instance.df.drop(colunas_removidas, axis=1, inplace=True)

    instance.tratamento_dados()
    instance.df_original = instance.df.copy()
    decoder = TSPDecoder(instance, qtd_grupos, qtd_variaveis, qtd_min_por_grupo)
    print(decoder.decode(cromossomos, False))
    instance.df.to_excel("saida_final.xlsx")

def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: python main_minimal.py <seed> <config-file> "
              "<num-generations> <tsp-instance-file>")
        sys.exit(1)

    ########################################
    # Read the command-line arguments and the instance
    ########################################

    seed = int(sys.argv[1])
    configuration_file = sys.argv[2]
    num_generations = int(sys.argv[3])
    instance_file = sys.argv[4]

    #Lendo dados e fazendo limpeza"
    print("Reading parameters...")
    brkga_params, _ = load_configuration(configuration_file)

    print(f"\n[{datetime.now()}] Reading TSP data...")

    instance = TSPInstance(instance_file)
    colunas_selecionadas = ['Compr_Renda', 'Nivel_Escolaridade', 'Taxa', 'Estado_Civil', 'Regiao', 'Flag_Efet', 'Nivel_Risco_Novo']
    colunas_removidas = [col for col in instance.df.columns if col not in colunas_selecionadas]
    instance.df.drop(colunas_removidas, axis=1, inplace=True)
    instance.df = instance.df[0:19995]
    instance.tratamento_dados()
    instance.df_original = instance.df.copy()

    #definir valores
    instance = instance
    qtd_grupos = 4
    qtd_variaveis = 5
    qtd_min_por_grupo = 50
    instance.num_nodes = len(instance.df.index)+qtd_variaveis

    cromossomo_inicial = [random.random() for _ in range(0,instance.num_nodes)]
    ci = 0
    print(f"Custo inicial {ci}")
    print("Building BRKGA data and initializing...")


    decoder = TSPDecoder(instance, qtd_grupos, qtd_variaveis, qtd_min_por_grupo)

    brkga = BrkgaMpIpr(
        decoder=decoder,
        sense=Sense.MAXIMIZE,
        seed=seed,
        chromosome_size=instance.num_nodes,
        params=brkga_params
    )
    import pandas as pd

    # cromossomo_inicial = pd.read_excel("Cromossomo.xlsx")
    brkga.set_initial_population([cromossomo_inicial])


    # NOTE: don't forget to initialize the algorithm.
    brkga.initialize()


    print(f"\n[{datetime.now()}] Evolving...")
    print("* Iteration | Cost | CurrentTime")

    best_cost = ci
    best_chromosome = cromossomo_inicial


    iteration = 0
    last_update_time = 0.0
    last_update_iteration = 0
    large_offset = 0

    start_time = time.time()
    run = True
    while run:
        iteration += 1
        print(f"Interação: {iteration}")

        # Evolves one iteration.
        brkga.evolve()

        # Checks the current results and holds the best.
        fitness = brkga.get_best_fitness()
        print(f"Fitness: {fitness}")
        if fitness > best_cost:
            last_update_time = time.time() - start_time
            update_offset = iteration - last_update_iteration

            if large_offset < update_offset:
                large_offset = update_offset

            last_update_iteration = iteration
            best_cost = fitness
            best_chromosome = brkga.get_best_chromosome()

            print(f"* {iteration} | {best_cost:.0f} | {last_update_time:.2f}")
        # end if

        # TODO (ceandrade): implement path relink calls here.
        # Please, see Julia version for that.

        iter_without_improvement = iteration - last_update_iteration

        # Check stop criteria.
        run = not (
            num_generations == iteration
        )
    total_elapsed_time = time.time() - start_time
    total_num_iterations = iteration

    print(f"[{datetime.now()}] End of optimization\n")

    print(f"Total number of iterations: {total_num_iterations}")
    print(f"Last update iteration: {last_update_iteration}")
    print(f"Total optimization time: {total_elapsed_time:.2f}")
    print(f"Last update time: {last_update_time:.2f}")
    print(f"Large number of iterations between improvements: {large_offset}")

    imprimir_saida(best_chromosome, instance_file, qtd_grupos, qtd_variaveis, qtd_min_por_grupo)


###############################################################################

if __name__ == "__main__":
    main()
