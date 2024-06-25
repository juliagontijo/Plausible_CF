import random as rnd
from scipy.spatial import distance
import pandas as pd
from sklearn import preprocessing
import warnings
import sys
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
from matplotlib.colors import ListedColormap

from dijsktra_algorithm import Graph, dijsktra_toall, dijsktra_tosome, dijsktra_to_some_final_nodes, dijsktra_tosome2

#Used for ordering evaluations
class individual:
    def __init__(self, index, score, distance, num_changes, aval_norm, dist_norm, predict_proba, density):
        self.index = index #Indicates the instance's position in the dataframe
        self.score = score #Indicates the score in relation to the proximity of the class boundary
        self.distance = distance #Indicates the distance from the original instance
        self.num_changes = num_changes #Indicates the number of changes for class change
        self.aval_norm = aval_norm #Indicates the final fitness with standardized metrics
        self.dist_norm = dist_norm #Indicates the normalized distance (distance and number of changes)
        self.predict_proba = predict_proba #Indicates de individual's class
        self.density = density
    def __repr__(self):
        return repr((self.index, self.score, self.distance, self.num_changes, self.aval_norm, self.dist_norm, self.predict_proba, self.density))

class counter_change:
    def __init__(self, column, value):
        self.column = column 
        self.value = value
    def __eq__(self, other):
        if self.column == other.column and self.value == other.value:
            return True
        else:
            return False
    def __repr__(self):
        return repr((self.column, self.value))    

#Used to generate a random value in the mutation operation
class feature_range:
    def __init__(self, column, col_type, min_value, max_value):
        self.column = column 
        self.col_type = col_type
        self.min_value = min_value
        self.max_value = max_value

    #Returns a random value to perform mutation operation
    def get_random_value(self):
        if self.col_type == 'int64' or self.col_type == 'int' or self.col_type == 'int16' or self.col_type == 'int8' or (self.col_type == 'uint8'):
            value = rnd.randint(self.min_value, self.max_value)
        else:  
            value = round(rnd.uniform(self.min_value, self.max_value), 2)
        return value
    
    #Checks if the attribute has only one value.
    def unique_value(self):
        if self.min_value != self.max_value:
            return False
        else:  
            return True    

    def __repr__(self):
        return repr((self.column, self.col_type, self.min_value, self.max_value)) 
        
class CSSE(object):
    
    def __init__(self, input_dataset, model, static_list = [], K = 3, num_gen = 30, pop_size = 100, per_elit = 0.1, cros_proba = 0.8, mutation_proba = 0.1, L1 = 1, L2 = 1, L3 = 1):
        #User Options
        self.static_list = static_list #List of static features
        self.K = K #Number of counterfactuals desired
        #Model
        self.input_dataset = input_dataset
        self.model = model
        #GA Parameters
        self.num_gen = num_gen
        self.pop_size = pop_size
        self.per_elit = per_elit
        self.cros_proba = cros_proba
        self.mutation_proba = mutation_proba
        #Objective function parameters
        self.L1 = L1 #weight assigned the distance to the original instance
        self.L2 = L2 #weight assigned the number of changes needed in the original instance
        self.L3 = L3 #weight assigned the density to the original instance
    
    #Get which index in the SHAP corresponding to the current class
    def getBadClass(self):   
        if self.current_class == self.model.classes_[0]:
            ind_cur_class = 0
        else:
            ind_cur_class = 1
        
        return ind_cur_class
    
    #Gets the valid values range for each feature
    def getFeaturesRange(self):
        features_range = []
       
        for i in range (0, self.input_dataset.columns.size):
            col_name = self.input_dataset.columns[i]
            col_type = self.input_dataset[col_name].dtype
            min_value = min(self.input_dataset[col_name])
            max_value = max(self.input_dataset[col_name])
            
            feature_range_ind = feature_range(col_name, col_type, min_value, max_value)
            features_range.append(feature_range_ind)
        
        return features_range
       
    def getMutationValue(self, currentValue, index, ind_feature_range):
        new_value = ind_feature_range.get_random_value()
        
        while currentValue == new_value:
            new_value = ind_feature_range.get_random_value()
        
        return new_value
    
    def equal(self, individual, population):
        aux = 0
        for i in range ( 1, len(population)):
            c = population.loc[i].copy()
            dst = distance.euclidean(individual, c)
            if dst == 0:
                aux = 1
        
        return aux

    def getPopInicial (self, df, features_range): 
        #The reference individual will always be in the 0 position of the df - so that it is normalized as well (it will be used later in the distance function)
        df.loc[0] = self.original_ind.copy()
        
        #Counting numbers of repeated individuals
        number_repetitions = 0
        
        #One more position is used because the zero position was reserved for the reference individual
        while len(df) < self.pop_size + 1:
            #Draw a feature to change
            index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            while df.columns[index_a] in self.static_list:
                index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            
            if not features_range[index_a].unique_value():
                #Mutation
                mutant = self.original_ind.copy()

                new_value =  self.getMutationValue(mutant.iloc[index_a], index_a, features_range[index_a])
                mutant.iloc[index_a] = new_value

                ni = self.equal(mutant, df)
                if ni == 0:
                    df.loc[len(df)] = mutant.copy()
                else:
                    #Assesses whether the GA is producing too many repeated individuals.
                    number_repetitions = number_repetitions + 1
                    if number_repetitions == 2*self.pop_size:
                        self.pop_size = round(self.pop_size - self.pop_size*0.1)
                        self.mutation_proba = self.mutation_proba + 0.1
                        #print('Adjusting population size...', self.pop_size)
                        number_repetitions = 0
    
    #Complete the standardized proximity and similarity assessments for each individual
    def getNormalEvaluation(self, evaluation, aval_norma):
        # print("aval_norma", aval_norma)
        scaler2 = preprocessing.MinMaxScaler()
        aval_norma2 = scaler2.fit_transform(aval_norma)
    
        i = 0
        while i < len(evaluation):
            evaluation[i].aval_norm = self.L1*aval_norma2[i,0] + self.L2*aval_norma2[i,1] + aval_norma2[i,2] + self.L3*aval_norma2[i,3]
            evaluation[i].dist_norm = self.L1*aval_norma2[i,0] + self.L2*aval_norma2[i,1]
        
            i = i + 1
    
    def numChanges(self, ind_con):
        num = 0
        for i in range(len(self.original_ind)):
            if self.original_ind[i] != ind_con[i]:
                num = num + 1
        
        return num
        
    def fitness(self, population, evaluation, ind_cur_class, density_computation):
        def getProximityEvaluation (proba):
            #Penalizes the individual who is in the negative class
            if proba < 0.5:
                predict_score = 0
            else:
                predict_score= proba
             
            return predict_score
               
        #Calculates similarity to the original instance
        def getEvaluationDist (ind, X_train_minmax):
            #Normalizes the data so that the different scales do not bias the distance
            a = X_train_minmax[0]
            b = X_train_minmax[ind]
            dst = distance.euclidean(a, b)
  
            return dst

        #Calculates density on midpoint to oriinal instance
        def getDensityEvaluation (ind, X_train_minmax, density_computation):
            density = density_computation.get_kde_density_2points(X_train_minmax[ind], X_train_minmax[0], 1)
  
            return density
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        
            predict_proba = self.model.predict_proba(population)
                    
        #Calculating the distance between instances
        scaler = preprocessing.MinMaxScaler()
        X_train_minmax = scaler.fit_transform(population)
    
        i = 0
        aval_norma = [] 
        while i < len(population):
            proximityEvaluation = getProximityEvaluation(predict_proba[i, ind_cur_class])
            evaldist = getEvaluationDist(i, X_train_minmax)
            #The original individual is in the 1st position
            evaldensity = getDensityEvaluation(i, X_train_minmax, density_computation)
            numChanges = self.numChanges(population.loc[i])

            # print("Returned dens value:", evaldensity)
            # print("Type of returned dens:", type(evaldensity))
            # if isinstance(evaldensity, np.ndarray):
            #     print("Shape of returned dens:", evaldensity.shape)
            # else:
            #     print("Returned dens is a scalar and does not have a shape.")

            ind = individual(i, proximityEvaluation, evaldist, numChanges, 0, 0, predict_proba[i, ind_cur_class], evaldensity)
            aval_norma.append([evaldist, numChanges, proximityEvaluation, evaldensity])
            evaluation.append(ind)
            i = i + 1

        self.getNormalEvaluation(evaluation, aval_norma)
       
    #Given a counterfactual solution returns the list of modified columns
    def getColumns(self, counter_solution):
        colums = []
        for j in range (0, len(counter_solution)):
            colums.append(counter_solution[j].column)
        
        return colums      
             
    #Checks if the new solution is contained in the solutions already found
    def contained_solution(self, original_instance, current_list, current_column_list, new_solution, new_column_solution):
        contained = False
        for i in range (0, len(current_list)):              
            if set(current_column_list[i]).issubset(new_column_solution):
                for j in range (0, len(current_list[i])):
                    pos = new_column_solution.index(current_list[i][j].column)
                    distancia_a = abs(original_instance[current_list[i][j].column] - current_list[i][j].value)
                    distancia_b = abs(original_instance[current_list[i][j].column] - new_solution[pos].value)
                    if distancia_b >= distancia_a:
                        contained = True

        return contained
      
    def elitism(self, evaluation, df, parents):
         
        num_elit = round(self.per_elit*self.pop_size)
        
        aval = []
        aval = evaluation.copy()
        aval.sort(key=lambda individual: individual.aval_norm)
        
        #contrafactual_ind = pd.DataFrame(columns=self.input_dataset.columns)
        solution_list = []
        solution_colums_list = []
        
        i = 0
        numContraf = 0
        while i < len(aval) and numContraf <= num_elit + 1:
            #Checks if the example belongs to the counterfactual class
            if aval[i].predict_proba < 0.5:
                ind_changes = []
                ind_colums_change = []
         
                #Gets counterfactual example change list
                ind_changes = self.getChanges(aval[i].index, parents)
                #Generates the list of columns modified in the counterfactual to check if there is already a solution with that set of columns
                ind_colums_change = self.getColumns(ind_changes)
                
                if ind_colums_change not in solution_colums_list:
                    #Check if one solution is a subset of the other
                    if not self.contained_solution(self.original_ind, solution_list, solution_colums_list, ind_changes, ind_colums_change):
                        #Include counterfactual in the list of examples of the final solution                    
                        df.loc[len(df)] = parents.iloc[aval[i].index].copy()                     
                                
                        #Add to the list of solutions (changes only)       
                        solution_list.append(ind_changes)
                        #Used to compare with the next counterfactuals (to ensure diversity)
                        solution_colums_list.append(ind_colums_change)
                                        
                        numContraf = numContraf + 1
                      
            i = i + 1
        return solution_list
    
    def roulette_wheel(self, evaluation):
        summation = 0
        #Performs roulette wheel to select parents who will undergo genetic operations
        for i in range (1, len(evaluation)): 
            summation = summation + 1/evaluation[i].aval_norm
    
        roulette = rnd.uniform( 0, summation )
    
        roulette_score = 1/evaluation[1].aval_norm
        i = 1
        while roulette_score < roulette:
            i += 1
            roulette_score += 1/evaluation[i].aval_norm
        
        return i
            
    def crossover (self, df, parents, evaluation, number_cross_repetitions):
        child = []
            
        corte = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            
        index1 = self.roulette_wheel(evaluation)
        index2 = self.roulette_wheel(evaluation)
        
        ind_a = parents.iloc[index1].copy()
        ind_b = parents.iloc[index2].copy()
            
        crossover_op = rnd.random()
        if crossover_op <= self.cros_proba:
            child[ :corte ] = ind_a[ :corte ].copy()
            child[ corte: ] = ind_b[ corte: ].copy()
        else:
            child = ind_a.copy()
        
        ni = self.equal(child, df)
        if ni == 0:
            df.loc[len(df)] = child.copy()
        else:
            #Assesses whether the GA is producing too many repeated individuals.
            number_cross_repetitions = number_cross_repetitions + 1
            if number_cross_repetitions == self.pop_size:
                self.pop_size = round(self.pop_size - self.pop_size*0.1)
                self.mutation_proba = self.mutation_proba + 0.1
                #print('Adjusting population size...', self.pop_size)
                number_cross_repetitions = 0
        #    print('repeated')
        return number_cross_repetitions
                       
    def mutation (self, df, individual_pos, features_range):
        ni = 1
        #Does not allow repeated individual
        while ni == 1:
            #Draw a feature to change
            index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            while df.columns[index_a] in self.static_list:
                index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            
            if not features_range[index_a].unique_value():
                #Mutation
                mutant = df.iloc[individual_pos].copy()
            
                #Draw the value to be changed
                new_value =  self.getMutationValue(mutant.iloc[index_a], index_a, features_range[index_a])  
                mutant.iloc[index_a] = new_value

                ni = self.equal(mutant, df)
                if ni == 0:
                    df.loc[individual_pos] = mutant.copy()
                #else:
                #    print('repeated')
     
    def getChanges(self, ind, dfComp):
        changes = []
        
        for i in range (len(dfComp.iloc[ind])):
            if self.original_ind[i] != dfComp.loc[ind][i]:
                counter_change_ind = counter_change(dfComp.columns[i], dfComp.loc[ind][i])
                changes.append(counter_change_ind)

        return changes
    
    #Generates the solution from the final population
    def getContrafactual(self, df, aval):
        
        contrafactual_ind = pd.DataFrame(columns=self.input_dataset.columns)
        solution_list = []
        solution_colums_list = []
        
        i = 0
        numContraf = 0
        while i < len(aval) and numContraf < self.K:
            #Checks if the example belongs to the counterfactual class
            if aval[i].predict_proba < 0.5:
                ind_changes = []
                ind_colums_change = []
         
                #Gets counterfactual example change list
                ind_changes = self.getChanges(aval[i].index, df)
                #Generates the list of columns modified in the counterfactual to check if there is already a solution with that set of columns
                ind_colums_change = self.getColumns(ind_changes)
                
                if ind_colums_change not in solution_colums_list:
                    #Check if one solution is a subset of the other
                    if not self.contained_solution(self.original_ind, solution_list, solution_colums_list, ind_changes, ind_colums_change):
                        #Include counterfactual in the list of examples of the final solution
                        contrafactual_ind.loc[len(contrafactual_ind)] = df.iloc[aval[i].index].copy()
                                
                        #Add to the list of solutions (changes only)       
                        solution_list.append(ind_changes)
                        #Used to compare with the next counterfactuals (to ensure diversity)
                        solution_colums_list.append(ind_colums_change)
                                        
                        numContraf = numContraf + 1
                        #print('solution_list ', solution_list)
                    #else:
                        #print('is contained ', ind_changes)
                #else:
                    #print('repeated ', ind_changes)
                      
            i = i + 1

        return contrafactual_ind, solution_list   
    
    def printResults(self, solution):
        print("Result obtained")
        if len(solution) != 0:
            for i in range(0, len(solution)): 
                print("\n")
                print(f"{'Counterfactual ' + str(i + 1):^34}")
                for j in range(0, len(solution[i])): 
                    print(f"{str(solution[i][j].column):<29} {str(solution[i][j].value):>5}")
        else:
            print('Solution not found. It may be necessary to adjust the parameters for this instance.')
                                                 
    def explain(self, original_ind, current_class, X, y):
        self.original_ind = original_ind #Original instance
        #self.ind_cur_class = ind_cur_class #Index in the shap corresponds to the original instance class
        self.current_class = current_class #Original instance class
        
        ind_cur_class = self.getBadClass()
    
        #Gets the valid values range of each feature
        features_range = []
        features_range = self.getFeaturesRange()

        #The DataFrame df will have the current population
        df = pd.DataFrame(columns=self.input_dataset.columns)

        #Calculate dataset space density
        dens = np.logspace(-5, -2, 5)[0]
        density_computation = density(self.model, weight_function= lambda x: -np.log(x), distance_threshold=1,
                density_threshold=dens, howmanypaths=10)
        density_computation.generateSpaceDensity(X,y)
        
        #Generates the initial population with popinitial mutants        
        self.getPopInicial(df, features_range)

        # PLOTTING INITIAL POP
        # UNCOMMENT WHEN PLOTTING 2D DATASET COUNTERFACTUAL
        def pred_func(x):
                return self.model.predict_proba(x)[:, 1]
        
        plot_decision_boundary_initial(X, y, pred_func)
        # STOP UNCOMMENT WHEN PLOTTING 2D DATASET COUNTERFACTUAL

        
        for g in tqdm(range(self.num_gen), desc= "Processing..."):
            #To use on the parents of each generation
            parents = pd.DataFrame(columns=self.input_dataset.columns)
    
            #Copy parents to the next generation
            parents = df.copy()
            
            #df will contain the new population
            df = pd.DataFrame(columns=self.input_dataset.columns)
            
            evaluation = []                         
                   
            #Assessing generation counterfactuals
            self.fitness(parents, evaluation, ind_cur_class, density_computation)
            
            #The original individual will always be in the 0 position of the df - So that it is normalized too (it will be used later in the distance function)
            df.loc[0] = self.original_ind.copy()
            
            #Copies to the next generation the per_elit best individuals
            self.elitism(evaluation, df, parents)
            
            number_cross_repetitions = 0
            while len(df) < self.pop_size + 1: #+1, as the 1st position is used to store the reference individual
                number_cross_repetitions = self.crossover(df, parents, evaluation, number_cross_repetitions)
                
                mutation_op = rnd.random()
                if mutation_op <= self.mutation_proba:
                    self.mutation(df, len(df) - 1, features_range)
            
            # PLOTTING EACH GENERATION
            # UNCOMMENT WHEN PLOTTING 2D DATASET COUNTERFACTUAL
                    
            #Getting the counterfactual set
            contrafactual_set_aux = pd.DataFrame(columns=self.input_dataset.columns)
            contrafactual_set_aux, solution_list_aux = self.getContrafactual(df, evaluation) 

            # def pred_func(x):
            #         return self.model.predict_proba(x)[:, 1]
            
            #plot_decision_boundary(X, y, pred_func, contrafactual_set_aux)
            # STOP UNCOMMENT WHEN PLOTTING 2D DATASET COUNTERFACTUAL
                 

        evaluation = []
    
        #Evaluating the latest generation
        self.fitness(df, evaluation, ind_cur_class, density_computation)
    
        #Order the last generation by distance to the original instance     
        evaluation.sort(key=lambda individual: individual.aval_norm)     
        
        #Getting the counterfactual set
        contrafactual_set = pd.DataFrame(columns=self.input_dataset.columns)
        contrafactual_set, solution_list = self.getContrafactual(df, evaluation)    

        # UNCOMMENT WHEN PLOTTING 2D DATASET COUNTERFACTUAL
        # def pred_func(x):
        #         return self.model.predict_proba(x)[:, 1]
        
        plot_decision_boundary(X, y, pred_func, contrafactual_set)
        # STOP UNCOMMENT WHEN PLOTTING 2D DATASET COUNTERFACTUAL

        x_withContrafactuals = np.concatenate([contrafactual_set, pd.DataFrame(X)])
        newy = pd.Series(np.zeros(len(contrafactual_set)))
        y_withContrafactuals = np.concatenate([newy, y])

        density_computation.generateSpaceDensity(x_withContrafactuals, y_withContrafactuals)
        density_computation.getGraph()
        p = density_computation.compute_path(original_ind, 0, contrafactual_set)
                 
        return contrafactual_set, solution_list


def plot_decision_boundary(X, y, func, contrafactual_set):
    h = 0.1
    xmin, ymin = np.min(X, axis=0)
    xmax, ymax = np.max(X, axis=0)

    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, h),
        np.arange(ymin, ymax, h)
        )

    cm = plt.cm.RdBu
    cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
    
    newx = np.c_[xx.ravel(), yy.ravel()]
        
    fig, ax = plt.subplots()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Z = clf.predict_proba(newx)[:, 1]
    Z = func(newx)
    Z = Z.reshape(xx.shape)

    v=contour_plot = ax.contourf(
        xx, yy,
        Z, 
        levels=20,
        cmap=cm, 
        alpha=.8)
    
    ax.scatter(X[:, 0], X[:, 1], 
                c=y, 
                cmap=cm_bright,
                edgecolors='k',
                zorder=1)    

     # Plot the subject (first position in X) as a green dot
    ax.scatter(X[0, 0], X[0, 1], color='green', edgecolors='k', zorder=2, label='Subject')

    # Plot counterfactual set individuals as black dots
    ax.scatter(contrafactual_set.iloc[:, 0], contrafactual_set.iloc[:, 1], color='black', marker='o', label='Counterfactuals')


    ax.grid(color='k', 
            linestyle='-', 
            linewidth=0.50, 
            alpha=0.75)

    plt.colorbar(v, ax=ax)

    # Adding a legend to identify the subject and counterfactuals
    ax.legend()

    plt.show()

def plot_decision_boundary_initial(X, y, func):
    h = 0.1
    xmin, ymin = np.min(X, axis=0)
    xmax, ymax = np.max(X, axis=0)

    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, h),
        np.arange(ymin, ymax, h)
        )

    cm = plt.cm.RdBu
    cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
    
    newx = np.c_[xx.ravel(), yy.ravel()]
        
    fig, ax = plt.subplots()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Z = clf.predict_proba(newx)[:, 1]
    Z = func(newx)
    Z = Z.reshape(xx.shape)

    v=contour_plot = ax.contourf(
        xx, yy,
        Z, 
        levels=20,
        cmap=cm, 
        alpha=.8)
    
    ax.scatter(X[:, 0], X[:, 1], 
                c=y, 
                cmap=cm_bright,
                edgecolors='k',
                zorder=1)    

     # Plot the subject (first position in X) as a green dot
    ax.scatter(X[0, 0], X[0, 1], color='green', edgecolors='k', zorder=2, label='Subject')

    ax.grid(color='k', 
            linestyle='-', 
            linewidth=0.50, 
            alpha=0.75)

    plt.colorbar(v, ax=ax)

    # Adding a legend to identify the subject and counterfactuals
    ax.legend()
    ax.set_title("Initial Population")

    plt.show()

def plot_decision_boundary_final(X, y, func):
    h = 0.1
    xmin, ymin = np.min(X, axis=0)
    xmax, ymax = np.max(X, axis=0)

    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, h),
        np.arange(ymin, ymax, h)
        )

    cm = plt.cm.RdBu
    cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
    
    newx = np.c_[xx.ravel(), yy.ravel()]
        
    fig, ax = plt.subplots()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Z = clf.predict_proba(newx)[:, 1]
    Z = func(newx)
    Z = Z.reshape(xx.shape)

    v=contour_plot = ax.contourf(
        xx, yy,
        Z, 
        levels=20,
        cmap=cm, 
        alpha=.8)
    
    ax.scatter(X[:, 0], X[:, 1], 
                c=y, 
                cmap=cm_bright,
                edgecolors='k',
                zorder=1)

    ax.grid(color='k', 
            linestyle='-', 
            linewidth=0.50, 
            alpha=0.75)

    plt.colorbar(v, ax=ax)
    return ax



def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_edges(kernel):
    edges = []

    n_samples = kernel.shape[0]
    for i in range(n_samples):
        for j in range(n_samples):
            if kernel[i, j] != 0 :
                edges.append([i, j, kernel[i, j]])
    return edges

def plot_path(X, path, ax, color='lightgreen', extra_point=None):
    if X.shape[1] != 2:
        return 0

    n_nodes = len(path)
    if isinstance(extra_point, np.ndarray):
        ax.plot([X[-1, 0], extra_point[0]],
                [X[-1, 1], extra_point[1]],
                'k', alpha=0.50)
        ax.scatter(extra_point[0], extra_point[1],
                    color='k',
                    marker='o',
                    facecolors='lightyellow',
                    edgecolors='lightyellow',
                    alpha = 0.80,
                    zorder=1,
                    s=250)
        
    args = {'color': 'lightgreen',
            'marker': 'x',
            's': 100}
    
    for idx in range(n_nodes-1):
        i = int(path[idx])
        j = int(path[idx + 1])
        ax.plot(X[[i, j], 0], X[[i, j], 1], 'k', alpha=0.50)
    
    ax.scatter(X[path[-1], 0], X[path[-1], 1],
                color='k',
                marker='o',
                facecolors=color,
                edgecolors=color,
                alpha = 0.50,
                zorder=2,
                s=150)

def plot_paths(X, howmanypaths, ax, all_paths):
    counter = 0
    colors=cm.Greens(np.linspace(0,1,howmanypaths))

    for idx, item in enumerate(all_paths):
        if counter > howmanypaths - 1:
            break
        path = item[-1]
        #if method in ['kde']:
        plot_path(X, path, ax, colors[counter])
        # else:
        #     plot_path(X, path, ax, colors[counter])
        counter += 1   




class density(object):
    def __init__(self, predictor, weight_function = None, 
        distance_threshold=None,
        density_threshold=None,
        prediction_threshold= None,
        howmanypaths = None,
        undirected = False):
        
        self.undirected = undirected

        if not hasattr(predictor, 'predict_proba'):
            raise ValueError('Predictor needs to have attribute: \'predict proba\'')
        else:
            self.predictor = predictor

        if weight_function is None:
            self.weight_function = lambda x: -np.log(x)
        else:
            self.weight_function = weight_function

        if density_threshold is None:
            self.density_threshold = 1e-5
        else:
            self.density_threshold = density_threshold

        if distance_threshold is None:
            self.distance_threshold = 'a'
        else:
            self.distance_threshold = distance_threshold

        if prediction_threshold is None:
            self.prediction_threshold = 0.60    
        else:
            self.prediction_threshold = prediction_threshold

        if howmanypaths is None:
            self.howmanypaths = 5
        else:
            self.howmanypaths = howmanypaths

    def generateSpaceDensity(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = self.X.shape
        if self.n_samples != self.y.shape[0]:
            raise ValueError('Inconsistent dimensions')
        self.predictions = self.predictor.predict_proba(X)
        self.get_kde()
        #self.kernel = self.get_kernel()
        #self.fit_graph()

    def get_kde(self):

        #bandwidths = np.logspace(-1, 3, 20)  
        bandwidths = np.logspace(-2, 0, 20)  

        #bandwidths = [0.65]
        grid = GridSearchCV(
            KernelDensity(kernel='gaussian'),
            {'bandwidth': bandwidths},
            # cv=4,
        )    
        grid.fit(deepcopy(self.X))
        self.density_estimator = grid.best_estimator_

    def get_kde_density_2points(self, ind, original_ind, mode):
        dens=0
        density_scorer = self.density_estimator.score_samples
        
        #v0 = ind.reshape(-1, 1)
        #v1 = original_ind.reshape(-1, 1)
            #CHECK WHAT THIS CONDITIONS IS ABOUT
            #if not self.check_conditions(v0, v1):
                #continue
        dist = distance.euclidean(ind, original_ind)
        midpoint = (ind + original_ind)/2
        density = density_scorer(midpoint.reshape(1, -1))
        lala = np.exp(density)
        #QUESTION: Should we take the distance in count here? Or just return the density itself?
        if mode == 1:
            dens = self.weight_function(np.exp(density)) * dist
        else:
            dens = self.weight_function(sigmoid(density)) * dist

        # print("Density-Distance Tradeoff Value (dens):", dens)
        # print("Type of dens:", type(dens))
        # if isinstance(dens, np.ndarray):
        #     print("Shape of dens:", dens.shape)
        # else:
        #     print("dens is a scalar and does not have a shape.")
        
        gypsy = np.exp(dens)
        return dens[0]
    
    def getGraph(self):
        self.kernel = self.get_kernel()
        self.fit_graph()

    def get_kernel(self):
        density_scorer = self.density_estimator.score_samples
        kernel = self.get_weights_kde(density_scorer, 1)
        self.kernel = kernel
        return kernel

    def get_weights_kde(self, density_scorer,
                    mode):
        k = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            for j in range(self.n_samples):#range(i):
                v1 = self.X[j, :].reshape(-1, 1)
                # if not self.check_conditions(v0, v1):
                #     continue
                dist = np.linalg.norm(v0 - v1) #euclidian distance
                if dist <= self.distance_threshold:
                    midpoint = (v0 + v1)/2
                    density = density_scorer(midpoint.reshape(1, -1))
                    if mode == 1:
                        k[i, j] = self.weight_function(np.exp(density)) * dist
                    else:
                        k[i, j] = self.weight_function(sigmoid(density)) * dist
                else:
                    k[i, j] = 0
        return k
    
    def fit_graph(self):        
        self.graph = Graph(undirected=self.undirected)
        edges = get_edges(self.kernel)
        for edge in edges:
            self.graph.add_edge(*edge) 

    def compute_path(
        self, 
        starting_point,
        target_class,
        contrafactual_set,
        plot = True
        #individual_edge_conditions=None
        ):
        #self.individual_edge_conditions = individual_edge_conditions
        if self.n_features != 2:
            plot = False
        self.target_class = target_class

        starting_point_index = np.where((self.X == starting_point.to_numpy()).all(axis=1))[0][0]
        self.candidate_targets = contrafactual_set.copy()
        # t0 = np.where(self.predictions >= self.prediction_threshold)[0]
        # t1 = np.where(self.y == self.target_class)[0]
        # kde = np.exp(self.density_estimator.score_samples(deepcopy(self.X))) 
        # t2 = np.where(kde >= self.density_threshold)[0]
        # self.candidate_targets = list(set(t0).intersection(set(t1)).intersection(set(t2)))
            
        #if self.individual_edge_conditions is None:
        # dist, paths = dijsktra_toall(
        #                         deepcopy(self.graph), 
        #                         starting_point_index
        #                         )


        self.personal_kernel = self.kernel
        self.personal_graph = Graph()
        edges = get_edges(self.personal_kernel)
        for edge in edges:
            self.personal_graph.add_edge(*edge) 
        dist, paths = dijsktra_tosome2(self.personal_kernel, 
                                          starting_point_index, 
                                          self.candidate_targets)
        
        # self.personal_kernel = self.kernel
        # self.personal_graph = Graph()
        # edges = get_edges(self.personal_kernel)
        # for edge in edges:
        #     self.personal_graph.add_edge(*edge) 
        # dist, paths = dijsktra_tosome(self.personal_graph, 
        #                                   starting_point_index, 
        #                                   self.candidate_targets)
        
        # else:
        #     self.personal_kernel = self.modify_kernel(self.kernel)
        #     self.personal_graph = Graph()
        #     edges = get_edges(self.personal_kernel)
        #     for edge in edges:
        #         self.personal_graph.add_edge(*edge) 
        #     dist, paths = dijsktra_tosome(self.personal_graph, 
        #                                   starting_point_index, 
        #                                   self.candidate_targets)
        
        if plot:
            def pred_func(x):
                return self.predictor.predict_proba(x)[:, 1]

            ax = plot_decision_boundary_final(self.X, self.y, pred_func)

        all_paths = []
        for item, path in paths.items():
            value, satisfied = self.condition(item)

            if satisfied:
                all_paths.append((item, self.X[item, :], dist[item], value, path))
        all_paths = sorted(all_paths, key=lambda x: x[2])
        
        if plot:
            plot_paths(self.X, self.howmanypaths, ax, all_paths)             
            
        return all_paths
    

    def condition(self, item):        
        pred = self.predictions[item, self.y[item]]
        if (self.y[item] == self.target_class
                and pred >= self.prediction_threshold):
            # if self.method == 'kde':
            kde = np.exp(self.density_estimator.score_samples(self.X[int(item), :].reshape(1, -1))) 
            if kde >= self.density_threshold:
                return (pred, kde), True
            # elif self.method in ['knn', 'egraph']:
            #     return (pred), True
        return 0, False



# def plot(self, x_train, y_train, csse_counterfactuals, original_ind):
    #     X = pd.concat([x_train, csse_counterfactuals])

    #     h = 0.1
    #     xmin, ymin = np.min(X.values, axis=0)
    #     xmax, ymax = np.max(X.values, axis=0)

    #     xx, yy = np.meshgrid(
    #         np.arange(xmin, xmax, h),
    #         np.arange(ymin, ymax, h)
    #         )

    #     cm = plt.cm.RdBu
    #     cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
        
    #     newx = np.c_[xx.ravel(), yy.ravel()]
            
    #     fig, ax = plt.subplots()
    #     ax.set_xlim(xmin, xmax)
    #     ax.set_ylim(ymin, ymax)
        
    #     # Z = clf.predict_proba(newx)[:, 1]
    #     Z = model.predict_proba(newx)
    #     Z = Z.reshape(xx.shape)

    #     v=contour_plot = ax.contourf(
    #         xx, yy,
    #         Z, 
    #         levels=20,
    #         cmap=cm, 
    #         alpha=.8)
        
    #     ax.scatter(X.values[:, 0], X.values[:, 1], 
    #                 c=y, 
    #                 cmap=cm_bright,
    #                 edgecolors='k',
    #                 zorder=1)

    #     ax.scatter(csse_counterfactuals.values[:, 0], csse_counterfactuals.values[:, 1], 
    #                 c='green', 
    #                 cmap=cm_bright,
    #                 edgecolors='k',
    #                 zorder=1)
        
    #     ax.scatter(original_instance.values[0, 0], original_instance.values[0, 1],
    #        c='k',
    #        edgecolors='k',
    #        zorder=1)


    #     ax.grid(color='k', 
    #             linestyle='-', 
    #             linewidth=0.50, 
    #             alpha=0.75)

    #     plt.colorbar(v, ax=ax)
