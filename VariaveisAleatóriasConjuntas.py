import time
import numpy as np
import matplotlib.pyplot as plt

distributedMatrix=[[0.1,0.05,0.04],
                    [0.12,0.15,0.16],
                    [0.14,0.06,0.1],
                    [0.03,0.01,0.04]]

bigMatrix=[[0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01],
            [0.01,0.01,0.01,0.01,0.01]]

nsamples = 10000
# Váriável Global para que todas as funções usem os mesmos números aleatoriamente gerados
guesses = np.random.sample(nsamples)
#============================================================================
def calculateMarginal(matrix):
    heigth = len(matrix)
    width = len(matrix[0])
    xMarginal = np.zeros(width)
    yMarginal = np.zeros(heigth)
    for y in range(heigth):
        for x in range(width):
            xMarginal[x] += matrix[y][x]
            yMarginal[y] += matrix[y][x]
# Esta verificação da um erro por causa de como o Python usa o Float
# Podendo retornar 0.99999999999999 mesmo com uma matriz correta
#     if (sum(xMarginal) != 1):
#         print("Erro! Probabilidades não somam 1")
    else:
        return (xMarginal, yMarginal)
#============================================================================
def getLimits(matrix, x, y):
    heigth = len(matrix)
    width = len(matrix[0])
    lowerLimit = 0
    higherLimit = 0
    for currentY in range(heigth):
        for currentX in range(width):
            higherLimit += matrix[currentY][currentX]
            if (currentX == x and currentY == y):
                return (lowerLimit, higherLimit)
            lowerLimit += matrix[currentY][currentX]
    return (lowerLimit, higherLimit)

# Probabilidade de um ponto especifico
def randomGuesses1(matrix, x, y):
    limits = getLimits(matrix,x,y)
    timesChoosen = 0
    for guess in guesses:
        if limits[0] <= guess and guess <= limits[1]:
            timesChoosen += 1
    return timesChoosen/nsamples
#============================================================================
def limitsList(matrix):
    heigth = len(matrix)
    width = len(matrix[0])
    lastLimit = 0
    limitList = []
    for y in range(heigth):
        for x in range(width):
            lastLimit += matrix[y][x]
            limitList.append(lastLimit)
    return limitList

# Geração de varíaveis aleatórias conjuntas
def randomGuesses2(matrix):
    startTime = time.time()
    limits = limitsList(matrix)
    numberOfElementsInMatrix = len(limits)
    timesChoosen = np.zeros(numberOfElementsInMatrix)
    for guess in guesses:
        for currentIndex in range(numberOfElementsInMatrix):
            if limits[currentIndex] >= guess:
                timesChoosen[currentIndex] += 1
                break
    eachPointProbability = []
    for point in timesChoosen:
        eachPointProbability.append(point/nsamples)
    print(f'Time for randomGuesses2: {time.time() - startTime}')
    return eachPointProbability
#============================================================================
def calculateYMarginalLimits(matrix):
    heigth = len(matrix)
    width = len(matrix[0])
    marginals = calculateMarginal(matrix)
    lastLimit = 0
    marginalLimitList = []
    for i in range(heigth):
        marginalLimitList.append(lastLimit)
        lastLimit += marginals[1][i]
    marginalLimitList.append(lastLimit)
    return marginalLimitList

def calculateChoosenIndex(matrix, matrixMarginalLimitList, heigth, width, guess):
    for y in range(heigth):
        if matrixMarginalLimitList[y+1] >= guess:
            lastMarginalLimit = matrixMarginalLimitList[y]
            for x in range(width):
                lastMarginalLimit += matrix[y][x]
                if lastMarginalLimit >= guess:
                    return (width*y)+x

# Geração de varíaveis aleatórias conjuntas usando distribuição marginal
def randomGuesses3(matrix):
    startTime = time.time()
    heigth = len(matrix)
    width = len(matrix[0])
    numberOfElementsInMatrix = heigth*width
    marginalLimitList = calculateYMarginalLimits(matrix)
    timesChoosen = np.zeros(numberOfElementsInMatrix)
    for guess in guesses:
        timesChoosen[calculateChoosenIndex(matrix,marginalLimitList,heigth,width,guess)] += 1
    eachPointProbability = []
    for point in timesChoosen:
        eachPointProbability.append(point/nsamples)
    print(f'Time for randomGuesses3: {time.time() - startTime}')
    return eachPointProbability
#============================================================================
# Cálculo da probabilidade de ser escolhido um y uma vez que um x já ter 
# sido escolhido ou vice-versa
def calculateFixedProbability(matrix, x, y, xFixed = True):
    marginals = calculateMarginal(matrix)
    if xFixed:
        return matrix[y][x]/marginals[0][x]
    else:
        return matrix[y][x]/marginals[1][y]

# Cálculo da probabilidade de ser escolhido um y uma vez que um x já ter 
# sido escolhido ou vice-versa para todos os pontos da matriz 
def calculateEveryFixedProbability(matrix):
    heigth = len(matrix)
    width = len(matrix[0])
    for x in range(width):
        for y in range(heigth):
            print(f'({x},{y}) -> {calculateFixedProbability(matrix,x,y)}, {calculateFixedProbability(matrix,x,y, False)}')
            
#============================================================================
print('\n======= randomGuesses1 =======\n')
print(f'randomGuesses1(10000,a,1,0) => {randomGuesses1(distributedMatrix,1,0)}\n')
print('\n======= randomGuesses2 =======\n')
print(f'randomGuesses2(10000,a) => {randomGuesses2(distributedMatrix)}\n')
print('\n======= randomGuesses3 =======\n')
randomGuesses3Answer = randomGuesses3(distributedMatrix)
print(f'randomGuesses3(10000,a) => {randomGuesses3Answer}\n')
print('\n======= Graph for randomly Genereted vector =======\n')
a = np.asarray(randomGuesses3Answer)
plt.imshow(np.matrix(a.reshape(4,3)))
plt.colorbar()
plt.show()
print('======= CalculatecalculateEveryFixedProbability =======')
calculateEveryFixedProbability(distributedMatrix)
print('\n======= randomGuesses times for bigMatrix =======')
nsamples = 100000
guesses = np.random.sample(nsamples)
randomGuesses2(bigMatrix)
randomGuesses3Answer = randomGuesses3(bigMatrix)

print('\n======= Graph for randomly Genereted vector with big Matrix =======\n')

a = np.asarray(randomGuesses3Answer)
plt.imshow(np.matrix(a.reshape(20,5)))
plt.colorbar()
plt.show()