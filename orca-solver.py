from gurobipy import *
import numpy as np

def setObjective(model):
    objectiveExpression = LinExpr()
    individualVars = []
    for individual in range(numCaseIndiv + numControlIndiv):
        individualVars.append(model.addVar(lb=0, ub=1, vtype=GRB.BINARY))

    for var in range(numCaseIndiv):
        objectiveExpression += 1.0 / float(numCaseIndiv) * individualVars[var]

    for var in range(numControlIndiv):
        objectiveExpression += -1.0 / float(numControlIndiv) * individualVars[var + numCaseIndiv]

    model.setObjective(objectiveExpression, GRB.MAXIMIZE)
    return model, individualVars

def readInputFile(filename):
    import fileinput
    snpNames = []
    inputMatrix = []
    with fileinput.input(files=filename) as f:
        for line in f:
            if line.find('rs') > -1:
                snpInputLine = []
                for col in line.split():
                    if col.find('rs') > -1:
                        snpNames.append(col)
                    if col.find('A') > -1 or \
                       col.find('T') > -1 or \
                       col.find('G') > -1 or \
                       col.find('C') > -1:
                        snpInputLine.append(col)
                inputMatrix.append(snpInputLine)

    return snpNames, inputMatrix


def buildBinaryGwas(inputMatrix):
    numRegions = len(inputMatrix)
    binaryMatrix = np.array([])
    for region in range(numRegions):
        row = np.array([])
        firstAllele = inputMatrix[region][0][0]
        for i in range(numCaseIndiv + numControlIndiv):
            if inputMatrix[region][i][0] != inputMatrix[region][i][1]:
                row = np.append(row, 0)
                row = np.append(row, 1)
                row = np.append(row, 1)
                row = np.append(row, 0)
            elif firstAllele == inputMatrix[region][i][1]:
                row = np.append(row, 1)
                row = np.append(row, 1)
                row = np.append(row, 0)
                row = np.append(row, 0)
            else:
                row = np.append(row, 0)
                row = np.append(row, 0)
                row = np.append(row, 1)
                row = np.append(row, 1)
        binaryMatrix = np.concatenate((binaryMatrix, row))
    binaryMatrix = np.reshape(binaryMatrix,
                              (len(inputMatrix) * (numCaseIndiv + numControlIndiv), 4))

    G = np.array([])
    for indiv in range(numCaseIndiv + numControlIndiv):
        gJ = np.array([])
        step = numCaseIndiv + numControlIndiv
        end = numRegions * step
        for region in range(indiv, end, step):
            gJ = np.append(gJ, binaryMatrix[region])
        G = np.concatenate((G, gJ))

    G = np.reshape(G, (numCaseIndiv + numControlIndiv, 4 * numRegions))
    return G

def addConstraints(initialLPModel, G, individualVars, desiredPatternSize):
    encodedGwasVars = []
    s = len(G[0])
    numIndiv = numCaseIndiv + numControlIndiv

    for region in range(s):
        encodedGwasVars.append(initialLPModel.addVar(lb=0, ub=1, vtype=GRB.BINARY))

    constraintL = LinExpr()
    for region in range(s):
        constraintL += encodedGwasVars[region]

    initialLPModel.addConstr(constraintL, GRB.EQUAL, desiredPatternSize)

    for indiv in range(numIndiv):
        constraintJ = LinExpr()
        constraintK = LinExpr()
        for region in range(s):
            constraintJ += G[indiv][region] * encodedGwasVars[region]
            constraintK += G[indiv][region] * encodedGwasVars[region]
        initialLPModel.addConstr(desiredPatternSize - constraintJ,
                                 GRB.LESS_EQUAL,
                                 s * (1 - individualVars[indiv]))
        initialLPModel.addConstr(constraintK - desiredPatternSize + 1,
                                 GRB.LESS_EQUAL,
                                 s * individualVars[indiv])

    return initialLPModel

def printSolution(lpModel, regionNames):
    solutionVars = lpModel.getVars()
    numCaseWithPattern = 0
    numControlWithPattern = 0
    markInSolution = []

    for j in range(numCaseIndiv):
        if solutionVars[j].X == 1:
            numCaseWithPattern += 1
    for j in range(numCaseIndiv, numCaseIndiv + numControlIndiv):
        if solutionVars[j].X == 1:
            numControlWithPattern += 1

    for i in range(numCaseIndiv + numControlIndiv,
                   4 * len(regionNames) + numCaseIndiv + numControlIndiv):
        if solutionVars[i].X == 1:
            markInSolution.append(1)
        else:
            markInSolution.append(0)

    print('case with pattern: ', numCaseWithPattern)
    print('control with pattern: ', numControlWithPattern)

    for i in range(1, 4 * len(regionNames) - 1):
        if markInSolution[i] == 1:
            print(snpNames[math.floor(i / 4)])


if __name__ == '__main__':
    inputFileName   = sys.argv[1]
    numCaseIndiv    = int(sys.argv[2])
    numControlIndiv = int(sys.argv[3])
    solutionSize    = int(sys.argv[4])

    initialLPModel = Model("initialLPModel")

    snpNames, inputMatrix = readInputFile(inputFileName)
    binaryGwas = buildBinaryGwas(inputMatrix)
    initialLPModel, individualVars = setObjective(initialLPModel)
    initialLPModel = addConstraints(initialLPModel, binaryGwas, individualVars, solutionSize)
    initialLPModel.optimize()
    printSolution(initialLPModel, snpNames)
    initialLPModel.write('initial-model.lp')
