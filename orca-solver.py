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
    binaryMatrix = np.reshape(binaryMatrix, (len(inputMatrix) * (numCaseIndiv + numControlIndiv), 4))

    G = np.array([])

    for indiv in range(numCaseIndiv + numControlIndiv):
        G = np.concatenate((G, []))

    print(binaryMatrix)

    print(G)

    for region in range(len(binaryMatrix)):
        for indiv in range(len(binaryMatrix[region]), 4):
            G = np.concatenate((G[indiv / 4]))

    return G

def addConstraints(initialLPModel, G, individualVars, desiredPatternSize):
    encodedGwasVars = []
    s = len(G)
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
            constraintJ += G[region][indiv] * encodedGwasVars[region]
            constraintK += G[region][indiv] * encodedGwasVars[region]
        initialLPModel.addConstr(desiredPatternSize - constraintJ,
                                 GRB.LESS_EQUAL,
                                 s * (1 - individualVars[indiv]))
        initialLPModel.addConstr(constraintK - desiredPatternSize + 1,
                                 GRB.LESS_EQUAL,
                                 s * individualVars[indiv])

    return initialLPModel



if __name__ == '__main__':
    inputFileName   = sys.argv[1]
    numCaseIndiv    = int(sys.argv[2])
    numControlIndiv = int(sys.argv[3])
    solutionSize    = int(sys.argv[4])

    initialLPModel = Model("initialLPModel")

    snpNames, inputMatrix = readInputFile(inputFileName)
    initialLPModel, individualVars = setObjective(initialLPModel)
    binaryGwas = buildBinaryGwas(inputMatrix)
    initialLPModel = addConstraints(initialLPModel, binaryGwas, individualVars, solutionSize)
    # initialLPModel.optimize()
    # initialLPModel.write('initial-model.lp')





