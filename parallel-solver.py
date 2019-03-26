from mpi4py import MPI
from model import *
import random
import time
import math

FINISHING_THRES = 1
MODEL_DIR = '/home/jcb7d7/data/orca-solver/'

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def parentInit():
    currentModel, binaryGwas, iVars, mVars, snpNames = buildInitialModel()
    counters = {'doMIP': True, 'cycles': 2,
                'bounds': [0, 1], 'childWorking': {},
                'allWorking': True, 'notWorking': 0}
    for i in range(1, size):
        counters['childWorking'][i] = False

    data = dict({
        'initialConstraints': binaryGwas,
        'addedConstraints': [],
        'finish': False
    })

    data['addedConstraints'], counters['infeasible'],\
    counters['intLP'] = genPiercingCuts(currentModel, data['addedConstraints'], False)

    i = 0
    while counters['bounds'][0] < counters['bounds'][1]:
        if counters['bounds'][0] != 0:
            if (counters['bounds'][1] - counters['bounds'][0]) / \
                    counters['bounds'][0] < FINISHING_THRES:
                data['finish'] = True
                counters['finish'] = True
        if counters['intLP']:
            break
        print(counters)
        i += 1
        if data['addedConstraints'][-1]['obj'] < counters['bounds'][1]:
            counters['bounds'][1] = data['addedConstraints'][-1]['obj']
        counters['allWorking'] = True
        counters['notWorking'] = 0

        for j in counters['childWorking']:
            if not counters['childWorking'][j]:
                counters['allWorking'] = False
                counters['notWorking'] = j

        if counters['allWorking']:
            mesg = comm.recv(source=MPI.ANY_SOURCE)
            counters['childWorking'][mesg['rank']] = False

            if 'infeasibleIP' in mesg:
                continue
            elif mesg['infeasible'] or counters['infeasible']:
                counters['infeasible'] = True
                counters['childWorking'][mesg['rank']] = False
                break
            elif 'lower' in mesg:
                if mesg['lower'] > counters['bounds'][0]:
                    counters['bestMark'] = mesg['snps']

            counters['bounds'] = processMesg(mesg, counters['bounds'])
            counters['childWorking'][mesg['rank']] = False

        else:
            counters['childWorking'][counters['notWorking']] = True
            if counters['doMIP']:
                data['doMIP'] = True
                comm.send(data, dest=counters['notWorking'])

            else:
                data['doMIP'] = False
                comm.send(data, dest=counters['notWorking'])
            counters['doMIP'] = not counters['doMIP']
            counters['cycles'] -= 1

        if counters['cycles'] <= 0:
            counters['cycles'] = 2
            data['addedConstraints'], counters['infeasible'], \
                counters['intLP'] = genPiercingCuts(currentModel, data['addedConstraints'], data['finish'])
            currentModel = addLPConstraints(currentModel, mVars, data['addedConstraints'], solutionSize)
            currentModel.write(MODEL_DIR + 'models/lp-model.lp')

    if counters['infeasible']:
        j = 0
        for i in range(1, len(counters['childWorking']) + 1):
            if counters['childWorking'][i]:
                j += 1
        for i in range(j):
            mesg = comm.recv(source=MPI.ANY_SOURCE)

        if 'lower' in mesg:
            if mesg['lower'] > counters['bounds'][0]:
                counters['bestMark'] = mesg['snps']

        counters['bounds'] = processMesg(mesg, counters['bounds'])

    print(counters)
    for i in range(len(counters['bestMark'])):
        print(snpNames[i])

    exit(0)


def childInit():
    while True:
        mesg = {}
        data = comm.recv(source=0)
        if data['doMIP']:
            mipModel = buildIPMIPModel(data)
            mipModel.write(MODEL_DIR + 'models/mip-model.lp')
            mipModel.optimize()
            if mipModel.status == GRB.Status.INFEASIBLE:
                print('infeasible MIP', mipModel.status)
                mesg['infeasible'] = True
                mesg['rank'] = rank
                comm.send(mesg, dest=0)
                break
            mesg['upper'] = mipModel.objVal
        else:
            ipModel = buildIPMIPModel(data)
            ipModel.write(MODEL_DIR + 'models/integral-model.lp')
            ipModel.optimize()
            if ipModel.status == GRB.Status.INFEASIBLE:
                print('infeasible IP', ipModel.status)
                mesg['infeasibleIP'] = True
                mesg['rank'] = rank
                comm.send(mesg, dest=0)
                continue
            mesg['lower'] = ipModel.objVal
            mesg['snps'] = []
            for i in range(numCaseIndiv + numControlIndiv, len(ipModel.getVars())):
                if ipModel.getVars()[i].X >= 1:
                    mesg['snps'].append(i - (numCaseIndiv + numControlIndiv))

        mesg['rank'] = rank
        mesg['infeasible'] = False
        comm.send(mesg, dest=0)


def processMesg(mesg, bounds):
    if 'upper' in mesg:
        if mesg['upper'] < bounds[1]:
            bounds[1] = mesg['upper']
    elif 'lower' in mesg:
        if mesg['lower'] > bounds[0]:
            bounds[0] = mesg['lower']
    return bounds


def genPiercingCuts(model, cuts, finish):
    model.write(MODEL_DIR + 'models/initial-model.lp')
    model.optimize()
    if model.status == GRB.Status.INFEASIBLE:
        print('infeasible LP', model.status)
        return cuts, True
    vars = model.getVars()
    toForceInt = []

    integerSolution = True

    for j in range(numCaseIndiv + numControlIndiv):
        if vars[j].X != 1 and vars[j].X != 0:
            toForceInt.append(1)
            integerSolution = False
        else:
            toForceInt.append(0)

    markVals = []

    for i in range(numCaseIndiv + numControlIndiv, len(vars)):
        markVals.append({i - (numCaseIndiv + numControlIndiv): vars[i].X})
        if vars[i].X != 0 and vars[i].X != 1:
            integerSolution = False

    markVals.sort(key=takeSecond)

    if finish:
        includedInSparseCut = []
        choices = np.arange(0, len(markVals))
        for i in range(solutionSize * 2):
            choice = random.randint(0, len(choices) - 1)
            includedInSparseCut.append(choices[choice])
            choices = np.delete(choices, choice, 0)
    else:
        currentCut = genNaiveCut(markVals)

        includedInSparseCut = checkIfCutIsDuplicate(currentCut, cuts, markVals)

    currentCut = {'indiv': toForceInt,
                  'mark': includedInSparseCut,
                  'obj': model.getObjective().getValue()}

    if isinstance(currentCut['mark'], list):
        currentCut['mark'].sort()
    else:
        keys = list(currentCut['mark'].keys())
        currentCut = []
        for i in currentCut:
            currentCut[i] = int(keys[i])
        currentCut.sort()

    cuts.append(currentCut)

    return cuts, False, integerSolution

def genNaiveCut(markVals):
    quantityTaken = percent / 100.0 * len(markVals) + solutionSize

    includedInSparseCut = []
    for i in range(len(markVals) - 1, 0, -1):
        if i > len(markVals) - quantityTaken:
            includedInSparseCut.append(list(markVals[i].keys()).pop())

    return includedInSparseCut

def checkIfCutIsDuplicate(currentCut, cuts, currentMark):
    while True:
        if len(cuts) == 0:
            break
        uniqCut = True
        for i in range(len(cuts)):
            if set(currentCut) == set(cuts[i]['mark']):
                if len(currentCut) > math.floor(size / 4):
                    randArray = np.random.permutation(currentMark)
                    for i in range(solutionSize):
                        currentCut.append(list(randArray[i].keys()).pop())
                    break
                uniqCut = False
                currentCut.append(list(currentMark[-1].keys()).pop())
                currentMark.pop(-1)
                break
        if uniqCut:
            break
        if len(currentMark) < 1:
            break
    return currentCut

def takeSecond(elem):
    return list(elem.values()).pop()


def buildIPMIPModel(data):
    model, binaryGwas, iVars, mVars, snpNames = buildInitialModel()
    if data['doMIP']:
        model = addLPConstraints(model, mVars, data['addedConstraints'], solutionSize)
        model = addMIPConstraints(model, iVars, data['addedConstraints'])
    else:
        model = addIPConstraints(model, iVars, mVars, data['addedConstraints'])

    return model

def buildInitialModel():
    readInputFile(inputFileName)

    snpNames, inputMatrix = readInputFile(inputFileName)
    binaryGwas = buildBinaryGwas(inputMatrix, numCaseIndiv,
                                 numControlIndiv)

    initialLPModel = Model("initialLPModel")
    initialLPModel, individualVars = setObjective(initialLPModel,
                                                  numCaseIndiv,
                                                  numControlIndiv)
    initialLPModel, markVars = addInitialConstraints(initialLPModel,
                                                     binaryGwas,
                                                     individualVars,
                                                     solutionSize,
                                                     numCaseIndiv,
                                                     numControlIndiv)

    return initialLPModel, binaryGwas, individualVars, markVars, snpNames



if __name__ == "__main__":
    inputFileName   = sys.argv[1]
    numCaseIndiv    = int(sys.argv[2])
    numControlIndiv = int(sys.argv[3])
    solutionSize    = int(sys.argv[4])
    percent = int(sys.argv[5])

    if rank == 0:
        parentInit()
    else:
        childInit()
