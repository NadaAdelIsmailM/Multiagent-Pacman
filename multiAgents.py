# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import math
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (nextPosition).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        nextGameState = currentGameState.generatePacmanSuccessor(action)

        nextPosition = nextGameState.getPacmanPosition()   # pacman

        foodList = currentGameState.getFood().asList()         # food coordinates
        foodDistList = [util.manhattanDistance(nextPosition, food)for food in foodList]
        foodDistList.sort()
        nearestFood = foodDistList[0]
        activeGhosts = [ghost for ghost in nextGameState.getGhostStates() if ghost.scaredTimer == 0] #get active ghosts who are not currently scared
        nearestActiveGhostDist = 0

        if len(activeGhosts) != 0:
            nearestActiveGhostDist = min([util.manhattanDistance(nextPosition, ghost.getPosition()) for ghost in activeGhosts]) #dist of nearest active ghost


        # weights
        ghostFear = 3
        foodMotivaction = 2.5
        if nextGameState.isLose():
            return -math.inf #BAD choice
        if nextGameState.isWin():
            return math.inf  #GOOD choice

        if nearestActiveGhostDist > 10:
            ghostFear = 0.5


        return ghostFear * nearestActiveGhostDist - foodMotivaction * nearestFood + nextGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        agentsNumber = gameState.getNumAgents()
        def mymax(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            mymaxint = -math.inf
            actions = state.getLegalActions(0)#get legal action for pacman
            for action in actions:
                nextState = state.generateSuccessor(0, action)
                mymaxint = max(mymaxint, mymin(1, nextState, depth))
            return mymaxint

        def mymin(agent, state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agent == (agentsNumber - 1): #last agent
                depth += 1
            newAgent = (agent + 1) % agentsNumber
            print(str(agent)+" "+str(newAgent)+" ")
            myminint = math.inf
            actions = state.getLegalActions(agent)  # get legal action for agent
            for action in actions:
                newState = state.generateSuccessor(agent, action)
                if newAgent == 0:
                    myminint = min(myminint, mymax(newState, depth))
                else:
                    myminint = min(myminint, mymin(newAgent, newState, depth))
            return myminint

        bestAction, bestScore = None, None
        actions = gameState.getLegalActions(0)  # get legal action for pacman
        for action in actions:
            state = gameState.generateSuccessor(0, action)
            stateScore = mymin(1, state, 0)
            if bestScore is None or bestScore < stateScore:
                bestScore, bestAction = stateScore, action
        return bestAction

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def myABmax(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            maxint = -math.inf
            actions=state.getLegalActions(0)
            for action in actions:
                nextState = state.generateSuccessor(0, action)
                maxint = max(maxint, myABmin(1, nextState, depth, alpha, beta))
                if maxint > beta: return maxint
                alpha = max(alpha, maxint)
            return maxint

        def myABmin(agent, state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agent == (numAgents - 1): depth += 1
            newAgent = (agent + 1) % numAgents
            minint = math.inf
            actions=state.getLegalActions(agent)
            for action in actions:
                newState = state.generateSuccessor(agent, action)
                if newAgent == 0:
                    minint = min(minint, myABmax(newState, depth, alpha, beta))
                else:
                    minint = min(minint, myABmin(newAgent, newState, depth, alpha, beta))
                if minint < alpha:
                    return minint
                beta = min(beta, minint)
            return minint

        bestAction, bestScore,alpha, beta = None, None,-math.inf, math.inf
        actions = gameState.getLegalActions(0)
        for action in actions:
            state = gameState.generateSuccessor(0, action)
            stateScore = myABmin(1, state, 0, alpha, beta)
            if bestScore == None or bestScore < stateScore:
                bestScore, bestAction = stateScore, action
            if bestScore > beta:
                return bestAction
            alpha = max(alpha, bestScore)
        return bestAction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        def mymax(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            maxi = -math.inf
            for action in state.getLegalActions(0):
                nextState = state.generateSuccessor(0, action)
                maxi = max(maxi, emax(1, nextState, depth))
            return maxi

        def emax(agent, state, depth):
            if state.isWin() or state.isLose()or depth == self.depth:
                return self.evaluationFunction(state)
            if agent == (numAgents - 1):
                depth += 1
            newAgent = (agent + 1) % numAgents
            numberofactions = len(state.getLegalActions(agent))
            expectedvalue = 0
            for action in state.getLegalActions(agent):
                newState = state.generateSuccessor(agent, action)
                if newAgent == 0:  # pacman
                    expectedvalue += mymax(newState, depth)
                else:
                    expectedvalue += emax(newAgent, newState, depth)
            return expectedvalue/numberofactions

        bestAction, bestScore = None, None
        for action in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, action)
            stateScore = emax(1, state, 0)
            if bestScore is None or bestScore < stateScore:
                bestScore, bestAction = stateScore, action

        return bestAction

        util.raiseNotDefined()
x=0
def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isLose():
        return -math.inf
    if currentGameState.isWin():
        return math.inf

    score = 0

    look_left=-1
    look_right=2
    look_down = -1
    look_up=2

    def getWalls(point, state):
        walls = 0
        for x in range(look_left, look_right):
            for y in range(look_down,look_up):
                if state.hasWall(point[0] + x, point[1] + y):
                    walls += 1
        return walls

    whereAmI = currentGameState.getPacmanPosition()
    # walls
    score += 2 * getWalls(whereAmI, currentGameState)

    foodDist = [util.manhattanDistance(whereAmI, food) for food in currentGameState.getFood().asList()]
    foodDist.sort()
    nearestFood = foodDist[0]
    avgFoodDist = len(foodDist) / sum(foodDist)
    foodMotivation = 0.5
    score += foodMotivation*avgFoodDist
    score += 1 / (0.8*nearestFood)

    # ghosts
    activeGhosts = [ghost for ghost in currentGameState.getGhostStates() if ghost.scaredTimer == 0]
    foodGhosts = [ghost for ghost in currentGameState.getGhostStates() if ghost.scaredTimer != 0]

    nearestActiveGhost = 0
    nearestFoodGhost = 0

    if activeGhosts!=[]:
        nearestActiveGhost = min([util.manhattanDistance(whereAmI, normalGhost.getPosition()) for normalGhost in activeGhosts])
    if foodGhosts!=[]:
        nearestFoodGhost = min([util.manhattanDistance(whereAmI, scaredGhost.getPosition()) for scaredGhost in foodGhosts])
        ScaredTimes = [scaredGhost.scaredTimer for scaredGhost in foodGhosts]

    # capsules
    caps = currentGameState.getCapsules()

    if caps!=[]:
        nearestCap = min([manhattanDistance(whereAmI, pos) for pos in caps])
        score += 1 / sum([manhattanDistance(whereAmI, pos) for pos in caps])
    else:
        nearestCap = 0

    # weights
    nearestGhostW = 2.7
    nearestCapW = 1

    if nearestActiveGhost > 5: nearestGhostW = 1
    score += nearestGhostW * nearestActiveGhost

    scaredGhostsScore = 0
    for ghost in foodGhosts:
        scaredGhostsScore += manhattanDistance(ghost.getPosition(), whereAmI)

    if scaredGhostsScore:
        if nearestFoodGhost < 5:
            score += 1 / scaredGhostsScore + sum(ScaredTimes) + 1 / (0.2 * nearestFoodGhost)
        else:
            score += 1 / scaredGhostsScore + sum(ScaredTimes) + 1 / nearestFoodGhost

    normalGhostsScore = 0
    for ghost in activeGhosts:
        normalGhostsScore += manhattanDistance(ghost.getPosition(), whereAmI)
    score += normalGhostsScore

    if nearestCap < 5:
        nearestCapW = 0.4

    if nearestCap:
        score += 1 / (nearestCap * nearestCapW)

    return score + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
