from itertools import chain, combinations
from aimacode.planning import Action
from aimacode.utils import expr

from layers import BaseActionLayer, BaseLiteralLayer, makeNoOp, make_node


class ActionLayer(BaseActionLayer):

    def _inconsistent_effects(self, actionA, actionB):
        """ Return True if an effect of one action negates an effect of the other

        Hints:
            (1) `~Literal` can be used to logically negate a literal
            (2) `self.children` contains a map from actions to effects

        See Also
        --------
        layers.ActionNode
        """
        # TODO: implement this function
        # compare each children
        for a in self.children[actionA]:
            for b in self.children[actionB]:
                if a == ~b:
                    return True
        return False


    def _interference(self, actionA, actionB):
        """ Return True if the effects of either action negate the preconditions of the other 

        Hints:
            (1) `~Literal` can be used to logically negate a literal
            (2) `self.parents` contains a map from actions to preconditions
        
        See Also
        --------
        layers.ActionNode
        """
        # TODO: implement this function
        # compare parents of a and children of b
        for a in self.parents[actionA]:
            for b in self.children[actionB]:
                if a == ~b:
                    return True
        #compare children of a and parents of b
        for a in self.children[actionA]:
            for b in self.parents[actionB]:
                if a == ~b:
                    return True
      
        return False

    def _competing_needs(self, actionA, actionB):
        """ Return True if any preconditions of the two actions are pairwise mutex in the parent layer

        Hints:
            (1) `self.parent_layer` contains a reference to the previous literal layer
            (2) `self.parents` contains a map from actions to preconditions
        
        See Also
        --------
        layers.ActionNode
        layers.BaseLayer.parent_layer
        """
        # TODO: implement this functionx
        # check mutex of preconditions of a and b
        for a in actionA.preconditions:
            for b in actionB.preconditions:
                if self.parent_layer.is_mutex(a, b):
                    return True
        return False  

class LiteralLayer(BaseLiteralLayer):

    def _inconsistent_support(self, literalA, literalB):
        """ Return True if all ways to achieve both literals are pairwise mutex in the parent layer

        Hints:
            (1) `self.parent_layer` contains a reference to the previous action layer
            (2) `self.parents` contains a map from literals to actions in the parent layer

        See Also
        --------
        layers.BaseLayer.parent_layer
        """
        # TODO: implement this function
        # check mutex of parents a and parents b
        for a in self.parents[literalA]:
            for b in self.parents[literalB]:
        #        if self.parent_layer.is_mutex(a, b):
        #            return True
        #return False  
                if not self.parent_layer.is_mutex(a, b):
                    return False
        return True

    def _negation(self, literalA, literalB):
        """ Return True if two literals are negations of each other """
        # TODO: implement this function
        # check literal a and b
        if literalA == ~literalB:
            return True
        else:
            return False


class PlanningGraph:
    def __init__(self, problem, state, serialize=True, ignore_mutexes=False):
        """
        Parameters
        ----------
        problem : PlanningProblem
            An instance of the PlanningProblem class

        state : tuple(bool)
            An ordered sequence of True/False values indicating the literal value
            of the corresponding fluent in problem.state_map

        serialize : bool
            Flag indicating whether to serialize non-persistence actions. Actions
            should NOT be serialized for regression search (e.g., GraphPlan), and
            _should_ be serialized if the planning graph is being used to estimate
            a heuristic
        """
        self._serialize = serialize
        self._is_leveled = False
        self._ignore_mutexes = ignore_mutexes
        self.goal = set(problem.goal)

        # make no-op actions that persist every literal to the next layer
        no_ops = [make_node(n, no_op=True) for n in chain(*(makeNoOp(s) for s in problem.state_map))]
        self._actionNodes = no_ops + [make_node(a) for a in problem.actions_list]
        
        # initialize the planning graph by finding the literals that are in the
        # first layer and finding the actions they they should be connected to
        literals = [s if f else ~s for f, s in zip(state, problem.state_map)]
        layer = LiteralLayer(literals, ActionLayer(), self._ignore_mutexes)
        layer.update_mutexes()
        self.literal_layers = [layer]
        self.action_layers = []

    # make a function according to the pseudo code
    def level_cost(self):
        #(1) pseudo code doesn't work
        if 0:
            for i in range(len(self.literal_layers)):
                if self.goal in self.literal_layers[i]:
                    return i
            return 1000

        #(2) calculate level cost
        # initialize parameter
        level = 0
        costs = [0]
        reach_goal = set()

        # check whether each goal is in literal layer
        for g in self.goal:
            if g in self.literal_layers[0]:
                reach_goal.add(g)

        if not self.goal - reach_goal:
            return costs

        # repeat calculation until it's leveled
        while not self._is_leveled:
            # extend the layer
            self._extend()
            level += 1

            # add costs and goal_reach for each goal if  it's in the literal layer
            for g in (self.goal - reach_goal):
                if g in self.literal_layers[level]:
                    costs.append(level)
                    reach_goal.add(g)

            # return the cost
            if not (self.goal - reach_goal):
                return costs

    def h_levelsum(self):
        """ Calculate the level sum heuristic for the planning graph

        The level sum is the sum of the level costs of all the goal literals
        combined. The "level cost" to achieve any single goal literal is the
        level at which the literal first appears in the planning graph. Note
        that the level cost is **NOT** the minimum number of actions to
        achieve a single goal literal.
        
        For example, if Goal_1 first appears in level 0 of the graph (i.e.,
        it is satisfied at the root of the planning graph) and Goal_2 first
        appears in level 3, then the levelsum is 0 + 3 = 3.

        Hints
        -----
          (1) See the pseudocode folder for help on a simple implementation
          (2) You can implement this function more efficiently than the
              sample pseudocode if you expand the graph one level at a time
              and accumulate the level cost of each goal rather than filling
              the whole graph at the start.

        See Also
        --------
        Russell-Norvig 10.3.1 (3rd Edition)
        """
        # TODO: implement this function
        #(1) pseudo code doesn't work
        if 0: 
            costs =[]
            self.fill()
            for g in self.goal:
                costs.append(self.level_cost(g))
            return sum(costs)

        #(2) simply sum the level_cost
        costs = sum(self.level_cost())
        return costs

    def h_maxlevel(self):
        """ Calculate the max level heuristic for the planning graph

        The max level is the largest level cost of any single goal fluent.
        The "level cost" to achieve any single goal literal is the level at
        which the literal first appears in the planning graph. Note that
        the level cost is **NOT** the minimum number of actions to achieve
        a single goal literal.

        For example, if Goal1 first appears in level 1 of the graph and
        Goal2 first appears in level 3, then the levelsum is max(1, 3) = 3.

        Hints
        -----
          (1) See the pseudocode folder for help on a simple implementation
          (2) You can implement this function more efficiently if you expand
              the graph one level at a time until the last goal is met rather
              than filling the whole graph at the start.

        See Also
        --------
        Russell-Norvig 10.3.1 (3rd Edition)

        Notes
        -----
        WARNING: you should expect long runtimes using this heuristic with A*
        """
        # TODO: implement maxlevel heuristic
        #(1) pseudo code doesn't work
        if 0:
            costs =[]
            self.fill()
            for g in self.goal:
                costs.append(self.level_cost(g))
            return max(costs)

        #(2) simply take the max from level_cost
        costs = max(self.level_cost())
        return costs
 

    def h_setlevel(self):
        """ Calculate the set level heuristic for the planning graph

        The set level of a planning graph is the first level where all goals
        appear such that no pair of goal literals are mutex in the last
        layer of the planning graph.

        Hints
        -----
          (1) See the pseudocode folder for help on a simple implementation
          (2) You can implement this function more efficiently if you expand
              the graph one level at a time until you find the set level rather
              than filling the whole graph at the start.

        See Also
        --------
        Russell-Norvig 10.3.1 (3rd Edition)

        Notes
        -----
        WARNING: you should expect long runtimes using this heuristic on complex problems
        """
        # TODO: implement setlevel heuristic
        #(1) pseudo code doesn't work
        if 0:
            costs =[]
            self.fill()
            i = 0
            for layer in self.literal_layers:
                i += 1
                allGoalsMet = True
                for g in self.goal:
                    if g not in layer:
                        allGoalsMet = False
                if not allGoalsMet:
                    continue
                
                goalsAreMutex = False
                for ga in self.goal:
                    for gb in self.goal:
                        if layer.is_mutex(ga, gb):
                            goalsAreMutex = True
                if not goalsAreMutex:
                    return i

        #(2) another way to calculate
        costs = 0

        # repeat calculation until it's leveled
        while not self._is_leveled:

            # initialize the parameters
            reach_goal = True
            mutex = False
            layers = self.literal_layers[-1]

            # check whether each goal is in the literal layers 
            for g in self.goal:
                if g not in layers:
                    reach_goal = False

            # if goal is not reached, count up the costs
            if reach_goal == False:
                costs += 1
                self._extend()
                continue
            
            # check if 2 goals are mutex
            for x in self.goal:
                for y in self.goal:
                    if layers.is_mutex(x, y):
                        mutex = True
                    
            # if there's no mutex, it's valid goal so return the costs
            if mutex == False:
                return costs
            # otherwise, count up the cost and extend the layer
            else:
                costs += 1
                self._extend()

        return costs
                
    ##############################################################################
    #                     DO NOT MODIFY CODE BELOW THIS LINE                     #
    ##############################################################################

    def fill(self, maxlevels=-1):
        """ Extend the planning graph until it is leveled, or until a specified number of
        levels have been added

        Parameters
        ----------
        maxlevels : int
            The maximum number of levels to extend before breaking the loop. (Starting with
            a negative value will never interrupt the loop.)

        Notes
        -----
        YOU SHOULD NOT THIS FUNCTION TO COMPLETE THE PROJECT, BUT IT MAY BE USEFUL FOR TESTING
        """
        while not self._is_leveled:
            if maxlevels == 0: break
            self._extend()
            maxlevels -= 1
        return self

    def _extend(self):
        """ Extend the planning graph by adding both a new action layer and a new literal layer

        The new action layer contains all actions that could be taken given the positive AND
        negative literals in the leaf nodes of the parent literal level.

        The new literal layer contains all literals that could result from taking each possible
        action in the NEW action layer. 
        """
        if self._is_leveled: return

        parent_literals = self.literal_layers[-1]
        parent_actions = parent_literals.parent_layer
        action_layer = ActionLayer(parent_actions, parent_literals, self._serialize, self._ignore_mutexes)
        literal_layer = LiteralLayer(parent_literals, action_layer, self._ignore_mutexes)

        for action in self._actionNodes:
            # actions in the parent layer are skipped because are added monotonically to planning graphs,
            # which is performed automatically in the ActionLayer and LiteralLayer constructors
            if action not in parent_actions and action.preconditions <= parent_literals:
                action_layer.add(action)
                literal_layer |= action.effects

                # add two-way edges in the graph connecting the parent layer with the new action
                parent_literals.add_outbound_edges(action, action.preconditions)
                action_layer.add_inbound_edges(action, action.preconditions)

                # # add two-way edges in the graph connecting the new literaly layer with the new action
                action_layer.add_outbound_edges(action, action.effects)
                literal_layer.add_inbound_edges(action, action.effects)

        action_layer.update_mutexes()
        literal_layer.update_mutexes()
        self.action_layers.append(action_layer)
        self.literal_layers.append(literal_layer)
        self._is_leveled = literal_layer == action_layer.parent_layer
