"""

Tree datatype and reordering methods


Copyright 2008 Michael Seiler
Rutgers University
miseiler@gmail.com

This file is part of ConsensusCluster.

ConsensusCluster is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ConsensusCluster is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ConsensusCluster.  If not, see <http://www.gnu.org/licenses/>.


"""

import math, random

try:
    import sa
    SA_C_EXT_ENABLED = 1
except:
    SA_C_EXT_ENABLED = 0


class Tree(object):
    """
    
    Tree

        Provides binary tree datatype for dendogramming

        Properties

            value   - If this is a leaf, this is the data held in it
            left    - If this is a node, this points to the node/leaf to its left
            right   - If this is a node, this points to the node/leaf to its right
            dist    - If this is a node, this is the cluster distance between left and right

            These properties are auto-updating:

            sequence    - This is a list of values from the leaves below the current node
                          i.e., a node with two leaves with values 2 and 3 would produce [2, 3]
                          The toplevel node always has the complete sequence of tree values from left to right
            depth       - This is the maximum distance to this node, counted from the leaves upward (from 0)
                          Its parent has depth equal to dist + the depth of the node with the larger depth beneath it
            parent      - This points to this node's parent node, if it has one
    
        Methods

            swap        - This swaps the left and right trees of the current node.  sequence properties are
                          recursively affected for itself and all parents, so that each node always has the correct sequence.
    
    """

    def __init__(self, value=None, left=None, right=None, dist=None):

        self.__dict__ = {'depth': 0, 'value': value, 'parent': None, 'left': left, 'right': right, 'dist': dist, 'cluster_id': None}

        if left is not None:    #If right isn't none, this needs to raise anyway
            
            self.depth = dist + max(left.depth, right.depth)
            self.sequence = self.left.sequence + self.right.sequence
            self.left.parent, self.right.parent = self, self

            self._flipped = 0   #Internal state identifying whether a swap has been made

        else:
            #Leaf
            self.sequence = [value]

    def swap(self):

        if self._flipped:
            self._flipped = 0
        else:
            self._flipped = 1

        self.left, self.right = self.right, self.left
        self._upd_parents()

    def _upd_parents(self):
        
        self.sequence = self.left.sequence + self.right.sequence
        
        if self.parent is not None:
            self.parent._upd_parents()


def reorder(tree, M):
    """
    
    reorder

        Use simulated annealing to reorder the tree in a way that maximises the similarity between close indices and maximises the
        dissimilarity between distant ones.  Why SA?  The Bar-Joseph tree-order algorithm is O(n^4), which seems like a great deal of work
        for large trees.

        NOTE: This function is expensive for small trees

        Usage
            
            order = treetype.reorder(tree, dist_matrix)

        Properties

            tree    - A treetype.Tree object.  It is assumed the leaves of this tree are indices of M
            M       - A distance matrix
    
        This function reorders the tree in place.

    """

    #Locals
    if not SA_C_EXT_ENABLED:
        cost        = lambda i, j, matrix: matrix[i][j]      #The similarity between two samples
        energy      = lambda x, matrix: sum([ cost(x[i], x[i+1], matrix) for i in range(len(x) - 1) ])**2  #Sum of the cost of each successive pair, squared
    else:
        energy = sa.c_energy

    def find_nodes(nodes, tree):
        if tree.depth == 0:
            return
        else:
            find_nodes(nodes, tree.left)
            find_nodes(nodes, tree.right)
    
            nodes.append(tree)
    
    exp, ran, rand_idx = math.exp, random.random, random.choice
    sample_len, temp, new_energy = len(M), 0.1, energy(tree.sequence, M)
    not_accepted, last_best, nodes = 0, 0, []
    
    find_nodes(nodes, tree)
    
    best_energy  = new_energy
    best_order = [ x._flipped for x in nodes ] #Track node state in binary sequence form so we can come back to it
    
    #print "Starting energy score:", new_energy
    
    for i in xrange(1, 4000000):

        last_energy = energy(tree.sequence, M)

        node = rand_idx(nodes)
        node.swap()             #Randomly swap a branch

        new_energy = energy(tree.sequence, M)

        if not (i % 2000) and i < 25000:                        #Ramp up temperature
            if ((i - not_accepted) / float(i)) < 0.1:
                temp += .1                                      #0.1 might even be too much here

        if not (i % 25000):
            temp = temp * 0.90
            #print "Acceptance rate:", ((i - not_accepted) / float(i))

        if new_energy < last_energy:
            if ran() > exp((new_energy - last_energy) / temp):
                node.swap()
                not_accepted += 1

        elif new_energy > best_energy:
            best_energy, best_order = new_energy, [ x._flipped for x in nodes ]
            #print "New best energy %f at %d" % (best_energy, i)
            last_best = i

        if (i - last_best) > 100000:            #No point in going farther than this, usually
            #print "Breaking at", i
            break

    #print "\nFinal energy score:", best_energy

    #Reorder the tree to the best order, in place
    for i in xrange(len(nodes)):
        if best_order[i] != nodes[i]._flipped:
            nodes[i].swap()
