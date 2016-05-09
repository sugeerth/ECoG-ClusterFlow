"""

Various distance metrics used in clustering


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

import math, numpy

try:
    import euclidean
    EUC_C_EXT_ENABLED = 1
except:
    EUC_C_EXT_ENABLED = 0

def euc(first_point, second_point):
    """Euclidean distance"""

    return math.sqrt(sum([ (second_point[i] - first_point[i])**2 for i in xrange(len(first_point)) ]))

def pearson(first_point, second_point):
    """Get Pearson score for two lists"""

    sum1 = sum(first_point)
    sum2 = sum(second_point)

    sum1Sq = sum([ pow(x, 2) for x in first_point ])
    sum2Sq = sum([ pow(x, 2) for x in second_point ])

    pSum = sum([ first_point[i] * second_point[i] for i in xrange(len(first_point)) ])

    num = pSum - (sum1 * sum2 / len(first_point))
    den = math.sqrt((sum1Sq - pow(sum1, 2) / len(first_point)) * (sum2Sq - pow(sum2, 2) / len(first_point)))

    if den == 0:
        return 0

    return 1.0 - num/den #Lower scores = higher correlation

def get_dist_func(name):
    """
    
    Valid names:
        Euclidean
        Pearson

    """

    if name == 'Euclidean':
        
        if EUC_C_EXT_ENABLED:
            return euclidean.euclidean
        else:
            return euc

    elif name == 'Pearson':
        
        #FIXME: Until I write my own c-extension, this is as good as it gets.  And it's SLOW.
        return lambda x, y: 1 - numpy.corrcoef(x,y)[0][1] #Again, we normalise -1 to distant and 1 to close. corrcoef returns the correlation matrix.

    else:

        raise ValueError, 'No distance function named: %s' % name
