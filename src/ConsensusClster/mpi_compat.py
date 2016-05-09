"""

MPI compatibility methods


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

try:
    import mpi
    MPI_ENABLED = 1
except:
    MPI_ENABLED = 0

import time, threading

def only_once(func):
    """

    Function decorator for ensuring only mpi rank 0 performs the decorated function

    Usage:

    @only_once
    def myfunc(etc)


    """

    def wrapped(*args, **kwds):
        if MPI_ENABLED:
            if mpi.rank == 0:
                return func(*args, **kwds)
        else:
            return func(*args, **kwds)
    
    return wrapped

def mpi_enabled():
    """Returns MPI on/off state"""

    return MPI_ENABLED

def sleep_nodes(tag):
    """

    Informs non-rank 0 nodes to sleep until they're told otherwise

    Tag argument must be an integer, and specifies which wake_nodes(int) call they should
    be waiting for.

    Tags used in ConsensusCluster:
    1 - Start
    2 - Wait for PCA results
    3 - Exit


    """

    if MPI_ENABLED:
        if mpi.rank != 0:
            r = mpi.irecv(0, tag)

            while not r:
                time.sleep(1)

def wake_nodes(tag):
    """

    Informs nodes asleep via sleep_nodes(int) to wake up.  Tag argument must be the same
    as the tag used to sleep them.  Tag must be an integer.

    Tags used in ConsensusCluster:
    1 - Start
    2 - Wait for PCA results
    3 - Exit


    """

    if MPI_ENABLED:
        if mpi.rank == 0:
            for r in mpi.WORLD:
                mpi.isend(1, r, tag)

class Thread(threading.Thread):
    """
    
    Convenience class to ensure all created threads are daemonised

    Daemon threads will quit when all non-daemon threads (such as the main thread)
    have ceased.

    """

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):

        threading.Thread.__init__(self, group, target, name, args, kwargs)

        self.setDaemon(1)


def thread_watcher(f, args, kwds):
    """Start f in a new thread, then put the node to sleep until tag 3 (exit) is sent"""

    thread = Thread(target=f, args=args, kwargs=kwds)
    thread.start()

    sleep_nodes(3)
