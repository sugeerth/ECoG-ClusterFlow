"""

Visualisation and reporting methods


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

import logging, sys, os, numpy

from itertools import cycle
from mpi_compat import *

try:
    import gtk
    gtk.gdk.threads_init()
    GTK_ENABLED = 1
except:
    GTK_ENABLED = 0

try:
    import Image, ImageDraw
    # from PIL import ImageDraw, ImageFont
    import StringIO
    HMAP_ENABLED = 1
except:
    HMAP_ENABLED = 0

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_ps import FigureCanvasBase as FigureCanvas
    MATPLOTLIB_ENABLED = 1
except:
    MATPLOTLIB_ENABLED = 0

if GTK_ENABLED:
    try:
        gtk._gtk.init_check()
        DISPLAY_ENABLED = 1
    except:
        DISPLAY_ENABLED = 0
else:
    DISPLAY_ENABLED = 0


def new_filename(filename, default, extension):
    """Try to avoid squashing old files by incrementing -num extensions to the filename"""

    if filename is None:
        filename = default

    newname = filename + extension
    dirlist = os.listdir(os.curdir)
    counter = 0

    while newname in dirlist:
        counter += 1
        newname = "".join([filename, ' - %s' % counter, extension])

    return newname

@only_once
def configure_logging(logname):
    """Open one logfile at a time for writing"""

    logfile = new_filename(logname, 'log', '.log')

    file_handler = logging.FileHandler(logfile, "a")    #Output log to logfile
    
    for handler in logging.root.handlers:
        handler.close()
        logging.root.removeHandler(handler)

    logging.root.addHandler(file_handler)
    logging.root.setLevel(logging.INFO)


class ConsoleDisplay(object):
    """
    
    ConsoleDisplay

        Usage:
            
            console = ConsoleDisplay(log, logname, tview)

                log     - Boolean variable.  If True, logging is enabled.
                          Note that attempting to change the current logfile via new_logfile
                          will make this True.

                logname - Name of new logfile.  .log suffix will be appended.  Optional.

                tview   - Gtk TextView object to send our output, rather than stdout.  Optional.
                          If there is an error initialising GTK or display, this will be set to None.

        Properties:

            logging_enabled - The current boolean logging state.

        WARNING:

            If console output is assigned to stdout rather than a GTK TextView, success/failure colouring
            may only work on linux terminals.  It hasn't yet been tested elsewhere.

    """
    
    def announce_wrap(self, announcement, f, *args, **kwds):
        """
        
        Usage:

            console.announce_wrap('Announcing function', function, *func_args, **func_kwds)

            This simple wrapper writes announcement, runs f, and outputs [OK] or [FAIL]
            depending on exception behaviour.

        This used to be a beautiful decorator, but I had no sensible way to pass console objects to it.

        """
            
        self.write(announcement)

        try:
            ret = f(*args, **kwds)
            self.success()
            return ret
        except:
            self.fail()
            raise
    
    @only_once
    def progress(self, message, cur, end):
        """Either outputs progress message to stdout or sets self.progress_frac, which can be used by a progress bar"""

        if self.tview is None:
            sys.stdout.write('\r%s %4d  of %4d\t' % (message, cur, end))
        else:
            self.progress_frac = min(cur/float(end), 1.0)

    @only_once
    def success(self):
        """Print success message"""

        self._status('[OK]', self.success_colour, self.success_tag)

    @only_once
    def fail(self):
        """Print failure message"""

        self._status('[FAIL]', self.fail_colour, self.fail_tag)

    def except_to_console(self, message, throw=False):
        """Writes message to console, shows failure, then raises an exception if throw is true"""

        self.write(message)
        self.fail()

        if throw:
            raise ValueError, message

    def _status(self, s, colour, tag=None):
        """Helper intermediate between success/fail and write"""

        if self.tview is None:
            self.write("".join([self.edge, colour, s, self.reset_colour]))
        else:
            self.write(s, tag)

    def _scroll_to_end(self, iter):
        """Make sure the TextView continues to scroll down with new information"""

        self.tview.scroll_to_mark(self.viewbuf.create_mark("end", iter, False), 0.05, True, 0.0, 1.0)

    @only_once
    def log(self, message, display=True):
        """Log message.  If display is True, message is also written to console"""
    
        if self.logging_enabled:
            logging.info(message)
            
            if display:
                self.write(message)

    @only_once
    def write(self, message, tag=None):
        """
        
        Write message to console.

        tag is an optional GTK TextTag for text formatting.

        This function is made threadsafe using a gtk locking mechanism.  Otherwise if display attempts are made before
        the previous one completes, the TextIter (pointer) becomes invalid and you risk crashing or worse.


        """
        
        if self.tview is None:
            print(message)
        else:
            gtk.gdk.threads_enter()
            iter = self.viewbuf.get_end_iter()
        
            if tag is None:
                self.viewbuf.insert(iter, message + '\n')
            else:
                self.viewbuf.insert(iter, self.edge)
                self.viewbuf.insert_with_tags(iter, message + '\n', tag)

            self._scroll_to_end(iter)
            gtk.gdk.threads_leave()

    def new_logfile(self, logname=None):
        """Turn on logging (if it's off), and create a new logfile named "logname" """

        self.logging_enabled = True
        configure_logging(logname)

    def stop_logging(self):
        """Turn off logging, and close all open logfiles permanently"""

        self.logging_enabled = False

        for handler in logging.root.handlers:
            handler.close()
            logging.root.removeHandler(handler)

    def __init__(self, log=False, logname=None, tview=None):
        
        self.tview = tview
        
        if tview is not None and DISPLAY_ENABLED:
            self.viewbuf = tview.get_buffer()
            self.progress_frac = 0.

            self.success_tag = self.viewbuf.create_tag(foreground='green')
            self.fail_tag = self.viewbuf.create_tag(foreground='red')
        else:
            self.success_tag = None
            self.fail_tag = None
            self.tview = None

        self.logging_enabled = log

        self.edge = ' ' * 60 + '\t'

        if log:
            configure_logging(logname)

        self.reset_colour = chr(27) + "[0m"     #Reset
        self.success_colour = chr(27) + "[32m"  #Green
        self.fail_colour = chr(27) + "[31m"     #Red


class Hmap(object):
    """

    Hmap

        Builds a heatmap of any iterable matrix.  Does nothing if HMAP_ENABLED isn't set.

        Usage:

            Hmap(matrix, bsize = 10)

            matrix      -   Iterable matrix.  Negative values make green squares, positive ones make red ones.
                            Lightness of each square is determined relative to all the others.  I.e., if the max value
                            in the matrix is 30, this square will be the brightest.  Likewise if the smallest value is 3,
                            that square will be the darkest.  All other squares will scale linearly between them.
            bsize       -   The box size for each value in the heatmap
        
        Methods

            save(filename = None)
                Saves the current image to filename in PNG format.  Otherwise filename is 'heatmap.png'. Auto-appends .png to filename.

            show()
                Creates a GTK window displaying the current image
                
                WARNING: Requires the Gtk_UI interface!

        Properties

            im
                The current image

    """

    def __init__(self, matrix, bsize = 10):

        self.im = None
        self.bsize = bsize

        if HMAP_ENABLED:
            self.im = self._create_hmap(matrix)

    def _get_lightness(self, values):

        if values:

            max_val = max(values)
            min_val = min(values)
    
            try:
                lightness = 255 / float(max_val - min_val)
            except ZeroDivisionError:
                if not max_val:
                    lightness = 0
                else:
                    lightness = 255     #If all data is the same nonzero amount, it still needs colour
        else:
            lightness = 0

        return lightness

    def _create_hmap(self, matrix):
       
        size = (len(matrix[0]) * self.bsize, len(matrix) * self.bsize)
    
        red_lightness = self._get_lightness([ x for i in xrange(len(matrix)) for x in matrix[i] if x >= 0 ])
        green_lightness = self._get_lightness([ x for i in xrange(len(matrix)) for x in matrix[i] if x < 0 ])

        im = Image.new('RGBA', size, 'white')
        draw = ImageDraw.Draw(im)
        
        for row in xrange(len(matrix)):
            for col in xrange(len(matrix[row])):
                
                if matrix[row][col] < 0:
                    colour = (0,int(abs(matrix[row][col]) * green_lightness),0)
                else:
                    colour = (int(matrix[row][col] * red_lightness),0,0)

                col_size = col * self.bsize
                row_size = row * self.bsize

                bcol_size = self.bsize + col_size
                brow_size = self.bsize + row_size

                draw.polygon([(col_size, row_size),
                              (bcol_size, row_size),
                              (bcol_size, brow_size),
                              (col_size, brow_size)], outline='black', fill=colour)
            
        return im
    
    def _image_to_pixbuf(self, im):  
        """
        Creates a pixbuf from an Image object.  Probably unnecessary.  In the future I could just invoke a save
        and load then png instead.
        
        """
        
        file1 = StringIO.StringIO()  
        im.save(file1, "ppm")  
        contents = file1.getvalue()  
        file1.close()  
        loader = gtk.gdk.PixbufLoader("pnm")  
        loader.write(contents, len(contents))  
        pixbuf = loader.get_pixbuf()  
        loader.close()  
        return pixbuf  
    
    def save(self, filename = None):
        """Save the current image to filename.  Automatically appends a .png extension"""

        self.im.save(new_filename(filename, 'heatmap', '.png'), 'png')

    def show(self):
        """Opens a GTK window and puts the heatmap in it.  Intelligent enough to work with the GUI as well."""
        
        window_only = 1 #What needs to be destroyed when the window is destroyed?

        if HMAP_ENABLED and DISPLAY_ENABLED:

            def destroy():
    
                if window_only:
                    window.destroy()
                else:
                    gtk.main_quit()
    
            gtk.gdk.threads_enter()
            window = gtk.Window()
            window.set_title("Showing heatmap...")
            window.set_border_width(10)
            window.set_resizable(False)
            window.connect("delete_event", lambda w, e: destroy())
        
            backbone = gtk.HBox(True)
            image = gtk.Image()
            image.set_from_pixbuf(self._image_to_pixbuf(self.im))
            backbone.pack_start(image)
        
            window.add(backbone)
            window.show_all()
            gtk.gdk.threads_leave()

            if gtk.main_level() == 0:
                window_only = 0
                gtk.main()

        else:
            raise "HmapError", "Error loading modules or unable to display"


class Clustmap(Hmap):
    """
    
    Clustmap
        
        Adds dendogram and labels to consensus matrix Hmap

        Usage:

            m = Clustmap(clust_data, labels = None, bsize = 10, tree_space = 200)

            m.save('filename')
            m.show()

            clust_data      - cluster.ConsensusCluster object which has a consensus_cluster attribute and optionally sample_id
                              labels for each sample in clust_data.datapoints
            labels          - Optional list of sample labels.  It should be the same length as clust_data.datapoints and it should
                              be in the same order as clust_data.consensus_matrix.  This is important!  ConsensusCluster reorders
                              its matrix but ignores datapoints, because that is input from elsewhere.  If you have a list which is
                              in the same order as datapoints, simply use the following list comprehension:
                                
                                [ MyLabels[x] for x in clust_data.reorder_indices ]
            bsize           - Size of heatmap datapoints.  Don't make this less than 10 unless you have a font which is smaller than 10pt.
            tree_space      - 200 is a good default.  If there is no tree (no reordering, or reordered by PAM method, for example),
                              this is ignored.

    """

    def __init__(self, clust_data, labels = None, bsize = 10, tree_space = 200):
        
        self.space = tree_space
        
        colours = ['blue', 'green', 'red', 'cyan', 'magenta', 'brown', 'orange']
        self.colour_map = self._init_colours(colours, [ x.cluster_id for x in clust_data.datapoints ])
        
        if labels is None:
            labels = [ clust_data.datapoints[x].sample_id for x in clust_data.reorder_indices ]

        try:
            self.font = ImageFont.load('courR08.pil') #Copyright (c) 1987 Adobe Systems, Inc., Portions Copyright 1988 Digital Equipment Corp.
        except IOError:
            self.font = None

        if len(clust_data.consensus_matrix) != len(labels):
            raise ValueError, "Number of columns and column label arrays have different lengths!"

        Hmap.__init__(self, clust_data.consensus_matrix, bsize = bsize) #Creates image in self.im if HMAP_ENABLED

        if self.im is not None:
            
            old_draw = ImageDraw.Draw(self.im)

            self.max_textsize = 0
            for label in labels:
                self.max_textsize = max(self.max_textsize, old_draw.textsize(label, font=self.font)[0])
            
            del old_draw #Keep GC from keeping the old image around

            if clust_data.tree is None:
                self.space = self.max_textsize + 5

            #Prepare
            newsize = (self.im.size[1] + self.space, self.im.size[0])  #To hold our rotated copy and some text
            im = Image.new('RGBA', newsize, 'white')
            
            #Trick to make vertical text when we're done, and add tree space
            im.paste(self.im.rotate(-90), (0, 0, self.im.size[1], self.im.size[0]))

            self.im = im
            self.draw = ImageDraw.Draw(self.im)
            
            #Actual work
            self._add_cluster_labels(labels)

            if clust_data.tree is not None:
                self._draw_dendogram(clust_data.tree)

            #Finish
            self.im = self.im.rotate(90)

    def _init_colours(self, colours, id_lst):
        """Assigns colours to a list of cluster_ids"""

        next_colour = cycle(colours)
        
        cluster_ids = dict.fromkeys(id_lst).keys()    

        col_map = dict(zip(cluster_ids, next_colour))
        col_map[None] = 'black'

        return col_map
        
    def _add_cluster_labels(self, labels):
        """Adds labels to the cluster image"""

        x_ptr = self.im.size[0] - self.space + 4
        y_ptr = 0

        for i in xrange(len(labels)):
            self.draw.text((x_ptr, y_ptr + i*self.bsize), labels[i], fill='black', font=self.font)

    def _draw_dendogram(self, tree):
        """Draws the tree.  It's a bit terrifying, I'll admit."""

        tree_seq = tree.sequence

        x_ptr = self.im.size[0] - self.space + 6 + self.max_textsize
        y_ptr = self.bsize / 2

        x_step = (self.im.size[0] - x_ptr) / tree.depth - 1

        x_loc = lambda scale: x_ptr + scale * x_step
        y_loc = lambda i: y_ptr + tree_seq.index(i) * self.bsize

        stack = [tree]
        active_level = [tree.depth]
        seen = []
        midpoints = []

        def draw_node(x_min_left, x_min_right, x_max, y_min, y_max, depth, colour):

            self.draw.line((x_min_left, y_min, x_max, y_min, x_max, y_min, x_max, y_max, x_min_right, y_max, x_max, y_max), fill=colour)
            midpoints.append(((abs(y_min - y_max) / 2 + min(y_min, y_max)), depth))

        def draw_bridge(endpoint):

            y_max = midpoints.pop()
            last_level = seen.pop()

            colour = self.colour_map[last_level.cluster_id]
            draw_node(x_loc(endpoint[1]), x_loc(y_max[1]), x_loc(last_level.depth), endpoint[0], y_max[0], last_level.depth, colour)

        while len(stack):
            next = stack.pop()
            level = active_level.pop()

            if seen and level > seen[-1].depth:
                stack.append(next)
                active_level.append(level)
                
                draw_bridge(midpoints.pop())
                continue

            if next.value is not None:
                draw_bridge([y_loc(next.value), 0])
            
            elif next.right.value is not None and next.left.value is not None:
                colour = self.colour_map[next.cluster_id]
                draw_node(x_ptr, x_ptr, x_loc(next.depth), y_loc(next.left.value), y_loc(next.right.value), next.depth, colour)

            else:
                seen.append(next)

                if next.right.value is None:
                    stack.extend([next.left, next.right])
                elif next.left.value is None:
                    stack.extend([next.right, next.left])

                active_level.extend([next.depth, next.depth])

        while len(seen):
            draw_bridge(midpoints.pop())


class Plot(object):

    """

    Plot

        Small front-end class to matplotlib's extensive plotting functions.

        Plot will take a list of 2 by X numpy matrices and a list of labels (optional) and draw them to the same grid
        with different (rotating) colours.  The labels, if set, will become a legend identifying these sets.

        Usage:

        Plot(plots, fig_label = '', legend = None)

            plots       - a list of numpy matrices to plot, each with 2 rows.  Row 0 is x-coord, row 1 is y-coord.

            fig_label   - The saved filename will become fig_label.png.  If fig_label is missing, it will be called "figure.png"

            legend      - a list of labels, same length as plots

        When Plot is initialised, the plot will be drawn and an image will be saved.

    """

    def __init__(self, plots, fig_label = None, legend = None):

        if MATPLOTLIB_ENABLED:
    
            self.colours = cycle(['bo', 'go', 'ro', 'co', 'mo', 'ko', 'yo', 'bs', 'gs', 'rs', 'cs', 'ms', 'ks', 'ys', 'b^', 'g^', 'r^', 'c^', 'm^', 'k^', 'y^'])

            self.legend = legend

            if self.legend is not None:
                if len(plots) != len(legend):
                    raise ValueError, "Plots and legend have different lengths!"
    
            self._write_fig(plots, fig_label)

    @only_once
    def _write_fig(self, plots, fig_label):

        fig = Figure()
        ax = fig.add_subplot(111)

        for i in xrange(len(plots)):

            if plots[i].shape[0] != 2:
                raise ValueError, "Attempting to plot matrix with row count other than 2"
            
            if self.legend is not None:
                ax.plot(plots[i][0], plots[i][1], self.colours.next(), label=self.legend[i])
            else:
                ax.plot(plots[i][0], plots[i][1], self.colours.next())

        if self.legend is not None:
            ax.legend(loc='best')

        canvas = FigureCanvas(fig)
        canvas.figure.savefig(new_filename(fig_label, 'figure', '.png'))


def km(times, events, labels, fig_label = None):
    """

    Plot the kaplan-meier curves

    times - list of arrays of survival times (or a matrix)
    censors - a list of event arrays, 1 for an event and 0 for censor (or a matrix)
    labels - a list of labels (for the lists)

    These should all be the same length

    """

    numcurves = len(times)

    if not numcurves == len(events) == len(labels):
        raise ValueError, 'Lists are not all the same length!'
        
    colours = cycle(['b', 'g', 'r', 'c', 'm', 'k', 'y'])

    fig = Figure()
    ax = fig.add_subplot(111)

    for i in xrange(numcurves):
        ar = numpy.array(times[i])
        sorting = tuple(numpy.argsort(ar))

        ar = ar.take(sorting)
        ev = numpy.array(events[i]).take(sorting)
        label = labels[i]

        points = numpy.array([0] + list(numpy.unique(ar))) #X-axis timepoints
        nonsurvival = numpy.zeros(len(points)) #float
        censors = nonsurvival.copy()

        bin = 1 #Ignore 0, everyone's alive
        cur = ar[0]

        for j in xrange(len(ar)):
            if ar[j] != cur: #Increment current timepoint, bin
                cur = ar[j]
                bin += 1

            if ev[j]:
                nonsurvival[bin] += 1 #Count events at each timepoint
            else:
                censors[bin] += 1

        rev_ind = tuple(reversed(range(len(censors))))
        mark_indices = tuple(numpy.nonzero(censors))
        total_at_risk = numpy.cumsum(censors.take(rev_ind) + nonsurvival.take(rev_ind)).take(rev_ind)
        
        #Interval adjustments
        censors[2] += censors[1]
        censors[1] = 0
        censors[-1] = 0
        total_at_risk -= censors

        probdist = numpy.cumprod(1 - nonsurvival/total_at_risk) #Y-axis

        colour = colours.next()

        #Draw them
        ax.step(points, probdist, colour, label=labels[i], where='mid')
        ax.plot(points.take(mark_indices), probdist.take(mark_indices), colour + '|')

    ax.legend(loc='best')
    minbound, maxbound = ax.get_ybound()
    ax.set_ybound(minbound, maxbound + 0.01)

    canvas = FigureCanvas(fig)
    canvas.figure.savefig(new_filename(fig_label, 'figure', '.png'))


if __name__ == '__main__':

    matrix = []

    for i in range(32):
        matrix.append([])
        for j in range(32):
            if i & 1:
                matrix[i].append(float(-1 * i * j * 0.5))
            else:
                matrix[i].append(float(i * j * 0.5))

    Hmap(matrix, bsize = 15).show()
