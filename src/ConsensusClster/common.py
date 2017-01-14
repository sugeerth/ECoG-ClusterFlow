"""

Default clustering workflow and associated front end


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

import warnings
warnings.simplefilter('ignore', DeprecationWarning)

import numpy, sys, time, os
import pca, parsers, cluster, display, scripts
from itertools import combinations as comb

from mpi_compat import *

try:
    import psyco
    psyco.full()
except:
    pass

if display.GTK_ENABLED:
    import gtk, gobject



class Gtk_UI(object):
    """

    Gtk_UI

        GTK front end to CommonCluster.  All arguments and keywords are the same.  See CommonCluster for details.

        For *useful* subclassing (as in, changes workflow to suit your needs), see CommonCluster.

        All display.ConsoleDisplay output will be placed in a GTK TextView instance instead.
        If GTK or display initialisation fails, display will run from the console.

    """

    def __init__(self, *args, **kwds):

        self.filename = None
        self.parser = None
        self.thread = None

        if not display.DISPLAY_ENABLED:
            self.consensus_procedure(*args, **kwds)

        else:
            gtk.gdk.threads_init()
            gobject.threads_init()
            
            self.load_display(*args, **kwds)

            self.mpi_wait_for_start()

    
    def mpi_wait_for_start(self):
        """Tells the other nodes to sleep until Begin Clustering is pressed"""

        if MPI_ENABLED:
            sleep_nodes(1)

            if mpi.rank != 0:
                t_args = mpi.bcast()    #Block until we get the go-ahead from 0

                if t_args is not None:
                    thread_watcher(self.consensus_procedure, (t_args[0], t_args[1]), t_args[2])

    @only_once
    def load_display(self, *args, **kwds):
            
        #Window Settings
        window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        window.connect("destroy", self.destroy)
        window.set_size_request(800, 600)
        window.set_resizable(False)

        #Boxen
        vbackbone = gtk.VBox(False)
        hbackbone = gtk.HBox(False)
        window.add(vbackbone)

        main_lvbox = gtk.VBox(False)
        main_rvbox = gtk.VBox(False)
        set_lvbox  = gtk.VBox(False)
        set_rvbox  = gtk.VBox(False)
        set_hbox   = gtk.HBox(False)

        hbackbone.pack_start(main_lvbox, True, True, 10)
        hbackbone.pack_end(main_rvbox, False, False, 10)

        #Tabs
        tabholder = gtk.HBox(False)
        tabs = gtk.Notebook()
        tabs.append_page(hbackbone, gtk.Label('Cluster'))
        tabs.append_page(set_hbox,  gtk.Label('Settings'))
        tabholder.pack_start(tabs, True, True, 10)

        #Menubar
        ui_str =    """
                    <ui>
                        <menubar name='Bar'>
                            <menu action='File'>
                                <menuitem action='Open'/>
                                <menuitem action='Quit'/>
                            </menu>
                            <menu action='Clustering'>
                                <menuitem action='Define Clusters'/>
                            </menu>
                        </menubar>
                    </ui>
                    
                    """

        uim = gtk.UIManager()
        window.add_accel_group(uim.get_accel_group())

        actgroup = gtk.ActionGroup('Cluster Menubar')
        actgroup.add_actions([ ('File', None, '_File'), ('Quit', gtk.STOCK_QUIT, '_Quit', None, 'Quits', self.destroy),
                               ('Open', gtk.STOCK_OPEN, '_Open File', None, 'Open a datafile', self.get_filename),
                               ('Clustering', None, '_Clustering'),
                               ('Define Clusters', None, '_Define Clusters', None, 'Set groups in a dataset from file', self.keep_list_dialog) ])

        uim.insert_action_group(actgroup, 0)
        uim.add_ui_from_string(ui_str)

        vbackbone.pack_start(uim.get_widget('/Bar'), False, False)
        vbackbone.pack_end(tabholder, True, True, 10)

        #Main Tab Stuff
        sw = gtk.ScrolledWindow()
        sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_ALWAYS)

        textview = gtk.TextView()
        textview.set_editable(False)
        sw.add(textview)

        self.console = display.ConsoleDisplay(log=False, tview=textview)
        
        self.progress = gtk.ProgressBar()
        self.pbar_timer = gobject.timeout_add(100, self._upd_pbar)

        main_lvbox.pack_start(sw, True, True, 10)
        main_lvbox.pack_end(self.progress, False, False, 10)

        self.startbutton = gtk.Button('Begin Clustering')
        self.startbutton.connect('clicked', self.run_clustering)

        quitbutton = gtk.Button('Quit')
        quitbutton.connect('clicked', self.destroy)

        label = gtk.Label("Consensus Cluster\n\nMichael Seiler\nRutgers University")
        label.set_justify(gtk.JUSTIFY_CENTER)

        for obj in (self.startbutton, quitbutton):
            main_rvbox.pack_start(obj, False, False, 10)

        main_rvbox.pack_end(label, True, True, 10)

        #Settings Tab
        
        #Fabulous frame-filling functions
        def framepack(box, framelst):
            for frame in framelst:
                box.pack_start(frame, True, True, 10)

        def packbox(box, lst):
            for obj in lst:
                box.pack_start(obj, True, False, 4)

        def setentrywidths(width, lst):
            for obj in lst:
                obj.set_width_chars(width)
                obj.set_max_length(width)

        def labelbox(box, label=None):
            if label is None: return box
            v = gtk.VBox(False)
            v.pack_start(gtk.Label(label), False, False, 4)
            v.pack_start(box, False, False, 4)
            return v

        def newboxen(frame, label1=None, label2=None):
            v, h, h2 = gtk.VBox(False), gtk.HBox(False), gtk.HBox(False)
            v.pack_start(labelbox(h, label1), True, False, 5)
            v.pack_end(labelbox(h2, label2), True, False, 5)
            frame.add(v)
            return v, h, h2

        #Frames
        algsframe = gtk.Frame()
        genframe  = gtk.Frame()
        pcaframe  = gtk.Frame()
        miscframe = gtk.Frame()
        
        for frame in ((algsframe, 'Algorithm'), (genframe, 'General'), (pcaframe, 'PCA'), (miscframe, 'Misc')):
            frame[0].set_label(frame[1])
        
        framepack(set_lvbox, (genframe, algsframe))
        framepack(set_rvbox, (pcaframe, miscframe))
        framepack(set_hbox, (set_lvbox, set_rvbox))

        #Genframe
        vbox, hbox, h2box = newboxen(genframe)

        self.k_min, self.k_max = gtk.Entry(), gtk.Entry()
        self.subs_entry, self.sub_frac_entry = gtk.Entry(), gtk.Entry()

        setentrywidths(2, (self.k_min, self.k_max))
        setentrywidths(4, (self.subs_entry, self.sub_frac_entry))

        packbox(hbox, (gtk.Label('K-Value Range'), self.k_min, gtk.Label('to'), self.k_max))
        packbox(h2box, (labelbox(self.subs_entry, 'Subsamples'), labelbox(self.sub_frac_entry, 'Fraction to Sample')))

        #Algsframe
        vbox, alg_hbox, h2box = newboxen(algsframe, 'Cluster Using')
        
        link_hbox  = gtk.HBox(False)
        vbox.pack_start(labelbox(link_hbox, 'Linkages'), True, False, 5)

        self.kmeanbox    = gtk.CheckButton(label='K-Means')
        self.sombox      = gtk.CheckButton(label='SOM')
        self.pambox      = gtk.CheckButton(label='PAM')
        self.hierbox     = gtk.CheckButton(label='Hierarchical')

        self.singlebox   = gtk.CheckButton(label='Single')
        self.averagebox  = gtk.CheckButton(label='Average')
        self.completebox = gtk.CheckButton(label='Complete')

        packbox(alg_hbox, (self.kmeanbox, self.sombox, self.pambox, self.hierbox))
        packbox(link_hbox, (self.singlebox, self.averagebox, self.completebox))

        self.finalbutton, self.distbutton = gtk.combo_box_new_text(), gtk.combo_box_new_text()
        
        for text in ('Hierarchical', 'PAM'):
            self.finalbutton.append_text(text)

        for text in ('Euclidean', 'Pearson'):
            self.distbutton.append_text(text)

        packbox(h2box, (labelbox(self.finalbutton, 'Cluster Consensus Using'), labelbox(self.distbutton, 'Distance Metric')))

        #Pcaframe
        vbox, hbox, h2box = newboxen(pcaframe, 'Normalisation')

        self.log2box     = gtk.CheckButton(label='Log2')
        self.submedbox   = gtk.CheckButton(label='Sub Medians')
        self.centerbox   = gtk.CheckButton(label='Center')
        self.scalebox    = gtk.CheckButton(label='Scale')

        self.pca_frac_entry, self.eig_weight_entry = gtk.Entry(), gtk.Entry()
        setentrywidths(4, (self.pca_frac_entry, self.eig_weight_entry))

        packbox(hbox, (self.log2box, self.submedbox, self.centerbox, self.scalebox))
        packbox(h2box, (labelbox(self.pca_frac_entry, 'PCA Fraction'), labelbox(self.eig_weight_entry, 'Eigenvalue Weight')))

        #Miscframe
        vbox, hbox, h2box = newboxen(miscframe)

        self.normvarbox = gtk.CheckButton('Set Variance to 1')
        packbox(hbox, [self.normvarbox])

        #Defaults and convenience dict for accessing values
        self.clus_alg_widgets = dict([(self.kmeanbox, cluster.KMeansCluster), (self.sombox, cluster.SOMCluster), (self.pambox, cluster.PAMCluster),
                                      (self.hierbox, cluster.HierarchicalCluster)])

        self.linkage_widgets = dict([(self.singlebox, 'single'), (self.averagebox, 'average'), (self.completebox, 'complete')])

        #Monstrosity
        self.settings = {   'kvalues': lambda: range(int(self.k_min.get_text()), int(self.k_max.get_text()) + 1),
                            'subsamples': lambda: int(self.subs_entry.get_text()),
                            'subsample_fraction': lambda: float(self.sub_frac_entry.get_text()),
                            'clustering_algs': lambda: [ self.clus_alg_widgets[x] for x in self.clus_alg_widgets if x.get_active() ],
                            'linkages': lambda: [ self.linkage_widgets[x] for x in self.linkage_widgets if x.get_active() ],
                            'final_alg': lambda: self.finalbutton.get_model()[self.finalbutton.get_active()][0],
                            'log2': self.log2box.get_active,
                            'sub_medians': self.submedbox.get_active,
                            'center': self.centerbox.get_active,
                            'scale': self.scalebox.get_active,
                            'pca_fraction': lambda: float(self.pca_frac_entry.get_text()),
                            'eigenvector_weight': lambda: float(self.eig_weight_entry.get_text()),
                            'norm_var': self.normvarbox.get_active,
                            'keep_list': lambda: self.keep_list,
                            'distance_metric': lambda: self.distbutton.get_model()[self.distbutton.get_active()][0]}
        
        self._set_defaults(args, kwds)

        window.show_all()
        
        gtk.gdk.threads_enter()
        gtk.main()
        gtk.gdk.threads_leave()

    @only_once
    def run_clustering(self, w, e=None):
        
        self.startbutton.set_sensitive(False)

        args = {}

        for setting in self.settings: #So only arguments from self.settings go to cluster...
            args[setting] = self.settings[setting]()

        if MPI_ENABLED:
            wake_nodes(1)
            
            if mpi.rank == 0:
                mpi.bcast((self.parser, self.filename, args))

        self.thread = Thread(target=self.consensus_procedure, args=(self.parser, self.filename), kwargs=args)
        self.thread.start()

    @only_once
    def destroy(self, w):

        if MPI_ENABLED:
    
            if mpi.rank == 0:
                if self.thread is None:
                    wake_nodes(1)
                    mpi.bcast(None)
                else:
                    wake_nodes(3)

        gobject.source_remove(self.pbar_timer)

        gtk.main_quit()

    @only_once
    def get_filename(self, w):

        def set_filename(w):
            self.filename = box.get_filename()
            self.parser = getattr(parsers, 'Parse' + parser_lst[button.get_active()])

            box.destroy()
            Thread(target = self._announce_fileparser).start()

        parser_lst = []

        #Get the names
        for key in parsers.__dict__:
            if key.find('Parse') == 0:
                pname = key[5:]

                if pname == 'Normal':           #Pretty sleazy way of hardcoding default
                    parser_lst.insert(0, pname)
                else:
                    parser_lst.append(pname)

        box = gtk.FileSelection("Select file")
        box.ok_button.connect("clicked", set_filename)
        box.cancel_button.connect("clicked", lambda w: box.destroy())
        box.set_resizable(False)

        button = gtk.combo_box_new_text()
        for text in parser_lst:
            button.append_text(text)

        button.set_active(0)
        
        parserbox = gtk.HBox(False)
        buttonlabel = gtk.Label('Select a Parser:')
        for obj in (buttonlabel, button):
            parserbox.pack_start(obj)

        box.action_area.pack_start(parserbox)

        box.show_all()

    @only_once
    def keep_list_dialog(self, w):

        chooser = gtk.FileChooserDialog('Open..', None, gtk.FILE_CHOOSER_ACTION_OPEN,
                                       (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_OPEN, gtk.RESPONSE_OK))

        chooser.set_select_multiple(True)
        chooser.set_default_response(gtk.RESPONSE_OK)
        chooser.set_size_request(640,480)

        filter = gtk.FileFilter()
        filter.set_name("All Files")
        filter.add_pattern("*")
        chooser.add_filter(filter)

        response = chooser.run()
        if response == gtk.RESPONSE_OK:
            self.keep_list = chooser.get_filenames()

        chooser.destroy()

        if self.keep_list is not None:
            Thread(target=self._announce_keep_list).start()

    def _set_defaults(self, args, kwds):

        #Widgety things
        if args and args[0] is not None and args[1] is not None:
            self.parser = args[0]
            self.filename = args[1]
            self._announce_fileparser()
        else:
            self.console.write('Welcome to ConsensusCluster!  Please select a file for reading.')

        clus_widget_algs = dict([ (v,k) for (k,v) in self.clus_alg_widgets.iteritems() ])
        widget_linkages  = dict([ (v,k) for (k,v) in self.linkage_widgets.iteritems() ])

        def set_kvalue(lst):
            self.k_min.set_text(str(lst[0]))
            self.k_max.set_text(str(lst[-1]))

        def set_clus(lst):
            for alg in lst:
                clus_widget_algs[alg].set_active(1)

        def set_link(lst):
            for lnk in lst:
                widget_linkages[lnk].set_active(1)

        set_widgets = {     'kvalues': set_kvalue,
                            'subsamples': lambda s: self.subs_entry.set_text(str(s)),
                            'subsample_fraction': lambda s: self.sub_frac_entry.set_text(str(s)),
                            'clustering_algs': set_clus,
                            'linkages': set_link,
                            'final_alg': lambda s: self.finalbutton.set_active([ x[0] for x in self.finalbutton.get_model() ].index(s)),
                            'log2': self.log2box.set_active,
                            'sub_medians': self.submedbox.set_active,
                            'center': self.centerbox.set_active,
                            'scale': self.scalebox.set_active,
                            'pca_fraction': lambda s: self.pca_frac_entry.set_text(str(s)),
                            'eigenvector_weight': lambda s: self.eig_weight_entry.set_text(str(s)),
                            'norm_var': self.normvarbox.set_active,
                            'distance_metric': lambda s: self.distbutton.set_active([ x[0] for x in self.distbutton.get_model() ].index(s)) }
    
        defaults = {        'kvalues': range(2,7),
                            'subsamples': 300,
                            'subsample_fraction': 0.8,
                            'clustering_algs': [ cluster.KMeansCluster ],
                            'linkages': [ 'average' ],
                            'final_alg': 'Hierarchical',
                            'log2': False,
                            'sub_medians': False,
                            'center': True,
                            'scale': False,
                            'pca_fraction': 1,
                            'eigenvector_weight': 0.25,
                            'norm_var': False,
                            'distance_metric': 'Pearson' }

        for key in defaults:
            if kwds.has_key(key):
                set_widgets[key](kwds[key])
            else:
                set_widgets[key](defaults[key])
    
    def _upd_pbar(self):

        self.progress.set_fraction(self.console.progress_frac)

        return True

    def _announce_fileparser(self):

        self.console.write("File '%s' selected for reading, using %s" % (self.filename, self.parser.__name__))
        self.console.success()

    def _announce_keep_list(self):

        self.console.write("The sample ids in the following files are defined as clusters:")
        for file in self.keep_list:
            self.console.write(file)
        self.console.success()


class CommonCluster(Gtk_UI):
    """

    CommonCluster

        Common class

        This class presents a default workflow for Consensus Clustering.  It is designed to be subclassed to suit your needs.

        See individual methods for advice on appropriate subclassing methodology.

        Usage:
            
            class MyCluster(CommonCluster):

                def __init__(self, parser, filename, log2, sub_medians, center, scale, pca_fraction, eigenvector_weight,
                             kvalues, subsamples, subsample_fraction, norm_var, **kwds):
                    
                    CommonCluster.__init__(self, parser, filename, ....

            Or simply CommonCluster(parser, filename, ....
            
            In either case, CommonCluster will be run with the following options:

                parser              - parsers.ParseX class, see parsers.py.  No default.
                filename            - File to be parsed by parser, see parsers.py.  No default.
                log2                - Take the log2 of all data.  Default: False
                sub_medians         - Subtract the median of sample medians from each entry in M.  Default: False
                center              - Normalise genes over all samples to have mean 0.  Default: True
                scale               - Normalise genes over all samples by dividing by the Root-Mean-Square.  Default: False
                pca_fraction        - Choose genes from those principle components that explain pca_fraction of the variance.  Default: 0.85
                eigenvector_weight  - Take the top eigenvector_weight fraction of genes that occur with high weights in selected components.  Default: 0.25
                kvalues             - List of K-Values to cluster.  Default: [2, 3, 4, 5, 6]
                subsamples          - Number of subsampling iterations to form consensus.  Default: 300
                subsample_fraction  - Fraction of samples/genes to cluster each subsample.  Default: 0.8
                norm_var            - Boolean variable.  If True, genes will be standardised to have variance 1 over all samples
                                      each clustering iteration.  Default: False

    """
    
    def __init__(self, parser, filename, **kwds):

        if not hasattr(self, 'console'):
            self.console = display.ConsoleDisplay(log=False)

        self.set_kwds(**kwds) #Allows subclassing methods to pass along variables that aren't set in the UI.

        if len(sys.argv) > 1:
            parser, filename, settings = self.handle_cmdline_args(parser, filename, **kwds)
            kwds.update(settings)

        if not self.use_gtk:
            self.consensus_procedure(parser, filename, **kwds)
        else:
            Gtk_UI.__init__(self, parser, filename, **kwds)

    def set_kwds(self, keep_list = None, pca_only = False, pca_legend = True, use_gtk = True, no_pca = False, coring = False, **kwds):

        self.keep_list = keep_list   #List of filenames of samples to keep, usually set by UI
        self.pca_only = pca_only   #Do PCA and then stop
        self.pca_legend = pca_legend  #Draw the PCA legend?
        self.use_gtk = use_gtk
        self.no_pca = no_pca
        self.coring = coring

    def handle_cmdline_args(self, parser, filename, **kwds):
        """

        Should be self-explanatory.  An unrecognised option or -h will pull up the usage.

        """
        #TODO
        #Add probe list
        #Add write_ratio

        avail = dict.fromkeys(['-f', '-p', '-d', '-c', '-h', '--nopca', '--log2', '--nolog2', '--center', '--nocenter',
                               '--scale', '--noscale', '--submedians', '--nosubmedians',
                               '--pcafraction', '--eigweight', '--krange', '--subsamples', '--subfraction', '--normvar',
                               '--nonormvar', '--snr', '--comparelogs', '--plist', '--noselection', '--help',
                               '--kmeans', '--som', '--pam', '--hier', '--euclidian', '--corr', '--coring'])

        @only_once
        def usage(unrec=None):

            if unrec is not None:
                print("Unrecognised option: %s\n" % unrec)

            print("USAGE: common.exe (or python common.py) [OPTIONS]\n")
            print("\t-f <filename>\t\t\tLoad <filename> for clustering.  Default Parser: Normal")
            print("\t-p <parser>\t\t\tParse <filename> with <parser>.  Only valid with the -f option.  E.g. 'Normal'")
            print("\t-d\t\t\t\tDon't init display, run from console. This happens automatically if there is no\n\t\t\t\t\tdisplay or the required libraries are unavailable.")
            print("\t--help, -h\t\t\tThis help.")
            
            print("\n\t**DATA NORMALISATION**\n")
            print("\t--log2, --nolog2\t\tPerform log2 reexpression, or turn it off. Default is off.")
            print("\t--submedians, --nosubmedians\tPerform median centring, or turn it off. Default is off.\n\t\t\t\t\tNOTE: Turning this on will turn off mean centring.")
            print("\t--center, --nocenter\t\tPerform mean centring, or turn it off. Default is on.\n\t\t\t\t\tNOTE: Turning this on will turn off median centring.")
            print("\t--scale, --noscale\t\tPerform RMS scaling, or turn it off. Default is off.")
            print("\t--normvar, --nonormvar\t\tNormalise variance to 1 for each subsample, or turn it off. Default is off.")
            
            print("\n\t**PCA AND FEATURE SELECTION**\n")
            print("\t--nopca\t\t\t\tDo not perform PCA at all. This precludes feature selection.\n\t\t\t\t\tUseful if your data is known to be singular.")
            print("\t--pcafraction <fraction>\tSelect features from the top <fraction> principle components. Default is 0.85")
            print("\t--eigweight <fraction>\t\tSelect the top <fraction> features by weight in each principle component.\n\t\t\t\t\tDefault is 0.25")
            print("\t--noselection\t\t\tDo not perform feature selection. Simply sets pcafraction and eigweight to 1.")
            
            print("\n\t**SAMPLE SELECTION**\n")
            print("\t-c <file1 file2 file3 ..>\tDefine samples (one on each line) in file1, etc as clusters.  Sample set will be\n\t\t\t\t\treduced to these samples, and their labels will be shown in logs and PCA plot.")
            print("\t--krange <fst> <snd>\t\tRepeat for each kvalue between <fst> and <snd> inclusive. Default is 2 to 6.")
            print("\t--subsamples  <number>\t\tNumber of clustering iterations to perform. Default is 300.")
            print("\t--subfraction <fraction>\tSelect a random <fraction> of the samples each iteration. Default is 0.80")

            print("\n\t**CLUSTERING OPTIONS**\n")
            print("\t--kmeans\t\t\tRun the K-Means algorithm")
            print("\t--som\t\t\t\tRun the Self-Organising Map algorithm")
            print("\t--pam\t\t\t\tRun the Partition Around Medoids algorithm")
            print("\t--hier\t\t\t\tRun the Hierarchical Clustering algorithm. Note that this option adds the Hierarchical\n\t\t\t\t\talgorithm to clustering iterations, rather than the 'final' consensus clustering.")
            print("\t--euclid\t\t\tCluster using the Euclidean distance metric")
            print("\t--corr\t\t\t\tCluster using the Pearson Correlation distance metric")
            print("\t--coring\t\t\tTurns on EXPERIMENTAL coring support. Additional logfiles and images are generated which\n\t\t\t\t\tdetail suggested 'core' clusters. Take its advice at your own risk!")
            print("\n\n\tExample: python common.py -f mydata.txt -d --kmeans --log2 --submedians --noselection -c clusterdefs/*")
            print("\n\tOpens mydata.txt, log2 reexpresses and median centres the data, performs no feature selection, and begin k-means clustering using the cluster definitions in the clusterdefs folder without using the GUI.\n")

        RUN_SNR = 0
        RUN_PLIST = 0
        last_opt = None
        
        opts = {}       # Contains recognised options as keys and a list of their arguments as values
        settings = kwds # Handle settings normally set via the GUI
        algs = []       # Clustering algorithms

        args = sys.argv[1:]

        for i in xrange(len(args)):
            if args[i] in avail:
                last_opt = args[i]
                opts[last_opt] = []     #If we use setdefault, how will we know if someone does -h?
            else:
                if last_opt is None:
                    usage(args[i])
                    sys.exit(0)

                opts[last_opt].append(args[i])

        for opt in opts:
            if opt == '-f':
                filename = os.path.realpath(opts[opt][0])
                
                if '-p' not in opts:
                    parser = parsers.ParseNormal

            elif opt == '-p': parser = eval('parsers.Parse' + opts[opt][0])
            elif opt == '-d': self.use_gtk = False
            elif opt == '-c': self.keep_list = [ os.path.realpath(x) for x in opts[opt] ]
            elif opt == '--coring': self.coring = True
            elif opt == '--nopca': self.no_pca = True
            elif opt == '--log2': settings['log2'] = True
            elif opt == '--kmeans': algs.append(cluster.KMeansCluster)
            elif opt == '--som': algs.append(cluster.SOMCluster)
            elif opt == '--pam': algs.append(cluster.PAMCluster)
            elif opt == '--hier': algs.append(cluster.HierarchicalCluster)
            elif opt == '--corr': settings['distance_metric'] = 'Pearson'
            elif opt == '--euclid': settings['distance_metric'] = 'Euclidean'
            elif opt == '--noselection':
                settings['eigenvector_weight'] = 1.0
                settings['pca_fraction'] = 1.0
            elif opt == '--submedians':
                settings['sub_medians'] = True
                settings['center'] = False
            elif opt == '--center':
                settings['center'] = True
                settings['sub_medians'] = False
            elif opt == '--scale': settings['scale'] = True
            elif opt == '--nolog2': settings['log2'] = False
            elif opt == '--nosubmedians': settings['sub_medians'] = False
            elif opt == '--nocenter': settings['center'] = False
            elif opt == '--noscale': settings['scale'] = False
            elif opt == '--pcafraction': settings['pca_fraction'] = float(opts[opt][0])
            elif opt == '--eigweight': settings['eigenvector_weight'] = float(opts[opt][0])
            elif opt == '--krange': settings['kvalues'] = range(int(opts[opt][0]), int(opts[opt][1])+1)
            elif opt == '--subsamples': settings['subsamples'] = int(opts[opt][0])
            elif opt == '--subfraction': settings['subsample_fraction'] = float(opts[opt][0])
            elif opt == '--normvar': settings['norm_var'] = True
            elif opt == '--nonormvar': settings['norm_var'] = False
            elif opt == '--comparelogs':
                import log_analyse

                log1 = opts[opt][0]
                log2 = opts[opt][1]
    
                log1_dict = log_analyse.gen_cluster_dict(log1)
                log2_dict = log_analyse.gen_cluster_dict(log2)
                
                log_analyse.compare(log1_dict, log2_dict, log1, log2)
                log_analyse.compare(log2_dict, log1_dict, log2, log1)
                sys.exit(0)

            elif opt == '--snr': RUN_SNR = 1

            #TODO:  This one requires passing a modified sdata, which would get overwritten later...
            #       I usually get around this in scripts by subclassing _preprocess, but I'm not sure where we could do it here
            elif opt == '--plist': RUN_PLIST = 1
            elif opt == '-h' or opt == '--help':
                usage()
                sys.exit(0)

        if RUN_SNR:
            if filename is None:
                print("To use SNR list generation, please use the -f flag (and optionally -p) to select a dataset\n")
                print("USAGE: python common.py -f mydata.txt --snr <outfile> <cluster1> <cluster2>\n")
                sys.exit(0)
               
            s = parser(filename)

            for k in ('log2', 'sub_medians', 'center', 'scale'):
                if k == 'center' and 'center' not in settings:
                    settings['center'] = True
                else:
                    settings.setdefault(k, False)

            s.normalise(log2=settings['log2'], sub_medians=settings['sub_medians'], center=settings['center'], scale=settings['scale'])

            outfile, clust1, clust2 = opts['--snr'][:3]

            scripts.write_ratio(s, clust1, clust2, outfile, pval_threshold=0.05, snr_threshold=0.5, ttest=True)
            sys.exit(0)

        if algs:
            settings['clustering_algs'] = algs

        return parser, filename, settings

    def consensus_procedure(self, parser, filename, log2=False, sub_medians=False, center=True, scale=False, pca_fraction=0.85, eigenvector_weight=0.25,
                 kvalues=range(2,7), subsamples=300, subsample_fraction=0.8, norm_var=False, keep_list=None, **kwds):
        """
    
        Initialise clustering procedure and tell the user what's going on.
    
        Don't worry if MPI fails.  It's supposed to if you aren't using it.
    
        """
        
        console = self.console
        
        if keep_list is not None:
            self.keep_list = keep_list

        try:

            @only_once
            def newdir():
                cdir   = os.path.realpath(os.curdir)
                tstamp = time.strftime("%Y-%m-%d %H.%M.%S")
                os.mkdir(tstamp)
                os.chdir(cdir + os.path.sep + tstamp)
                return cdir

            cdir = newdir()

            if parser is None or filename is None:
                raise ValueError, 'No parser or no filename selected!'
    
            self.sdata = console.announce_wrap('Parsing data...', parser, filename, console)
    
            if self.keep_list is not None:
                self.sdata, self.defined_clusters = console.announce_wrap('Removing samples not found in %s...' % ", ".join(self.keep_list), scripts.scale_to_set, self.sdata, *self.keep_list)
    
            console.announce_wrap('Preprocessing data...', self._preprocess)
    
            idlist = [ x.sample_id for x in self.sdata.samples ]

            import pprint
            print "idlist, sdata"
            # pprint.pprint(self.sdata)
            # pprint.pprint(idlist)
            # pprint.pprint(dict.fromkeys(idlist))

            if len(dict.fromkeys(idlist)) != len(idlist):
                raise ValueError, 'One or more Sample IDs are not unique!\n\n\nHere is the data'+str(len(dict.fromkeys(idlist)))+'len'+str(len(idlist))
    
            console.announce_wrap('Running PCA...', self.run_pca, log2, sub_medians, center, scale, pca_fraction, eigenvector_weight, self.pca_legend, self.no_pca)
            
            if not self.pca_only:
                console.announce_wrap('Postprocessing data...', self._postprocess)
        
                console.write("Using MPI?")
            
                if MPI_ENABLED:
                    console.success()
                else:
                    console.fail()
        
                for i in kvalues:
                    self.run_cluster(i, subsamples, subsample_fraction, norm_var, kwds)

        except:
            console.except_to_console(str(sys.exc_info()[1]))

        self._complete_clustering(cdir, kwds)

    @only_once
    def makeplot(self, M, V, label, pca_legend=True, defined_clusters=None):
        """
    
        Use matplotlib and display.py's Plot function to draw the samples along the first two Principle Components
    
        Usage: makeplot(sdata, V, label)
            
            sdata   - parsers.Parse object containing sample points and data in numpy.array form
            V       - The eigenvectors of the covariance matrix as determined by SVD
            label   - The filename will be of the form "label - timestamp.png"
            defined_clusters - A dict of cluster ids and their sample_ids.  This overrides those defined in the GUI, if any.
    
        If matplotlib isn't installed, this function will simply do nothing.
    
        WARNING:    Depending on how the matrix is decomposed you may find different, but also correct, values of V
                    This will manifest itself as the same plot, but reflected in one or both directions
        
        """
    
        plots = []
        legend = []
    
        N = numpy.dot(V[:2], numpy.transpose(M))
    
        if defined_clusters is None and hasattr(self, 'defined_clusters'):
            defined_clusters = self.defined_clusters
            
        if defined_clusters is not None:

            indices = {}
            total_ind = 0
            sample_ids = [ x.sample_id for x in self.sdata.samples ]
            
            for cluster_id in defined_clusters:
                indices[cluster_id] = scripts.union(sample_ids, defined_clusters[cluster_id])[0]
                total_ind += len(indices[cluster_id])

            for cluster in indices:
                plot = N.take(tuple(indices[cluster]), 1)
        
                if plot.any():
                    plots.append(plot)
                    legend.append(cluster)

            if total_ind < len(self.sdata.samples): #Unlabeled samples?
                leftover = numpy.setdiff1d(xrange(len(self.sdata.samples)), sum([ indices[id] for id in indices ], []))
                
                plot = N.take(tuple(leftover), 1)

                plots.append(plot)
                legend.append('Unlabeled')

        else:
            #No kept files, just do as you're told
            legend = None
            plots = [N]     

        if not pca_legend:
            legend = None
    
        display.Plot(plots, legend = legend, fig_label = label)
    
    def run_pca(self, log2, sub_medians, center, scale, pca_fraction, eigenvector_weight, pca_legend=True, no_pca=False):
        """
    
        Create a matrix from self.sdata.samples, normalise it, and then run PCA to reduce dimensionality.
    
        Usage: self.run_pca(log2, sub_medians, center, scale, pca_fraction, eigenvector_weight)
    
            log2                - Take the log2 of all data.
            sub_medians         - Subtract the median of sample medians from each entry in M.
            center              - Normalise genes over all samples to have mean 0.
            scale               - Normalise genes over all samples by dividing by the Root-Mean-Square.
            pca_fraction        - Choose genes from those principle components that explain pca_fraction of the variance.
            eigenvector_weight  - Take the top eigenvector_weight fraction of genes that occur with high weights in selected components.

        This function runs makeplot once the data has been normalised.
        A logfile called "PCA results - timestamp.log" will be created with PCA result information.

        Note:

            MPI compatibility has changed somewhat in recent times.  Now, in order to save memory, only rank 0 performs PCA while
            other nodes wait in sleep timers.  Once PCA has completed, the nodes are woken and the reduced data is broadcast.
    
        """

        def reduce_dims(M, gene_indices):
            
            self.sdata.M = M

            if hasattr(self.sdata, 'gene_names') and len(self.sdata.gene_names):
                self.sdata.gene_names = self.sdata.gene_names.take(gene_indices)

                console.new_logfile('PCA Results - Feature list')
                console.log("\nReliable features:\n", display=False)
                
                for name in self.sdata.gene_names:
                    console.log("%s" % name, display=False)
    
        console = self.console

        if MPI_ENABLED:
            sleep_nodes(2)

            if mpi.rank != 0:
                M, gene_indices = mpi.bcast()    #Block until we get the go-ahead from 0
                reduce_dims(M, gene_indices)
                return
    
        console.new_logfile('PCA results')
        
        M = self.sdata.M
    
        console.log("Normalising %sx%s matrix" % (len(self.sdata.samples), len(self.sdata.M[0])))
        print "Executing NOt here", self.sdata.M
    
        M = pca.normalise(M, log2=log2, sub_medians=False, center=center, scale=scale) #PCA requires & performs mean centering, so it makes sense to subtract medians afterwards
        
        print "Executing here"
        if not self.no_pca:        
            
            V, gene_indices = pca.get_pca_genes(M, pca_fraction, eigenvector_weight)

            console.log("Found %s principle components in the top %s fraction" % (len(V), pca_fraction))
            console.log("Found %s reliable features occurring with high weight (top %s by absolute value)" % (len(gene_indices), eigenvector_weight))
            
            self.makeplot(M, V, 'PCA results - PC2v1 - All samples', pca_legend)
            self.makeplot(M, V[1:], 'PCA results - PC3v2 - All samples', pca_legend)
        
            #Reduce dimensions
            M = M.take(gene_indices, 1)

        else:
            #Ugly...
            gene_indices = tuple(range(M.shape[1]))
        
        if sub_medians:
            M = pca.subtract_medians(M)

        if MPI_ENABLED:
            wake_nodes(2)
    
            if mpi.rank == 0:
                mpi.bcast((M, gene_indices))

        reduce_dims(M, gene_indices)
    
    def run_cluster(self, num_clusters, subsamples, subsample_fraction, norm_var, kwds):
        """
    
        Run the clustering routines, generate a heatmap of the consensus matrix, and fill the logs with cluster information.
    
        Each time this is run it will create a logfile with the number of clusters and subsamples in its name.  This contains
        information on which samples where clustered together for that particular K value.
    
        Usage: self.run_cluster(num_clusters, subsamples, subsample_fraction, norm_var, kwds)
    
            num_clusters        - K value, or the number of clusters for the clustering functions to find for each subsample.
            subsamples          - The number of subsampling iterations to run.  In each subsample, the genes, samples, or both may
                                  be randomly selected for clustering.  This helps to ensure robust clustering.  More subsamples, more
                                  robust clusters.
            subsample_fraction  - The fraction of SNPs, samples, or both to take each subsample.  0.8 is a good default.
            norm_var            - Boolen variable.  If True, , genes will be standardised to have variance 1 over all samples
                                  each clustering iteration.
            kwds                - Additional options to be sent to cluster.ConsensusCluster
    
        It's probably a very bad idea to subclass run_cluster.  The _report and _save_hmap functions are almost certainly what you want.
    
        """
   
        console = self.console
        console.new_logfile(logname = '%s clusters - %s subsamples' % (num_clusters, subsamples))
        
        console.log("\nSamples: %s" % len(self.sdata.samples))
    
        console.write("\nClustering data...")
    
        args = locals()
        del args['self']
        args.update(kwds)

        print args
        #Actual work
        clust_data = cluster.ConsensusCluster(self.sdata, **args)
        
        console.write("\n\nBuilding heatmap...")
        # colour_map = self._save_hmap(clust_data, **args)
        colour_map = None

        if display.HMAP_ENABLED:
            console.success()
        else:
            console.fail()

        console.write("Generating logfiles...")
        self._report(clust_data, colour_map=colour_map, **args)

        @only_once
        def core():
            #TESTING
            dc = {}
            
            M = clust_data.consensus_matrix.take(tuple(numpy.argsort(clust_data.reorder_indices)), 0)
            #for sam in clust_data.datapoints:
            #    dc.setdefault(str(sam.cluster_id), []).append(sam.sample_id)
            V, core_samples = pca.get_pca_genes(M, 0.85, 0.15)
            
            for i in core_samples:
                sam = clust_data.datapoints[clust_data.reorder_indices[i]]
                dc.setdefault(str(sam.cluster_id), []).append(sam.sample_id)

            #dc['Core'] = [ clust_data.datapoints[clust_data.reorder_indices[x]].sample_id for x in core_samples ]
            
            console.new_logfile('%s cluster suggested cores' % num_clusters)

            for i in dc:
                console.log('\nCluster %s core:\n' % i, display=False)
            
                for sample_id in dc[i]:
                    console.log("\t%s" % sample_id, display=False)

            self.makeplot(M, V, '%s Cluster PCA Plot' % num_clusters, pca_legend = True, defined_clusters = dc)

        if self.coring:
            console.write("\nCreating Consensus PCA Plot...")
            core()

        clust_data._reset_clusters() #Set cluster_ids to None
    
    @only_once
    def _report(self, clust_data, console, **kwds):
        """

        _report is called by run_cluster after each clustering set at a particular k-value is complete.

        Its job is to inform the user which clusters went where.  This can be done to the screen and to the logfile using console.log()

        Subclassing:

            @only_once
            def _report(self, clust_data, console, **kwds):

                etc...

            clust_data.datapoints is a list of SampleData objects, each of which has a cluster_id attribute.  This attribute indicates
            cluster identity, and any SampleData objects that share it are considered to be in the same cluster.  This doesn't have to be
            1, 2, 3...etc.  In fact, it doesn't have to be a number.

            See display.ConsoleDisplay for logging/display usage.

            You may want to subclass _report if you want to report on additional information, or if you simply want to turn this logging feature off.

        """

        #SNR Threshold
        threshold = 0.5
        
        #Initialise the various dictionaries
        colour_map = kwds['colour_map']

        if hasattr(self, 'defined_clusters'):
            sample_id_to_cluster_def = {}

            for cluster_def in self.defined_clusters:
                for sample_id in self.defined_clusters[cluster_def]:
                    sample_id_to_cluster_def[sample_id] = cluster_def

        cluster_sample_ids = dict()
        cluster_sample_indices = dict()
    
        for clust_obj in [ (clust_data.datapoints[x].sample_id, clust_data.datapoints[x].cluster_id, x) for x in clust_data.reorder_indices ]:
            sample_id, cluster_id, sample_idx = clust_obj

            cluster_sample_ids.setdefault(cluster_id, []).append(sample_id)
            cluster_sample_indices.setdefault(cluster_id, []).append(sample_idx)
    
        #Start writing the log
        console.log("\nClustering results")
        console.log("---------------------")
    
        console.log("\nNumber of clusters: %s\nNumber of subsamples clustered: %s\nFraction of samples/features used in subsample: %s" % (kwds['num_clusters'], kwds['subsamples'], kwds['subsample_fraction']))
        console.log("\n---------------------")
        console.log("\nClusters")

        cluster_list = list(enumerate(cluster_sample_ids)) #(num, cluster_id) pairs

        print cluster_list

        for cluster in cluster_list:
            cluster_num, cluster_id = cluster

            console.log("\nCluster %s :\n" % (cluster_num))
    
            for sample_id in cluster_sample_ids[cluster_id]:
                if hasattr(self, 'defined_clusters'):
                    console.log("\t%s\t\t%s" % (sample_id, sample_id_to_cluster_def[sample_id]))
                else:
                    console.log("\t%s" % sample_id)

        M = clust_data.M
        
        buffer = []
        clsbuffer = []
        
        if hasattr(self.sdata, 'gene_names'):
            
            for i, j in comb(xrange(len(cluster_list)), 2):
                clust1, clust2 = cluster_list[i], cluster_list[j] #Still num, id pairs
                
                ttest = True # v0.5: On by default
                if kwds.has_key('ttest'):
                    ttest = kwds['ttest']

                ratios = pca.snr(M, cluster_sample_indices[clust1[1]], cluster_sample_indices[clust2[1]], threshold=threshold, significance=ttest)
                
                if ratios:
                    buffer.append("\nCluster %s vs %s:" % (clust1[0], clust2[0]))
                    buffer.append("--------------------\n")
                    buffer.append("Gene ID\t\tCluster %s Avg\tCluster %s Avg\tSNR Ratio\tp-value" % (clust1[0], clust2[0]))

                    for ratio, gene_idx, mean1, mean2, pval in ratios:
                        buffer.append("%10s\t\t%f\t\t%f\t\t%f\t\t%s" % (self.sdata.gene_names[gene_idx], mean1, mean2, ratio, pval))

                if kwds.has_key('classifier') and kwds['classifier'] and ratios:
                    clsbuffer.append("\nCluster %s vs %s:" % (clust1[0], clust2[0]))
                    clsbuffer.append("--------------------\n")

                    classif_list = pca.binary_classifier(M, cluster_sample_indices[clust1[1]], cluster_sample_indices[clust2[1]], threshold)
                    #Returns (a, b), where a is w in (wi, i) pairs and b is w0
                    clsbuffer.append("w0 is %s" % classif_list[1])
                    clsbuffer.append("\nGene ID\t\tMultiplier")

                    for result in classif_list[0]:
                        clsbuffer.append("%10s\t%f" % (self.sdata.gene_names[result[1]], result[0]))
        
        def write_buffer(name, desc, buf):
            console.new_logfile(name)
            console.log(desc, display=False)

            for line in buf:
                console.log(line, display=False)

        if buffer:
            write_buffer('SNR Results - %s clusters - %s subsamples' % (kwds['num_clusters'], kwds['subsamples']), "SNR-ranked features with ratio greater than %s" % threshold, buffer)

        if clsbuffer:
            write_buffer('Binary Classifier - %s clusters - %s subsamples' % (kwds['num_clusters'], kwds['subsamples']), "Based on SNR-ranked features with ratio greater than %s" % threshold, clsbuffer)
    
    @only_once
    def _save_hmap(self, clust_data, **kwds):
        """

        _save_hmap uses display.Clustmap to produce a heatmap/dendrogram of the consensus matrix produced by cluster.ConsensusCluster

        Subclassing:

            @only_once
            def _save_hmap(self, clust_data, **kwds):

                etc

            Really, the best reason to subclass _save_hmap is to change the heatmap labels.  See display.Clustmap for additional syntax.

            example: display.Clustmap(clust_data, [ clust_data.datapoints[x].sample_class for x in clust_data.reorder_indices ]).save('Consensus Matrix')

                ...will create a file called Consensus Matrix.png, which contains the consensus matrix heatmap labeled by sdata.samples[x].sample_class.

            clust_data.reorder_indices is (predictably) a list of indices which constitute the best order.  Since cluster.ConsensusCluster
            reorders the consensus matrix (clust_data.consensus_matrix) for you (but doesn't touch sdata.samples/clust_data.datapoints), you'll
            need to reorder the label list accordingly.  This can be just a list of labels as well, though once again you'll have to reorder your list
            to match reorder_indices.  A list comprehension of the general form [ labels[x] for x in clust_data.reorder_indices ] will do this for you,
            assuming labels is in the same order as sdata.samples/clust_data.datapoints.

            display.Clustmap.save() creates an image file and saves it to disk.
            display.Clustmap.show() opens a GTK window with the image.  Requires GTK.  See display.py.

        """

        filename = lambda s: "%s - %s clusters - %s subsamples" % (s, kwds['num_clusters'], kwds['subsamples'])

        if clust_data.datapoints[0].sample_class is not None:
            labels = [ " - ".join([str(clust_data.datapoints[x].sample_id), str(clust_data.datapoints[x].sample_class)]) for x in clust_data.reorder_indices ]
        else:
            labels = [ clust_data.datapoints[x].sample_id for x in clust_data.reorder_indices ]

        map = display.Clustmap(clust_data, labels)
        map.save(filename('Consensus Matrix'))

        return map.colour_map

    def _preprocess(self):
        """

        Any data preprocessing that needs to be done BEFORE PCA should be done by subclassing this method

        Subclassing:

            def _preprocess(self):

                etc

        _preprocess shouldn't return anything, so any preprocessing should be done by extracting self.sdata.sample[x].data objects and
        putting them back when you're done.

        example: Take sequence data found by parser and convert it into a binary agreement matrix by comparing it to some reference
                 sequence
        
        This method does nothing on its own.

        """

        pass

    def _postprocess(self):
        """

        Any data postprocessing that needs to be done AFTER PCA but BEFORE CLUSTERING should be done by subclassing this method

        Subclassing:

            def _postprocess(self):

                etc

        _postprocess shouldn't return anything, so any postprocessing should be done by extracting self.sdata.sample[x].data objects and
        putting them back when you're done.

        example: Choose a random subset of the data to cluster, rather than the entire set

        This method does nothing on its own.

        """

        pass

    def _complete_clustering(self, cdir, kwds):
        """

        Run when the clustering finishes.

        Right now it just ungreys the button and resets graphical things.

        """

        if cdir is not None:
            os.chdir(cdir)

        if display.DISPLAY_ENABLED and self.use_gtk: 

            if hasattr(self, 'startbutton'):
                self.startbutton.set_sensitive(True)

            self.mpi_wait_for_start()


if __name__ == '__main__':

    parser   = None
    filename = None

    args = {}

    CommonCluster(parser, filename, **args)
