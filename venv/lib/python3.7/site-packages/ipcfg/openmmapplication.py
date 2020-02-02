"""
Two base classes used by the openmm command line script. The first,
AppConfigurable, is a subclass of IPython's Configurable that adds a few
features, like stricter error checking, the concept that some of its traits
are 'active' and some are inactive, the ability to print out config files
in which the active traits are listed and the inactive traits are commented
out, etc.

The second class, OpenMMApplication, is a subclass of IPython's Application
that adds some features customized for openmm, like config file parsing that's
not based on a profile directory, but instead comes from a --config flag,
more complex validation, and the ability to auto-initialize the classes that
serve to hold its detailed configuration option. Those classes have their
traits automatically aliased on the command line.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import os
import sys
import copy
import logging

from .IPython import traitlets
from .IPython.traitlets import Bytes, Instance, List, TraitError, Unicode, CBool, CInt, CBytes, CFloat, Enum, Dict
from .IPython.configurable import Configurable, SingletonConfigurable
from .IPython.text import indent, dedent, wrap_paragraphs
from .IPython.loader import ConfigFileNotFound
from .ini_loader import IniFileConfigLoader
from .argparse_loader import ArgParseLoader


#-----------------------------------------------------------------------------
# Dirty Hacks
#-----------------------------------------------------------------------------

# Replace the function add_article in traitlets with a version that makes
# the error message thrown when a trait is invalid better.

# WITH HACK
# $ openmm --dt=1.0*A
# openmm: error: The 'dt' trait of the dynamics section must have units of
# femtosecond, but a value in units of angstrom was specified.

# WITHOUT HACK
# $ openmm --dt=1.0*A
# openmm: error: The 'dt' trait of a Dynamics must have units of femtosecond,
# but a value in units of angstrom was specified.

_super_traitlets_add_article = traitlets.add_article
def _traitlets_add_article(name):
    if name in [c.__name__ for c in AppConfigurable.__subclasses__()]:
        return 'the %s section' % name.lower()
    else:
        return _super_traitlets_add_article(object)
traitlets.add_article = _traitlets_add_article


# displaynames
CBool._displayname = 'Boolean'
CInt._displayname = 'Integer'
CBytes._displayname = 'String'
CFloat._displayname = 'Float'

#-----------------------------------------------------------------------------
# Classes
#-----------------------------------------------------------------------------

class LevelFormatter(logging.Formatter, object):
    """Formatter with additional `highlevel` record

    This field is empty if log level is less than highlevel_limit,
    otherwise it is formatted with self.highlevel_format.

    Useful for adding 'WARNING' to warning messages,
    without adding 'INFO' to info, etc.
    """
    highlevel_limit = logging.WARN
    highlevel_format = " %(levelname)s |"

    def format(self, record):
        if record.levelno >= self.highlevel_limit:
            record.highlevel = self.highlevel_format % record.__dict__
        else:
            record.highlevel = ""

        return super(LevelFormatter, self).format(record)


class OpenMMApplication(SingletonConfigurable):
    """Baseclass for the OpenMM application script, with the methods
    for printing the help text and loading the config file (boring stuff)
    """
    name = Unicode(u'Application')
    configured_classes = List()
    option_description = Unicode('')

    # The log level for the application
    log_level = Enum((0,10,20,30,40,50,'DEBUG','INFO','WARN','ERROR','CRITICAL'),
                    default_value=logging.WARN,
                    config=True,
                    help="Set the log level by value or name.")
    def _log_level_changed(self, name, old, new):
        """Adjust the log level when log_level is set."""
        if isinstance(new, basestring):
            new = getattr(logging, new)
            self.log_level = new
        self.log.setLevel(new)

    log_datefmt = Unicode("%Y-%m-%d %H:%M:%S", config=True,
        help="The date format used by logging formatters for %(asctime)s"
    )
    def _log_datefmt_changed(self, name, old, new):
        self._log_format_changed()

    log_format = Unicode("[%(name)s]%(highlevel)s %(message)s", config=True,
        help="The Logging format template",
    )
    def _log_format_changed(self, name, old, new):
        """Change the log formatter when log_format is set."""
        _log_handler = self.log.handlers[0]
        _log_formatter = LevelFormatter(new, datefmt=self.log_datefmt)
        _log_handler.setFormatter(_log_formatter)

    log = Instance(logging.Logger)
    def _log_default(self):
        """Start logging for this application.

        The default is to log to stderr using a StreamHandler, if no default
        handler already exists.  The log level starts at logging.WARN, but this
        can be adjusted by setting the ``log_level`` attribute.
        """
        log = logging.getLogger(self.__class__.__name__)
        log.setLevel(self.log_level)
        log.propagate = False
        _log = log # copied from Logger.hasHandlers() (new in Python 3.2)
        while _log:
            if _log.handlers:
                return log
            if not _log.propagate:
                break
            else:
                _log = _log.parent
        if sys.executable.endswith('pythonw.exe'):
            # this should really go to a file, but file-logging is only
            # hooked up in parallel applications
            _log_handler = logging.StreamHandler(open(os.devnull, 'w'))
        else:
            _log_handler = logging.StreamHandler()
        _log_formatter = LevelFormatter(self.log_format, datefmt=self.log_datefmt)
        _log_handler.setFormatter(_log_formatter)
        log.addHandler(_log_handler)
        return log

    # the alias map for configurables
    aliases = Dict({'log-level' : 'Application.log_level'})

    def __init__(self, **kwargs):
        SingletonConfigurable.__init__(self, **kwargs)
        # Ensure my class is in self.classes, so my attributes appear in command line
        # options and config files.
        if self.__class__ not in self.classes:
            self.classes.insert(0, self.__class__)


    def initialize(self, argv=None):
        '''Do the first steps to configure the application, including
        finding and loading the configuration file'''
        # load the config file before parsing argv so that
        # the command line options override the config file options
        if argv is None:
            argv = copy.copy(sys.argv)

        new_argv = []  # filtered version
        arg_iter = iter(argv)
        self.config_file_path = ''
        for arg in arg_iter:
            if arg.startswith('--config='):
                self.config_file_path = arg.split('=')[1]
            elif arg == '--config':
                self.config_file_path = next(arg_iter)
            else:
                new_argv.append(arg)

        argv = new_argv

        # if the user was using make_config or did not specify a path
        # to the config file, then don't error if no config file is found.
        error_on_no_config_file = not (any(a == 'make_config' for a in sys.argv) or len(self.config_file_path) == 0)
        self.load_config_file(self.config_file_path, error_on_not_exists=error_on_no_config_file)

        self.parse_command_line(argv)

    def initialize_configured_classes(self):
        for klass in filter(lambda c: c != self.__class__, self.classes):
            traitname = klass.__name__.lower()
            self.log.debug('Initializing %s options from config/command line.' % traitname)
            trait = self.class_traits()[traitname]
            if trait is None:
                raise AttributeError(
                    '''To use initialize_classes, you need to make Instance trait on your application
                    with the name of each of the items in classes that will be used to
                    hold the initialized value. I coulndn't find an Instance trait named
                    %s''' % traitname)

            if not isinstance(trait, Instance):
                raise AttributeError("%s needs to be an Instance trait" % trait)

            instantiated = klass(config=self.config)
            self.configured_classes.append(instantiated)
            setattr(self, traitname, instantiated)

    def validate(self):
        for cls in self.configured_classes:
            cls.validate()

    def load_config_file(self, filename, path=None, error_on_not_exists=False):
        """Load a .py based config file by filename and path."""
        loader = IniFileConfigLoader(filename, path=path)
        try:
            config = loader.load_config()
        except ConfigFileNotFound as e:
            # problem finding the file, raise
            if error_on_not_exists:
                self.error(e)
        except Exception as e:
            self.error(e, header=True)
        else:
            self.log.debug("Loaded config file: %s", loader.full_filename)
            self.update_config(config)

    def print_description(self):
        "Print the application description"
        lines = []
        lines.append('')
        lines.append(self.short_description)
        lines.append('=' * len(self.short_description))
        lines.append('')
        for l in wrap_paragraphs(self.long_description):
            lines.append(l)
            lines.append('')
        print os.linesep.join(lines)

    def print_options(self):
        if not self.aliases:
            return
        lines = ['Options']
        lines.append('-'*len(lines[0]))
        lines.append('')
        self.print_alias_help()
        print

    def print_help(self, classes=False):
        """Print the help for each Configurable class in self.classes.

        If classes=False (the default), only flags and aliases are printed.
        """
        #self.print_subcommands()
        self.print_options()

        if classes:
            # skip self, since it just contains logging information. we put
            # all the configurables into these classes, so that we get nice
            # panels for the user
            for cls in filter(lambda c: c != self.__class__, self.classes):
                cls.class_print_help()
                print
        else:
            print "To see all available configurables, use `--help-all`"
            print

    def print_alias_help(self):
        """Print the alias part of the help."""
        if not self.aliases:
            return

        lines = []
        classdict = {self.__class__.__name__: self.__class__}
        for cls in self.classes:
            # include all parents (up to, but excluding Configurable) in available names
            for c in cls.mro()[:-3]:
                classdict[c.__name__] = c

        for alias, longname in self.aliases.iteritems():
            classname, traitname = longname.split('.',1)
            cls = classdict[classname]

            trait = cls.class_traits(config=True)[traitname]
            help = cls.class_get_trait_help(trait).splitlines()
            # reformat first line
            help[0] = help[0].replace(longname, alias) #+ ' (%s)'%longname
            if len(alias) == 1:
                help[0] = help[0].replace('--%s='%alias, '-%s '%alias)
            lines.extend(help)
        # lines.append('')
        print os.linesep.join(lines)


    def error(self, message=None, header=False):
        "Error out with a message"
        if header:
            self.print_description()
            self.print_help()

        if message:
            self.log.error(str(message))
            self._print_message(
                '\nTo see all available configurables, use `--help-all`\n', sys.stderr)
        sys.exit(2)

    def exit(self, exit_status=0):
         self.log.debug("Exiting application: %s" % self.name)
         sys.exit(exit_status)

    def _print_message(self, message, file=None):
        if message:
            if file is None:
                file = sys.stderr
            file.write(message)

    def parse_command_line(self, argv=None):
        """Parse the command line arguments."""
        argv = sys.argv[1:] if argv is None else argv

        if argv and argv[0] == 'help':
            # turn `ipython help notebook` into `ipython notebook -h`
            argv = argv[1:] + ['-h']

        if any(x in argv for x in ('-h', '--help-all', '--help')):
            self.print_description()
            self.print_help('--help-all' in argv)
            self.exit(0)

        if '--version' in argv or '-V' in argv:
            self.print_version()
            self.exit(0)

        loader = ArgParseLoader(argv=argv, classes=self.classes, aliases=self.aliases)
        config = loader.load_config()
        self.update_config(config)
        # store unparsed args in extra_args
        self.extra_args = loader.extra_args


class AppConfigurable(Configurable):

    """Subclass of Configurable that ensure's there arn't any extraneous
    values being set during configuration.

    Also adds a validate() method that gets called during initialization
    that you can use to check that things get set correctly.

       """
    application = Instance('ipcfg.openmmapplication.OpenMMApplication')

    log = Instance('logging.Logger')
    def _log_default(self):
        return self.application.log

    def _application_default(self):
        return OpenMMApplication.instance()

    specified_config_traits = List(help='''List of the names of the traits on this
        Configurable that were set during initialization and did not just
        inherit their default value. (Which traits were actually set in the
        command line or config file).''')

    def active_config_traits(self):
        return self.class_traits(config=True).keys()

    def __init__(self, config={}):
        for key in config[self.__class__.__name__].keys():
            if key not in self.class_traits():
                self.application.error(
                    '%s has no configurable trait %s' % (self.__class__.__name__, key))

        super(AppConfigurable, self).__init__(config=config)
        self.specified_config_traits = config[self.__class__.__name__].keys()

    def validate(self):
        "Run any validation on the traits in this class"
        pass

    def config_section(self):
        """Get the config section with all of the active configurable traits
        placed in, and the inactive configurable traits commented out, in .ini
        format"""

        def c(s):
            """return a commented, wrapped block."""
            s = '\n\n'.join(wrap_paragraphs(s, 78))

            return '# ' + s.replace('\n', '\n# ')

        # section header
        breaker = '#' + '-' * 78
        klass = self.__class__.__name__
        lines = [breaker, '[%s]' % klass]

        # get the description trait
        desc = self.__class__.class_traits().get('description')
        if desc:
            desc = desc.default_value
        else:
            # no description trait, use __doc__
            desc = getattr(self.__class__, '__doc__', '')
        if desc:
            lines.append(breaker)
            lines.append(c(desc))

        lines.extend([breaker, ''])

        # all of the configurable traits that are currently activated
        active_config_traits = self.active_config_traits()

        for name, trait in self.__class__.class_traits(config=True).iteritems():
            help = trait.get_metadata('help') or ''
            lines.append(c(help))
            if 'Enum' in trait.__class__.__name__:
                lines.append(c(indent('Choices: [%s]' % (', '.join(trait.values,)))))

            item = '%s = %s' % (name, getattr(self, name))
            if name in active_config_traits:
                lines.append(item)
            else:
                if name in self.xml_override:
                    lines.append(c('NOTE: Deactivated because system is being loaded from an XML file.'))
                lines.append(c(item))
            lines.append('')
        return '\n'.join(lines)

    @classmethod
    def class_get_trait_help(cls, trait, inst=None):
        """Get the help string for a single trait.

        If `inst` is given, it's current trait values will be used in place of
        the class default.
        """
        assert inst is None or isinstance(inst, cls)
        lines = []
        if hasattr(trait.__class__, '_displayname'):
            traittype = trait.__class__._displayname
        else:
            traittype = trait.__class__.__name__

        header = "--%s <%s>" % (trait.name, traittype)
        lines.append(header)
        if inst is not None:
            lines.append(indent('Current: %r' % getattr(inst, trait.name), 4))
        else:
            try:
                dvr = repr(trait.get_default_value())
            except Exception:
                dvr = None  # ignore defaults we can't construct
            if dvr is not None:
                if len(dvr) > 64:
                    dvr = dvr[:61] + '...'
                lines.append(indent('Default: %s' % dvr, 4))
        if 'Enum' in trait.__class__.__name__:
            # include Enum choices
            lines.append(indent('Choices: %r' % (trait.values,)))

        help = trait.get_metadata('help')
        if help is not None:
            help = '\n'.join(wrap_paragraphs(help, 76))
            lines.append(indent(help, 4))
        return '\n'.join(lines)
