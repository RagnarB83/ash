import KNARRsettings


# Author: Vilhjalmur Asgeirsson, 2019.

def PrintDate():
    string1, string2 = GetDate()
    print('Time: %s Date: %s' % (string2, string1))
    return None


def GetDate():
    import datetime
    date = datetime.datetime.now()
    string1 = date.strftime('%d.%m.%Y')
    string2 = date.strftime('%H:%M')
    return string1, string2


def PrintLogo():
    print("               K N A R R       ")
    print("                ___|__         ")
    print("                \-\-\-\        ")
    print("                /_/_/_/   p>   ")
    print("            (\_____|_____/)    ")
    print("           ~~~~/ / / / /~~~    ")
    print("           ~~~~~~~~~~~~~~~~    ")
    print("                               ")
    return None


def PrintBunny():
    print("     /)/)    ")
    print("    (^.^)    ")
    print("   ('')('')  ")
    print("             ")
    return None


def PrintUnhappyBunny():
    print("     /)/)    ")
    print("    (v.v)    ")
    print("   ('')('')  ")
    print("             ")
    return None


def PrintVersion():
    version = '0.0 (dev version)'
    print('%s' % version)
    return None


def PrintHeader():
    PrintLogo()
    PrintVersion()
    print('Author: %s ' % 'Vilhjalmur Asgeirsson')
    print('University of Iceland')
    PrintDate()
    return None


def GetDivider():
    s = ['=='] * 25
    return '%2s' * 25 % tuple(s)


def PrintDivider():
    print("%s" % GetDivider())
    return None


def PrintCredit():
    print('')
    print('I would like to thank: ')
    print('          H. Jonsson, R. Bjornsson, B.O. Birgirsson, O-P Koistinen')
    print('for significant contributions to KNARR and its development')
    print ('')
    return None


def PrintJob(strengur):
    print('')
    s = ['**'] * 25
    print('%2s' * 25 % (tuple(s)))
    print(strengur.center(50))
    print('%2s' * 25 % (tuple(s)))
    print('')
    return None


def PrintJobDone(jobtype, time):
    print('')
    print('%s finished successfully in %s seconds!' % (jobtype, round(time, 2)))


def PrintCallBack(fname, calculator=None, atoms=None, optimizer=None, parameters=None):
    fname = fname + '.info'

    print('Job information file written: %s\n' % fname)

    with open(fname, "w") as f:
        f.write("** KNARR INFORMATION FILE ** \n")
        string1, string2 = GetDate()
        f.write("Time: %s Date: %s\n" % (string2, string1))
        divider = GetDivider()
        if calculator is not None:
            calcinfo = calculator.GetInfo()
            f.write("%s \n" % divider)
            f.write("Information about calculation: \n")
            f.write("%s \n" % divider)
            for i in calcinfo:
                f.write("%s \n" % i)
        if optimizer is not None:
            optimizerinfo = optimizer.GetInfo()
            f.write("%s \n" % divider)
            f.write("Information about optimizer: \n")
            f.write("%s \n" % divider)
            for i in optimizerinfo:
                f.write("%s \n" % i)

        if atoms is not None:
            atomsinfo = atoms.GetInfo()
            f.write("%s \n" % divider)
            f.write("Information about system: \n")
            f.write("%s \n" % divider)
            for i in atomsinfo:
                f.write("%s \n" % i)
        if parameters is not None:
            f.write("%s \n" % divider)
            f.write("Job input parameters: \n")
            f.write("%s \n" % divider)
            for i in parameters:
                string_1 = i
                string_2 = parameters[i]
                f.write('%s = %s \n' % (string_1, string_2))

    return None


def PrintAvailableJobs():
    print('Available jobs:')
    for i in KNARRsettings.job_types:
        print(i.lower())
    return


def PrintMaxIter(maxiter):
    print('Maximum number of iterations reached (%i). Stopping optimization.' % maxiter)
    PrintUnhappyBunny()
    return None


def PrintConverged(iterations, forcecalls):
    print('')
    print('Calculation successfully converged in %i iterations and %i func. calls!' % (iterations, forcecalls))
    print('')
    return None
