import os, os.path, errno, sys, glob, subprocess
import hashlib, pickle, time, multiprocessing, json
import numpy as np
from collections import namedtuple


####################################################################
######################## TAGS FOR OBSERVABLES AND DIRECTORIES
####################################################################

pi_binned = ['vr','taud','ml','vt','m','e']

def retag(tag):
    
    if tag == 'vr':
        return 'm'
    elif tag == 'vt':
        return 'm'
    elif tag == 'taud':
        return 'e'
    elif tag == 'isw_lin':
        return 'm'
    elif tag == 'lensing':
        return 'm'
    elif tag == 'ml':
        return 'm'
    else:
        return tag  
    
def direc(tag1,tag2, conf,N_bins = None, sigma_photo_z = None, A_electron = None, LSSexperiment = None):
    
    if N_bins == None:
        N_bins = conf.N_bins
    if sigma_photo_z == None:
        sigma_photo_z = conf.sigma_photo_z
    if A_electron == None:
        A_electron = conf.A_electron
    if LSSexperiment == None:
        LSSexperiment = conf.LSSexperiment
    
    if tag1 in pi_binned or tag2 in pi_binned or ('g' in [tag1,tag2] and LSSexperiment == 'LSST'):
        name = 'Pi_bins='+str(N_bins)
    else:
        name = ''
    
    if 'g' in [tag1,tag2]:  
        name += '/LSS='+str(LSSexperiment)
        if LSSexperiment == 'LSST':
            name +='/spz='+str(sigma_photo_z)
    if 'e' in [tag1,tag2] or 'taud' in [tag1,tag2]:
        name += '/Ae='+str(A_electron)
    return name


####################################################################
######################## DATA STORAGE AND MANIPULATION
####################################################################

excluded = ['N_bins','A_electron','LSSexperiment','sigma_photo_z','sigma_cal','beamArcmin_T'
            ,'noiseTuKArcmin_T','beamArcmin_pol','noiseTuKArcmin_pol','cleaning_mode']

def mkdir_p(path):
    """Recursively create a directory path"""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def is_primitive(val) :
    """ Check if value is a 'primitive' type"""
    primitive_types = [int, float, bool, str]
    return type(val) in primitive_types


def get_basic_conf(conf_module, exclude = True) :
    """
    Get dictionary of values in conf_module, excluding keys starting with '__',
    and include only values with True is_primitive(val)
    """
    d = conf_module.__dict__
    
    # Filter out keys starting with '__',
    # Make sure values are a "primitive" type
    new_dict = {}
    
    if exclude:
        for key, val in d.items() :
            if key[0:2] != "__" and key not in excluded and is_primitive(val) :
                new_dict[key] = val
    else:
        for key, val in d.items() :
            if key[0:2] != "__" and is_primitive(val) :
                new_dict[key] = val
          
    return new_dict


def dict_to_obj(basic_conf_dict) :
    return namedtuple("conf", basic_conf_dict.keys())(*basic_conf_dict.values())


def get_hash(basic_conf) :
    """Convert module -> dictionary with only 'primitive' members -> serialized string -> md5"""
    serialized__str = json.dumps(basic_conf, sort_keys=True).encode('utf-8')
    return hashlib.md5(serialized__str).hexdigest()


def get_output_directory(basic_conf, dir_base = '',) :
    basic_conf_id_str = get_hash(basic_conf)
    output_directory = "output/" + basic_conf_id_str + "/" + dir_base + "/"
    mkdir_p(output_directory)
    return output_directory


def dump(basic_conf, data, file_base, dir_base = '') :
    """
    Dump data to a path uniquely determined by a hashed basic_conf
    along with a metadata file containing (primitive) data from basic_conf.
    """
    if not exists(basic_conf, 'metadata') :
        write_basic_conf(basic_conf)

    # Output data
    output_directory = get_output_directory(basic_conf, dir_base)
    filename = output_directory + file_base + '.p'
    pickle.dump( data, open( filename, "wb" ) )


def write_basic_conf(basic_conf) :
    output_directory = get_output_directory(basic_conf)
    print("Writing basic_config in", output_directory)

    # Pickled basic_conf data for re-reading
    filename = output_directory + "metadata.p"
    pickle.dump( basic_conf, open( filename, "wb" ) )
    
    # Human-readable basic_conf data
    filename = output_directory + "metadata.txt"
    fout = open(filename, "w")
    for key, val in basic_conf.items():
        fout.write(str(key) + ' = '+ str(val) + '\n')
    fout.close()


def load_basic_conf(hashstr) :
    """Load basic conf data from a given hash string"""
    filename = "output/" + hashstr + "/metadata.p"
    if not os.path.isfile(filename) :
        raise Exception('Data associated with hash "'+str(hashstr)+'" not found.')
    data = pickle.load( open( filename, "rb" ) )
    return data


def load(basic_conf, file_base, dir_base = '') :
    """Load data from a path uniquely determined by a hashed basic_conf"""
    basic_conf_hash = get_hash(basic_conf)
    output_directory = get_output_directory(basic_conf, dir_base)
    filename = output_directory + file_base + '.p'
    data = pickle.load( open( filename, "rb" ) )
    return data

def timed_load(basic_conf, file_base, dir_base = '') :
    """Load data from a path uniquely determined by a hashed basic_conf"""
    basic_conf_hash = get_hash(basic_conf)
    print("Loading "+file_base+" data. Hash is", basic_conf_hash)
    start = time.time()
    data = load(basic_conf, file_base, dir_base)
    end = time.time()
    print("Done loading "+file_base+" in", end-start, "seconds.")
    return data


def exists(basic_conf, file_base, dir_base = '') :
    """Check if data exists at a path"""
    output_directory = get_output_directory(basic_conf, dir_base)
    filename = output_directory + file_base + '.p'
    return os.path.isfile(filename)


def plots_path(basic_conf, dirname) :
    basic_conf_hash = get_hash(basic_conf)
    # Make plots directory
    plots_path = "output/"+basic_conf_hash+"/plots/" + dirname + "/"
    mkdir_p(plots_path)
    return plots_path


def get_n_cores() :
    cpu_count = multiprocessing.cpu_count()
    if cpu_count > 8 :
        # probably on a cluster node, use the whole thing
        return cpu_count
    else :
        # probably something local, save a couple cores
        return max(1, int( cpu_count - 2 ))
    
    
def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()
    

