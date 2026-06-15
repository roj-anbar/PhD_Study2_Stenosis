# -----------------------------------------------------------------------------------------------------------------------
# oasis_problem_PT.py 
# This is the problem definition for Oasis/NSfracStep.py (patient-specific arteries/veins).
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2025-09
#
# PURPOSE:
#   - To provide case specific problem parameters (e.g., mesh loading, boundary conditions, time hooks) for Oasis NSfracStep solver (legacy FEniCS).
#
# REQUIREMENTS:
#   - Oasis package available locally: https://github.com/mikaem/Oasis.git
#   - BSLSolver available locally: https://github.com/Biomedical-Simulation-Lab/BSLSolver.git
#
# EXECUTION:
#   - This script is used as a problem file when calling the Oasis solver (as in the oasis_solver_PT.sh).
#   - <mpirun -n $NP oasis NSfracStep problem=oasis_problem_PT>.
#   - Note: This script cannot be ran directly.
#
# INPUTS:
#   - mesh_name            : basename of mesh in ./data (expects .xml.gz and .info)
#   - period               : waveform period [ms]
#   - timesteps            : time steps per cycle
#   - cycles               : number of cycles
#   - viscosity (nu)       : kinematic viscosity [mm^2/ms] (≡ m^2/s in consistent units)
#   - uOrder               : velocity polynomial order for FE
#   - save_frequency       : save every N steps
#   - checkpoint           : write restart every N steps
#   - inlet_BC_type        : type of the inlet boundary condition --> choose from: {'pulsatile', 'ramp', 'custom'} (default is 'pulsatile')
#
# Optional
#   - restart_folder       : previous results folder to restart from
#   - zero_pressure_outlets: set True to force 0-pressure at outlets
#   - include_gravity, flat_profile_at_intlet_bc (typo kept for compatibility)
#
# OUTPUTS:
#   - Results written under ./results/{case_fullname}.
#   - Per-step HDF5 + XDMF via BSLSolver.common.h5io.
#
# NOTES:
#   - Units in comments follow Oasis code: mm, ms, mL/s (consistent with FEniCS fields).
#   - Keep behavior compatible with existing postprocessing.
#
# Adapted from Artery.py originally written by Mehdi Najafi (2018) and Anna Haley (2022). 
# Copyright (C) 2025 University of Toronto, Biomedical Simulation Lab.
# -----------------------------------------------------------------------------------------------------------------------

__authors__   = "Mehdi Najafi <mnuoft@gmail.com>. Anna Haley <ahaley@mie.utoronto.ca>. Rojin Anbarafshan <rojin.anbar@gmail.com>"
__date__      = "2018-2025"
__copyright__ = "Copyright (C) 2018-2025 UofT"
__license__   = "Private"

# ---------------------------------------- Imports and Basic Setup -----------------------------------------------------
from oasis import * 
from oasis.problems import * 
from oasis.problems.NSfracStep import *

from dolfin import *
import numpy as np
import sys, time, os, glob, pickle
from os import getpid, path, makedirs, getcwd
#from probe import * - this is never used and probably out of date AH

from BSLSolver import *
from BSLSolver.common import CustomFunction, h5io, Womersley
#from BSLSolver.common import naming
#from BSLSolver.common import WSS #only works in serial

# MPI Setup
mpi_size = MPI.size(MPI.comm_world)
mpi_rank = MPI.rank(MPI.comm_world)


# Setup timer
if mpi_rank == 0:
    from dolfin import Timer
    _global_timer = Timer()
    initial_wall_time = _global_timer.elapsed()[0]

# I/O helper from BSLSolver
h5stdio = h5io.HDF5StdIO()

# Reasonable wall-time guard (in seconds)
max_wtime_before_kill = (23.5*60*60)


# ---------------------------------------- Utilities Functions -----------------------------------------------------

#//////////////////////////////////////////////////////////////////////
def mpi_comm():
    return MPI.comm_world


#//////////////////////////////////////////////////////////////////////
def print_section_header(title, width=100):
    if mpi_rank == 0:
        print ("-"*width)
        print(title)
        sys.stdout.flush()

#//////////////////////////////////////////////////////////////////////
def print_section_footer(width=100):
    if mpi_rank == 0:
        print ("-"*width)
        sys.stdout.flush()

#//////////////////////////////////////////////////////////////////////
def tuple2str(t, fmt='%12.10f'):
    return ','.join([fmt]*len(t))%tuple(t)


#//////////////////////////////////////////////////////////////////////
def info_gray(s, check=True):
    if mpi_rank == 0 and check:
        print ("\033[1;37;30m%s\033[0m"%s)


#//////////////////////////////////////////////////////////////////////
# Get the zero leading string for given time step No.
def step_str(i, l=10):
    a = str(i)
    la = l - len(a)
    return '0'*la + a

#//////////////////////////////////////////////////////////////////////
# Type check parser
def get_cmdarg(cmdline, key, default_value = None):
    """Retrieve key from commandline kwargs with light type coercion based on default."""
    if key in cmdline.keys():
        value = cmdline[key]
        if default_value:
            if type(default_value) is int:
                return int(value)
            if type(default_value) is float:
                return float(value)
            if type(default_value) is bool:
                return bool(eval(value))
        return value
    return default_value


#//////////////////////////////////////////////////////////////////////
def beta(err, p):
    if p < 0:
        if err >= 0.1:
            return 0.5
        else:
            return 1.0 - 5*err**2
    else:
        if err >= 0.1:
            return 1.5
        else:
            return 1.0  + 5*err**2


#//////////////////////////////////////////////////////////////////////
def w(P):
    return 1.0 / ( 1.0 + 20.0*abs(P))


#//////////////////////////////////////////////////////////////////////
def get_file_paths(results_folder):
    if mpi_rank == 0:
        counter = 1
        to_check = path.join(results_folder, "data", "%s")
        while path.isdir(to_check % str(counter)):
            counter += 1

        if counter > 1:
            counter -= 1
        # if not path.exists(path.join(to_check % str(counter), "VTK")):
        #     makedirs(path.join(to_check % str(counter), "VTK"))
    else:
        counter = 0

    counter = MPI.max(MPI.comm_world, counter)

    common_path = path.join(results_folder, "data", str(counter), "VTK")
    file_u = [path.join(common_path, "u%d.h5" % i) for i in range(3)]
    file_p = path.join(common_path, "p.h5")
    file_nu = path.join(common_path, "nut.h5")
    file_u_mean = [path.join(common_path, "u%d_mean.h5" % i) for i in range(3)]
    files = {"u": file_u, "p": file_p, "u_mean": file_u_mean, "nut": file_nu}

    return files


#//////////////////////////////////////////////////////////////////////
def read_mesh_info(mesh_info_path, key):
    """
    Parse the casename.info file for <INLETS> or <OUTLETS> blocks.
    Retrive the boundary information from the info file.

    Returns:
        ids        : list[int]      boundary ids
        idfr       : list[float]    inlet mean flows (mL/s) OR outlet ratios (unitless)
        ida        : list[float]    areas (mm^2)
        fcs        : list[str]      waveform tags (for inlets); '_' for outlets
    """
    # Extract inflow rate and outflow split ratios
    # Sample
    # p162
    # <INLETS>
    # 3 ICA_V27:FC_MCA_10 (3.4278309480,13.32740523503,-28.2355071983) (-0.2155297545,0.12005896011,-0.9690886291) 1.7649351905 9.7860492613   A*0.27
    #
    # <OUTLETS>
    # 1  None  (-16.8228963362,-1.42906111694,17.7845745237)  (-0.9847740285,-0.16502447805,-0.0546537689)   0.8568220486   2.3063814691   0.3318956234191912
    # 2  None    (7.9704219814,-9.22385217254,15.0763376349)   (0.7251682706,-0.49373752256,-0.4799523290)   1.3398999124   5.6402011155   0.6681043765808088

    #info = open(path.splitext(mesh_path)[0]+'.info', 'r').read()
    info = open(mesh_info_path, 'r').read()

    # Looking for the given key
    p1 = info.find(key)
    if p1<0:
        return [], [], []
    p1 += len(key)
    p2 = info.find('<', p1)
    if p2<0:
        buf = info[p1:]
    else:
        buf = info[p1:p2-1]
    lines = buf.split('\n')

    ids = []
    idfr = []
    ida = []
    idr = []
    fcs = []
    # Reaing at the key values
    for line in lines:
        ls = line.split()
        if len(ls) > 1:
            # id
            ids.append(int(ls[0]))# eval(ls[ 0]))
            # radius
            idr.append(eval(ls[-3]))
            # area
            ida.append(eval(ls[-2]))
            # flowrate or arearatio
            s = ls[-1].replace('A[','a[').replace('R[','r[')
            s = s.replace('A',ls[-2]).replace('R',ls[-3])
            idfr.append(s)
            # wave form
            fcs.append(ls[1])
    # evaluate all the expressions in the flowrates and outflow ratios
    for i,expr in enumerate(idfr):
        for j,k in enumerate(ids):
             expr = expr.replace( 'r[%d]'%k, str(idr[j])).replace( 'a[%d]'%k, str(ida[j]))
        idfr[i] = eval(expr)

    # sum of area ratio correction:
    if key == '<OUTLETS>':
        idfr[-1] = 1.0 - sum(idfr[:-1])

    # print the summary
    for i,s in enumerate(idfr):
        if mpi_rank == 0 and key == '<INLETS>': print ('Inlet  id:', ids[i], ' flowrate (mL/s):', s)
        if mpi_rank == 0 and key == '<OUTLETS>': print ('Outlet id:', ids[i], ' flowrate ratio:', s)


    return ids, idfr, ida, fcs


"""
#//////////////////////////////////////////////////////////////////////
# check if the period is mentioned in the fc waveform file
def _not_used_get_period_from_fcs(fcs):
    periods = [951.0 for f in fcs]
    for i,f in enumerate(fcs):
        fcs_i_filename = f.split(':')[-1]
        if path.exists( path.join('./data', fcs_i_filename) ):
            fcs_ifname = path.join('./data', fcs_i_filename)
        else:
            fcs_ifname = path.join(path.dirname(path.abspath(__file__)), 'data', fcs_i_filename)
            if not path.exists( fcs_ifname ):
                print ('<!> Cannot find the waveform:', fcs_i_filename)
        for line in open(fcs_ifname,'r').readlines():
            if line.strip()[0] in ['#','!','/']:
                p = line.find('period_ms')
                if p > 0:
                    periods[i] = float(''.join((ch if ch in '0123456789.-e' else ' ') for ch in line[p+9:]).strip().split(' ')[0])
    return periods
"""


# ---------------------------------- Setup Parameters ---------------------------------------------------
#//////////////////////////////////////////////////////////////////////
def problem_parameters(commandline_kwargs, NS_parameters, **NS_namespace):
    """
    Fill NS_parameters with case-specific settings.
    Also prepares HDF5 naming template for BSLSolver IO.
    Note: Parameters are in mm and ms!!
    """
    
    mesh_name       = get_cmdarg(commandline_kwargs, "mesh_name")
    mesh_path       = path.join("./data", mesh_name + ".xml.gz")
    mesh_info_path  = path.join("./data", mesh_name + ".info")

    if mpi_rank == 0: print('Reading mesh information:', mesh_info_path)

    # Check that the mesh files exist
    if mesh_name is None:
        raise RuntimeError("mesh_name is required (expects ./data/<mesh_name>.xml.gz and .info).")

    if mesh_path == None:
        print('<!> Unable to run without a mesh file.')

    # Obtain mesh information
    id_in, Q_means, inlet_area, fcs = read_mesh_info(mesh_info_path, '<INLETS>')
    id_out, area_ratio, outlet_area, _ = read_mesh_info(mesh_info_path, '<OUTLETS>')


    restart_folder = get_cmdarg(commandline_kwargs, 'restart_folder')

    if restart_folder:
        f = open(path.join(restart_folder, 'params.dat'), 'rb')
        NS_parameters.update(pickle.load(f))
        f.close()
        NS_parameters['restart_folder'] = restart_folder
        #Assign case name: pipe_ipcs_ab_cn_VWI_03k_LICA_constant_ts10000_cycles3_uOrder1
        case_name = NS_parameters['case_name']
        case_fullname = NS_parameters['case_fullname']

    else:

        case_name       = get_cmdarg(commandline_kwargs, 'mesh_name')
        period          = get_cmdarg(commandline_kwargs, 'period', 915.0)   # [ms]
        timesteps       = get_cmdarg(commandline_kwargs, 'timesteps', 2000)
        no_of_cycles    = get_cmdarg(commandline_kwargs, 'cycles', 2)
        save_freq       = get_cmdarg(commandline_kwargs, 'save_frequency', 5)
            
        if mpi_rank == 0: print('Found out period [ms] = %s '%str(period))


        # Build a descriptive case_fullname
        #txt = ''
        #for i, id in enumerate(id_in):
        #    txt += '_I%d_%s_Q%d'%(id,fcs[i].replace(':','_'),int(Q_means[i]*100))
        #txt += '_Per%d'%int(period)

        #case_fullname = ("art_" + mesh_name + txt + "_Newt370" + "_ts" + str(timesteps) + "_cy" + str(cycles) + "_uO" + str(uOrder))
        case_fullname = (mesh_name + "_ts" + str(timesteps) + "_cy" + str(no_of_cycles))
        results_folder = f"./results/{case_fullname}_saveFreq{save_freq}"

        #####--------- IMPORTANT: OASIS expects all parameters in [mm] and [ms]! -------------####
        
        # Calculate kinematic viscosity nu (all in SI units) -->
        mu_Pas   = get_cmdarg(commandline_kwargs, 'viscosity_mu_Pas', 0.0037)
        rho_kgm3 = get_cmdarg(commandline_kwargs, 'density_kgm3', 1057)
        nu_m2s   = mu_Pas/rho_kgm3
        nu_mm2ms = nu_m2s*1000 #convert from SI to units that oasis expects

        NS_parameters.update(
            case_name           = case_name,
            case_fullname       = case_fullname,
            results_folder      = results_folder,
            mesh_path           = mesh_path, 
            id_in               = id_in,
            id_out              = id_out,
            area_ratio          = area_ratio,

            no_of_cycles        = no_of_cycles,                                                         # total number of cycles
            period              = period,                                                               # time of a single cycle [ms]
            T                   = period * no_of_cycles,                                                # total simulation time [ms]
            dt                  = period / timesteps,                                                   # timestep size [ms]
            time_steps          = timesteps,
            nu                  = nu_mm2ms, #get_cmdarg(commandline_kwargs, 'viscosity', 0.0035),       # kinematic viscosity [mm^2/ms]
            velocity_degree     = get_cmdarg(commandline_kwargs, 'uOrder', 1),                          # FE degree of velocity

            save_freq           = get_cmdarg(commandline_kwargs, 'save_frequency', 5),                  # save every N steps 
            save_first_cycle    = get_cmdarg(commandline_kwargs, 'save_first_cycle', False),            # flag to save first cycle or not
            save_exact_tsteps   = get_cmdarg(commandline_kwargs, 'save_exact_tsteps', False),           # save exact timesteps
            #save_tsteps_list   = get_cmdarg(commandline_kwargs, 'save_exact_tsteps', False), 
            checkpoint          = get_cmdarg(commandline_kwargs, 'checkpoint', 500),                    # write restart every N steps
            killtime            = get_cmdarg(commandline_kwargs, 'maxwtime', max_wtime_before_kill),
            dump_stats          = 1000,
            compute_flux        = 5,
            save_step           = get_cmdarg(commandline_kwargs, 'save_step', 100000),                  # Mehdi doesn't use the oasis output
            print_intermediate_info = 1000,                                                              # Controls the frequency of printing summary of timings to log file
            #print_WSS          = get_cmdarg(commandline_kwargs, 'print_WSS', True),
            #plot_interval      = 10e10,


            inlet_BC_type               = get_cmdarg(commandline_kwargs, 'inlet_BC_type', 'pulsatile'), # choose from 'ramp', 'pulsatile', 'custom'
            not_zero_pressure_outlets   = not get_cmdarg(commandline_kwargs, 'zero_pressure_outlets', False),
            include_gravity             = get_cmdarg(commandline_kwargs, 'include_gravitational_effects', False),
            flat_profile_at_intlet_bc   = get_cmdarg(commandline_kwargs, 'flat_profile_at_intlet_bc', False),
            
            use_krylov_solvers  = True,
            krylov_solvers      = dict(
                monitor_convergence=False,
                error_on_nonconvergence=False, 
                nonzero_initial_guess=True, 
                maximum_iterations=200,
                relative_tolerance=1e-8,
                absolute_tolerance=1e-8)
            )

    # Write parameters to log
    if mpi_rank == 0:
        # Check if the kinematic viscosity (nu) is within correct range
        nu = NS_parameters['nu']
        if not (0.003 <= nu <= 0.004):
            raise ValueError("Error: Kinematic viscosity (nu) is out of expected range (0.003-0.004 [mm^2/ms])! CHECK THE UNTIS! \n")
        else:
            print(f"\nBlood kinematic viscosity (nu) is within expected range: nu (mm^2/ms)= {nu:.4f} \n")

        # Print all NS_parameters to log
        info_gray(str(NS_parameters))

    
    # Initialize BSLSolver HDF5 pattern
    f = int(np.log10(NS_parameters['T']))+2
    g = len(str(int(NS_parameters['time_steps']*NS_parameters['no_of_cycles'])))+1
    h5stdio.init(NS_parameters['results_folder'], case_fullname+'_curcyc_%%d_t=%%0%d.4f_ts=%%0%dd_up.h5'%(f+4,g))



# --------------------------------------- Mesh ----------------------------------------------------------
#//////////////////////////////////////////////////////////////////////
def mesh(mesh_path, **NS_namespace):
    """Oasis function to create the mesh."""

    print_section_header('Loading mesh file: ' + mesh_path)

    #mesh_folder = mesh_path #path.join(path.dirname(path.abspath(__file__)), mesh_path)

    m =  Mesh(mesh_path)
    m.mpi_comm = mpi_comm

    # Mesh statistics
    num_points    = Function(FunctionSpace(m, "CG", 1)).vector().size()
    vol           = MPI.sum(MPI.comm_world, assemble(Constant(1)*dx(m)))
    cell_dia      = [Cell(m,i).circumradius() for i in range (m.num_cells())]
    avg_cell_dia  = sum(cell_dia) / len(cell_dia)
    num_cells     = int( MPI.sum(MPI.comm_world, m.num_cells()) )
    #num_points   = int( MPI.sum(MPI.comm_world, m.num_vertices()) ) // shared points?
    hmin          = MPI.min(MPI.comm_world, m.hmin()) #[mm]
    hmax          = MPI.max(MPI.comm_world, m.hmax()) #[mm]
    num_facets    = int( MPI.sum(MPI.comm_world, m.num_facets()) )
    
    pss = mesh_path.rfind('/')
    pss = 0 if pss < 0 else pss+1
    pos = mesh_path.rfind('.xml.gz')
    if pos < 0: pos = mesh_path.rfind('.')
    mesh_h5_filename = mesh_path[pss:pos]+'.h5'
    mesh_h5_filepathname = os.path.join( NS_namespace['results_folder'], mesh_h5_filename)

    # Write mesh parameters to log
    if mpi_rank == 0:
        #print ("-"*100)
        print ("Mesh Name:                  ", mesh_path)
        print ("Number of cells:            ", num_cells)
        print ("Number of points:           ", num_points)
        print ("Number of facets:           ", num_facets)
        print ("Mesh Volume:                ", vol)
        print ("Min cell diameter [mm]:     ", hmin)
        print ("Max cell diameter [mm]:     ", hmax)
        print ("Average cell diameter [mm]: ", avg_cell_dia)
        sys.stdout.flush()
        #info(m, False)

    fd = MeshFunction("size_t", m, m.geometry().dim() - 1, m.domains())

    inout_area = {}
    dS = {}
    for id in NS_namespace['id_in']:
        dS[id] = ds(id, domain=m, subdomain_data=fd)
        inout_area[id] = abs( assemble(1.0*dS[id]) )
    for id in NS_namespace['id_out']:
        dS[id] = ds(id, domain=m, subdomain_data=fd)
        inout_area[id] = abs( assemble(1.0*dS[id]) )
    NS_namespace['inout_area'] = inout_area

    normals = FacetNormal(m)

    if mpi_rank == 0:
        print('writing ', mesh_h5_filepathname)
        sys.stdout.flush()

    # Output the Mesh file into HDF5 format
    Hdf = HDF5File(m.mpi_comm(), mesh_h5_filepathname, "w")
    Hdf.write(m, '/Mesh')
    Hdf.close()

    h5stdio.SetMeshInfo(mesh_h5_filepathname, mesh_h5_filename, num_cells, num_points)

    print_section_footer()

    return m, dS, fd, normals, m.geometry().dim(), inout_area

#//////////////////////////////////////////////////////////////////////
# Oasis hook: Overrides the default parameters
def post_import_problem(NS_parameters, mesh, commandline_kwargs, NS_expressions, **NS_namespace):
    """Oasis hook: Called after importing from problem."""

    # Update NS_parameters with all parameters modified through command line
    for key, val in commandline_kwargs.items():
        if isinstance(val, dict):
            NS_parameters[key].update(val)
        else:
            NS_parameters[key] = val

    # If the mesh is a callable function, then create the mesh here.
    if callable(mesh):
        mesh, dS, fd, nors, dim, inout_area= mesh(**NS_parameters)

    assert(isinstance(mesh, Mesh))

    # Returned dictionary to be updated in the NS namespace
    d = dict(mesh=mesh, dS=dS, subdomain_data=fd, normals=nors, dim=dim, inout_area=inout_area)
    d.update(NS_parameters)
    d.update(NS_expressions)
    return d



# --------------------------------------- Boundary Conditions --------------------------------------------

"""
#//////////////////////////////////////////////////////////////////////
# Read Inflow wave form and return the flow rate at all times
def flow_waveform(Qmean, cycles, period, time_steps, FC):
    omega = (2.0 * np.pi / period) #* cycles
    an = []
    bn = []

    #Load the Fourier Coefficients
    infile_FC = open( path.join(path.dirname(path.abspath(__file__)), 'data', FC), 'r').readlines()
    for line in infile_FC:
        abn = line.split()
        an.append(float(abn[0]))
        bn.append(float(abn[1]))

    t_values = np.linspace(0, period*cycles, num=time_steps)
    Q_values = []
    for t in t_values:
        Qn = 0 + 0j
        t1 = t / cycles
        for i in range (len(an)):
            Qn = Qn + (an[i]-bn[i]*1j)*np.exp(1j*i*omega*t1)
        Qn = abs(Qn)
        Q_values.append( Qmean * Qn )
        #print (t, Qn)
    return t_values, Q_values
"""

#//////////////////////////////////////////////////////////////////////
def make_poiseuille_bcs(mesh, ds_inlet, Q_inflow, **NS_namespace):
    """
    Build Poiseuille expressions aligned with the inlet's average normal.
    Q_inflow in mL/s (consistent legacy units).
    """

    dim = mesh.geometry().dim()

    x = SpatialCoordinate(mesh) #[mm]
    area   = assemble(Constant(1.0)*ds_inlet) #[mm^2]
    center = (assemble(x[0]*ds_inlet)/area, assemble(x[1]*ds_inlet)/area, assemble(x[2]*ds_inlet)/area ) #[mm]
    

    # ----------------------- Obtain inlet parameters ---------------------- #
    
    ### 1. Compute the area-weighted average normal ###

    # Obtain raw normals
    # n_raw[i]: i-th component (i = 0,1,2) of the unit outward normal vector on each boundary facet
    n_raw  = FacetNormal(mesh)

    # Average the normals over the inlet
    n_avg  = np.array([assemble(n_raw[i]*ds_inlet) for i in range(dim)])
    
    # Calculate the length of average normal components (~ inlet area) -> used for normalization
    n_len  = np.sqrt(sum([n_avg[i]**2 for i in range(dim)]))

    # Normalize average normals -> normal: unit vector representing the average outward normal of the inlet patch
    normal = n_avg/n_len


    ### 2. Compute other parameters
    n0, n1, n2 = normal[0], normal[1], normal[2]
    c0, c1, c2 = center[0], center[1], center[2] #[mm]
    R = np.sqrt(area/np.pi) #[mm]

    # umax calculated based on input flowrate
    u_max  = 2.0 * Q_inflow / area        #[m/s] == [mm/ms] == [ml/s / mm^2]
    Reynolds = u_max*2*R/NS_parameters["nu"]
    

    dt = NS_parameters['dt']
    max_dt = 0.5*mesh.hmin()/u_max  # based on CFL = 0.5

    if mpi_rank == 0:
        print(f"Starting simulations for dt = {dt} [ms]")
        print(f"Suggested timestep [ms] = {max_dt:0.6f}\n")

        print (f"Inlet properties: \n"
            f"R [mm]        =   {R:.4f} \n"
            f"Area [mm2]    =   {area:.4f} \n"
            f"Q [mL/s]      =   {Q_inflow:.4f} \n"
            f"umax [m/s]    =   {u_max:.4f} \n"
            f"Reynolds      =   {Reynolds:.1f} \n"
            f"centroid [mm] =   [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}] \n"
            f"normal        =   [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}] \n"
            )


    # -------------------- Create Expressions for each direction ------------------- #
    # Obtain inlet poiseuille velocity (one component per axis)
    uin_expressions = [[],[],[]]

    # The ramp equation for inlet flowrate is embedded in the below Kernel: Q_in = 2*t/1000 + 0.01 -> t is in [ms]
    kernel = (
        "-ncomp * (2.0 * (2*(t/1000) + 0.01)/ area) * (1.0 - "
        "( pow((x[0]-c0) - n0 * ((x[0]-c0)*n0 + (x[1]-c1)*n1 + (x[2]-c2)*n2), 2) + "
        "  pow((x[1]-c1) - n1 * ((x[0]-c0)*n0 + (x[1]-c1)*n1 + (x[2]-c2)*n2), 2) + "
        "  pow((x[2]-c2) - n2 * ((x[0]-c0)*n0 + (x[1]-c1)*n1 + (x[2]-c2)*n2), 2) ) "
        " / (R*R) )"
    )

    for j in range(dim):      
        uin_expressions[j] = Expression(kernel, ncomp=normal[j], t=0., area=area,
                                        c0=c0, c1=c1, c2=c2,
                                        n0=n0, n1=n1, n2=n2,
                                        R=R, degree=2)

    return uin_expressions


#//////////////////////////////////////////////////////////////////////
# Create Boundary conditions
def create_bcs(u_, p_, p_1, t, NS_expressions, V, Q, area_ratio, mesh, subdomain_data, 
               dS, normals, results_folder, mesh_path, nu,
               id_in, id_out, velocity_degree, pressure_degree, no_of_cycles,
               T, not_zero_pressure_outlets, flat_profile_at_intlet_bc, **NS_namespace):

    print_section_header('Inspecting boundaries and making boundary conditions:')

    # Mesh function / boundaries
    fd = subdomain_data

    # Extract inflow rate and outflow split ratios
    mesh_info_path = path.join('./data', NS_namespace["mesh_name"]+'.info')
    id_in, Q_means, inlet_area, fcs     = read_mesh_info(mesh_info_path, '<INLETS>')
    id_out, area_ratio, outlet_area, _  = read_mesh_info(mesh_info_path, '<OUTLETS>')


    # 1. Inlet BCs
    # Womersley boundary condition at inlet
    id_in_count = len(id_in)
    if mpi_rank == 0:
        print ('Inlet BC type is:', NS_parameters['inlet_BC_type'])
        print('Inlet', 'BCs' if id_in_count > 1 else 'BC', 'on boundaries:' if id_in_count > 1 else 'on boundary', id_in)
        firststr = '    %8s    %-12s    %10s    %15s    %6s'%('inlet_id','wave_form','period(ms)','flowrate(mL/s)','cells')
        secondstr = 'Inlets & Outlets Information\n'+'  id   %-45s  %-45s   %-12s   %-12s'%('center','normal','radius','area')

  
    inlets = []
    inout_area = {}
    bc_inlet_u = [[],[],[]]

    for i in range(id_in_count):
        fcs_i_filename = fcs[i].split(':')[-1]
        tmp_a, tmp_c, tmp_r, tmp_n = Womersley.compute_boundary_geometry_acrn(mesh, dS[id_in[i]], normals)
        
        # Create the inlet flow based on the flow type given by user
        if NS_parameters['inlet_BC_type'] == 'pulsatile': #if fcs_i_filename[0:3] == 'FC_':
            if mpi_rank == 0:
                print ('- loading inflow wave form:', fcs_i_filename)
            inlet_i = Womersley.make_womersley_bcs_2(NS_namespace["period"], Q_means[i], fcs_i_filename, mesh, nu, tmp_a, tmp_c, tmp_r, tmp_n, velocity_degree, flat_profile_at_intlet_bc)
        

        # Added by Rojin A.
        elif NS_parameters['inlet_BC_type'] == 'ramp':
  
            # Surface integrand over the boundaries
            inlet_tag = id_in[i] #2
            ds_inlet = dS[inlet_tag]
            Q_inflow = 2*t/1000 + 0.01 #[ml/s]

            # for debugging
            #if mpi_rank == 0:
                #print('Inlet tag = ', inlet_tag) 
                #print('ds_inlet  = ', ds_inlet)  # for debugging

            inlet_i = make_poiseuille_bcs(mesh, ds_inlet, Q_inflow)


        else: #THIS DOES NOT CURRENTLY WORK #NS_parameters['inlet_BC_type'] == 'custom'
            if mpi_rank == 0:
                print ('- loading custom inflowrate function:', fcs_ifname)
            inlet_i = CustomFunction.make_custom_function_bcs(NS_namespace["period"], Q_means[i], fcs_i_filename, mesh, nu, tmp_a, tmp_c, tmp_r, tmp_n, velocity_degree, flat_profile_at_intlet_bc)
            
        inlets.append(inlet_i)
        bci = [DirichletBC(V, ilt, fd, id_in[i]) for ilt in inlet_i]
        for j in range(3): bc_inlet_u[j].append(bci[j])

        count = len( bci[0].get_boundary_values() )
        inout_area[id_in[i]] = tmp_a
        
        if mpi_rank == 0:
            # print (dir(dS[id_in[i]]))
            firststr += "\n    %8d    %-12s    %10g    %15.8g    %6d"%(id_in[i], fcs_i_filename, NS_namespace["period"], Q_means[i], count)
            secondstr += "\nI %2d   %-45s  %-45s   %-12.10f   %-12.10f"%( id_in[i], tuple2str(tmp_c), tuple2str(tmp_n), tmp_r, tmp_a)

    NS_expressions["inlet"] = inlets

    # Reset the time in boundary condition expressions
    if NS_parameters['inlet_BC_type'] == 'pulsatile': # Added by Rojin A.
        for inlet in NS_expressions["inlet"]:
            for uc in inlet:
                uc.set_t(t)

    elif NS_parameters['inlet_BC_type'] == 'ramp': # Added by Rojin A.
        for inlet in NS_expressions["inlet"]:
            for uc in inlet:
                uc.t = t


    if mpi_rank == 0: print(firststr)


    # 2. Wall BCs
    # Create No-slip BCs for the velocity at the walls
    wall = Constant(0.0)
    bc_wall = DirichletBC(V, wall, fd, 0) # wall is always with the id zero
    bc_wall_len = len(bc_wall.get_boundary_values())
    
    #if mpi_rank == 0: print( 'Wall BC on ' + str(bc_wall_len) , 'cells')


    # Add wall no-slip BC to each velocity component
    for bc_u in bc_inlet_u:
        bc_u.append(bc_wall)


    # 3. Outlet BCs
    id_out_count = len(id_out)
    bc_p = []
    if mpi_rank == 0:
        print('Outlet', 'BCs' if id_out_count > 1 else 'BC', 'on boundaries:' if id_out_count > 1 else 'on boundary', id_out)
        print("    outlet_id    mass_flow_ratio      cells")
    for i, ind in enumerate(id_out):
        tmp_a, tmp_c, tmp_r, tmp_n = Womersley.compute_boundary_geometry_acrn(mesh, dS[id_out[i]], normals)
        inout_area[ind] = tmp_a
        if mpi_rank == 0:
            secondstr += "\nO %2d   %-45s  %-45s   %-12.10f   %-12.10f"%( ind, tuple2str(tmp_c), tuple2str(tmp_n), tmp_r, tmp_a)
        if not_zero_pressure_outlets:
            if NS_parameters['restart_folder']:
                p_initial = assemble(p_*dS[ind]) / inout_area[ind]
            else:
                p_initial = area_ratio[i]
        else:
            p_initial = 0
        outflow = Expression("p", p=p_initial, degree=pressure_degree)
        bc = DirichletBC(Q, outflow, fd, ind)
        bc_p.append(bc)
        NS_expressions[ind] = outflow
        count  = len(bc.get_boundary_values())
        if mpi_rank == 0: print(' '*8, '%4d    %14.12f    %8d'%(ind, p_initial, count))

    if mpi_rank == 0: print(secondstr)

    print_section_footer(132)

    # Return boundary conditions in dictionary
    return dict(u0=bc_inlet_u[0], u1=bc_inlet_u[1], u2=bc_inlet_u[2], p=bc_p)


# ---------------------------------- Oasis Hooks ---------------------------------------------------

#//////////////////////////////////////////////////////////////////////
def pre_solve_hook(mesh, V, Q, newfolder, results_folder, u_, mesh_path,
                   restart_folder, tstep, velocity_degree, nu,**NS_namespace):

    if restart_folder is None:
        # Get files to store results
        files = get_file_paths(results_folder)
        NS_parameters.update(dict(files=files))
    else:
        files = NS_namespace["files"]


    return dict(hdf5_link=h5stdio,
                files=files, #inout_area=NS_parameters['inout_area'],
                final_time=NS_namespace['T'], current_cycle=0, 
                timesteps=NS_namespace['time_steps'], total_cycles=NS_namespace['no_of_cycles'],
                timestep_cpu_time=0, current_time=time.time(), cpu_time=0)


#//////////////////////////////////////////////////////////////////////
def temporal_hook(u_, p_, p, q_, V, mesh, tstep, compute_flux,
                  dump_stats, newfolder, id_in, files, id_out, inout_area, subdomain_data,
                  normals, save_freq, save_first_cycle, hdf5_link, NS_expressions, current_cycle,
                  total_cycles, area_ratio, t, dS, timestep_cpu_time, current_time, 
                  cpu_time, final_time, timesteps, not_zero_pressure_outlets, **NS_namespace):

    # update the current cycles
    current_cycle = int(tstep / timesteps)
    if mpi_rank == 0:
        # Calculate CFL
        max_u = max(u_[0].vector().get_local().max(), u_[1].vector().get_local().max())
        CFL = NS_parameters['dt']*max_u/mesh.hmin()
        print (f'For cycle= {current_cycle},   tstep= {tstep},  current time (ms)= {t:.4f}')
        print (f"worst CFL number is: {CFL:.4f}") 



    fd = subdomain_data

    # Update boundary condition
    if NS_parameters['inlet_BC_type'] == 'pulsatile': # Added by Rojin A.
        for inlet in NS_expressions["inlet"]:
            for uc in inlet:
                uc.set_t(t)

    elif NS_parameters['inlet_BC_type'] == 'ramp': # Added by Rojin A.
        for inlet in NS_expressions["inlet"]:
            for uc in inlet:
                uc.t = t

    timestep_cpu_time = time.time() - current_time
    current_time = time.time()
    cpu_time += timestep_cpu_time

    # Do not proceed if the time step is less than 3
    if tstep < 3: return

    Q_ideals = {}
    # In-Going Flux & pressure
    flux_in = {}
    Q_ins = {}
    pressure_in = {}
    for id in id_in:
        #inout_area[id] = abs( assemble(1.0*ds(id, domain=mesh, subdomain_data=fd)) )
        pressure_in[id] = -assemble(p_*dS[id]) / inout_area[id]
        flux_in[id] = assemble(dot(u_, normals)*dS[id])
        Q_ins[id] = abs(flux_in[id])
    Q_ins_sum = sum(Q_ins.values())
    if mpi_rank == 0: print(f'Q_ins (mL/s): {Q_ins_sum:.4f}, umax_in (m/s): {2*Q_ins_sum/dS[id]} \n') #print('Q_ins:', Q_ins_sum, Q_ins, '\n')

    # Out-Going Flux
    flux_out = {}
    Q_outs =  {}
    pressure_out = {}
    for id in id_out:
        #inout_area[id] = abs( assemble(1.0*ds(id, domain=mesh, subdomain_data=fd)) )
        pressure_out[id] = assemble(p_*dS[id]) / inout_area[id]
        flux_out[id] = assemble(dot(u_, normals)*dS[id])
        Q_outs[id] = abs(flux_out[id])
    Q_outs_sum = sum(Q_outs.values())

    # Compute flux and update pressure condition
    if not_zero_pressure_outlets:
      for i, out_id in enumerate(id_out):
        Q_ideals[i] = area_ratio[i]*Q_ins_sum
        p_old = NS_expressions[out_id].p

        # Gin and Steinman et al., A Dual-Pressure Boundary Condition
        # for use in Simulations of Bifurcating Conduits
        R_optimal = area_ratio[i]
        R_actual = Q_outs[out_id] / Q_ins_sum

        M_err = abs(R_optimal / R_actual)
        R_err = abs(R_optimal - R_actual)

        if p_old < 0:
            E = 1 + R_err / R_optimal
        else:
            E = -1 * ( 1 + R_err / R_optimal )

        # 1) Linear update to converge first 100 tsteps of the first cycle
        delta = (R_optimal - R_actual) / R_optimal
        if tstep < 100:
            h = 0.1
            if p_old > 1 and delta < 0:
                NS_expressions[out_id].p  = p_old
            else:
                NS_expressions[out_id].p  = p_old * ( 1 - delta*h)

        # 2) Dual pressure BC
        else:
            if p_old > 2 and delta < 0:
                NS_expressions[out_id].p  = p_old
            else:
                NS_expressions[out_id].p  = p_old * beta(R_err,p_old) * M_err ** E
        

    #Print the flow rates, fluxes, pressure
    if mpi_rank == 0:

        if NS_parameters['inlet_BC_type'] == 'pulsatile':
            # #Flux In/Out Flux, Velocity, Pressure
            print ("~"*88)
            print ("Flow Rate or Flux Error is: ", 100.*(abs(sum(flux_in.values()))-abs(sum(flux_out.values())))/abs(sum(flux_in.values())),"%")
            print ("~"*88)
            print ("%3s  %2s          %-15s  %-18s  %-18s  %-18s  %-18s"%('I/O','id','Flux', 'Ideal_Flux', 'Velocity', 'Pressure', 'New Pressure'))
            for id in id_in: ######BUGGG FOUND BY ROJIN A.: why we're printing flux_in two times?? #####
                print ("%-3s  %2d  % 16.15f  % 16.15f  % 16.15f  % 16.15f"%('In', id, flux_in[id], flux_in[id], flux_in[id]/inout_area[id], pressure_in[id]))
            for i, id in enumerate(id_out):
                print ("%-3s  %2d  % 16.15f  % 16.15f  % 16.15f  % 16.15f  % 16.15f"%('Out', id, flux_out[id], area_ratio[i]*Q_ins_sum, flux_out[id]/inout_area[id], pressure_out[id], NS_expressions[id].p))
            print ("~"*88)


        sys.stdout.flush()

        elapsed_wall_time = _global_timer.elapsed()[0] - initial_wall_time


    # ---------------------------------- Saving ----------------------------------------        
    
    # Save velocity and pressure for all cycles
    if save_first_cycle:
        if tstep % save_freq == 0:
            h5stdio.Save( current_cycle, t, tstep, Q_ins, Q_outs, NS_parameters, 'Step-%06d'%tstep, q_) #multiple nodes?
            if mpi_rank == 0:
                h5stdio.SaveXDMF( os.path.join(NS_parameters['results_folder'], NS_parameters['case_fullname']+'.xdmf') )
            

    # Save velocity and pressure (only if it's not the first cycle)
    else:
        if (current_cycle > 0):
            if tstep % save_freq == 0:	
                h5stdio.Save( current_cycle, t, tstep, Q_ins, Q_outs, NS_parameters, 'Step-%06d'%tstep, q_) #multiple nodes?
                if mpi_rank == 0:
                    h5stdio.SaveXDMF( os.path.join(NS_parameters['results_folder'], NS_parameters['case_fullname']+'.xdmf') )


#//////////////////////////////////////////////////////////////////////
def theend_hook(stop, newfolder, results_folder, **NS_namespace):
    if mpi_rank == 0:
        if stop:
            if path.exists(path.join(newfolder,'complete')):
                os.remove(path.join(newfolder,'complete'))
            last_lines = open(path.join(newfolder,'incomplete'),'w')
        else:
            if path.exists(path.join(newfolder,'incomplete')):
                os.remove(path.join(newfolder,'incomplete'))
            last_lines = open(path.join(newfolder,'complete'),'w')

        last_lines.write('\nTry: ' + newfolder.split('/')[-1])
        last_lines.write('\nCheckpoint: ' + newfolder)
        last_lines.close()

    #print ('Process %d/%d Terminated.'%(mpi_rank,mpi_size))

