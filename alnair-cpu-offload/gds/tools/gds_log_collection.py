#!/usr/bin/env python3

import os, sys, argparse, subprocess, re, tempfile, time, socket, datetime, shutil, io, tarfile, platform, glob, getopt, errno
from shutil import copyfile
GDS_TOOLS_PATH = "/usr/local/cuda/gds/tools/"
CUFILE_LOG_FILE_DEFAULT_NAME = "cufile.log"

#append data to filename
def add_to_file(temp_dir, filename, data):
    file_path = temp_dir + "/" + filename
    file_instance = open(file_path,'a')
    file_instance.write(data)
    file_instance.close()

#Add the contents of the file to tarball   
def add_to_tarfile(output_filename, file_to_add):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(file_to_add, arcname=os.path.basename(file_to_add))
    tar.close()

#Run the command in shell and save the output under temp_dir/filename
def run_cmd_and_add_output_to_file(cmd, temp_dir, filename):
    output = subprocess.run([cmd], stdout=subprocess.PIPE, universal_newlines=True, shell=True)
    add_to_file(temp_dir, filename, cmd + "\n")
    add_to_file(temp_dir, filename, "=========================================" + "\n")
    add_to_file(temp_dir, filename, output.stdout + "\n")
    add_to_file(temp_dir, filename, "\n")

def help():
    print("This tool is used to collect logs from the system that are relevant for debugging.")
    print("It collects logs such as os and kernel info, nvidia-fs stats, dmesg logs, syslogs,")
    print("System map files and per process logs like cufile.json, cufile.log, gdsstats, process stack, etc.")
    print("A compressed tarfile is generated at the end of this process.")
    print("Usage ./gds_log_collection.py [options]")
    print("options:")
    print("     -h  help")
    print("     -d  destination directory for log collection")
    print("     -f  file1,file2,..(Note: there should be no spaces between ',')")
    print("         These files could be any relevant files apart from the one's being collected(e.g. crash files)")
    print("Usage examples:")
    print("sudo ./gds_log_colection.py - Collects all the relevant logs")
    print("sudo ./gds_log_colection.py -f file1,file2 - Collects all the relevant files as well as user specifed files.")
# Defining main function
def main(argv):
    crash_files_list = []
    if os.geteuid() != 0:
        sys.exit("You need to have root privileges to run this script.\nPlease try again, using 'sudo'. Exiting...")

    # where tar file will be created
    tar_file_location = "/tmp/"
    if argv:
        try:
            opts, args = getopt.getopt(argv,"hf:d:")
        except getopt.GetoptError:
            help()
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                help()
                sys.exit()
            elif opt == "-f":
                crash_files_string = arg
                crash_files_list = [x.strip() for x in crash_files_string.split(',')]
            elif opt == "-d":
                tar_file_location = arg + "/"

    #mkdir if necessary
    try:
        os.makedirs(tar_file_location, exist_ok = True)
    except OSError as e:
        print("invalid tar file location errror:" + str(e))
        sys.exit(1)

    #temp dir
    log_file_prefix = 'gds-logs-{hostname}-{time}'.format(hostname=socket.gethostname(),
                        time=datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    temp_dir = tempfile.TemporaryDirectory(prefix=log_file_prefix + "-")
    
    #Get OS and kernel info
    cmd = "uname -a"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "os_info")
    cmd = "cat /etc/os-release"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "os_info")
    
    #Get proc cmd line
    cmd = "cat /proc/cmdline"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "proc_cmdline")

    #Get Module info
    cmd = "modinfo nvidia_fs"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "modinfo_output")
    cmd = "modinfo nvme"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "modinfo_output")
    cmd = "modinfo nvme_rdma" 
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "modinfo_output")
    cmd = "modinfo rpcrdma"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "modinfo_output")
    cmd = "modinfo wekafsio"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "modinfo_output")
    cmd = "modinfo lustre"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "modinfo_output")
    cmd = "modinfo lnet"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "modinfo_output")

    #Get dmesg logs
    cmd = "dmesg"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "dmesg_output")

    #Get gdschek.py O/P
    cmd = GDS_TOOLS_PATH + "/gdscheck.py -pvV"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "gdscheck_py_output")

    #Get nvidia_fs stats
    cmd = "cat /proc/driver/nvidia-fs/stats"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "nvidia_fs_stats")
    
    cmd = "cat /proc/driver/nvidia-fs/peer_affinity"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "gdscheck_nvidia_fs_peer_affinity")
    
    cmd = "cat /proc/driver/nvidia-fs/peer_distance"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "nvidia_fs_peer_distance")

    #Get nvidia-smi O/P
    cmd = "nvidia-smi"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "nvidia_smi_output")
    
    cmd = "nvidia-smi topo -m"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "nvidia_smi_output")
    
    #Get lspci O/P

    cmd = "lspci -vvv"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "lspci_output")

    #Get cpu and meminfo
    cmd = "cat /proc/cpuinfo"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "proc_cpuinfo_output")
    
    cmd = "cat /proc/meminfo"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "proc_meminfo_output")

    #Get ofed info
    cmd = "ofed_info -s"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "ib_stats_output")
    
    cmd = "ibstatus"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "ib_stats_output")

    cmd = "ibdev2netdev"
    run_cmd_and_add_output_to_file(cmd, temp_dir.name, "ib_stats_output")

    #Get syslog and kernel messages
    kernel_log_files_dir=temp_dir.name + "/kernel_log_files"
    os.mkdir(kernel_log_files_dir)
    path = "/var/log/syslog*"
    for filename in glob.glob(path):
        if os.path.isfile(filename):
            shutil.copy(filename, kernel_log_files_dir)

    path = "/var/log/messages*"
    for filename in glob.glob(path):
        if os.path.isfile(filename):
            shutil.copy(filename, kernel_log_files_dir)    
    
    #Copy crash files
    crash_files_dir=temp_dir.name + "/crash_files"
    os.mkdir(crash_files_dir)
    for crash_file in crash_files_list:
        if os.path.isfile(crash_file):    
            shutil.copy(crash_file, crash_files_dir)
        else:
            print("Please enter a valid file path.", crash_file, " is not valid")
            sys.exit(2)

    #Get the system map file and vmlinuz
    cmd = "uname -r"
    output = subprocess.run([cmd], stdout=subprocess.PIPE, shell=True)
    uname_r_output = output.stdout
    uname_r_output_strip = uname_r_output.decode().strip()
    system_map_file = ("/boot/" + "System.map-" + uname_r_output_strip)
    system_map_file.strip()
    if os.path.isfile(system_map_file):
        shutil.copy(system_map_file, temp_dir.name)
    else:
        print("System map file does not exist under", system_map_file)
    
    vmlinuz_file = ("/boot/" + "vmlinuz-" + uname_r_output_strip)
    if os.path.isfile(vmlinuz_file):
        shutil.copy(vmlinuz_file, temp_dir.name)
    else:
        print("vmlinuz not found under", vmlinuz_file)

    #Get the cufile.log file in the current directory, if there is any
    if os.path.isfile("./" + CUFILE_LOG_FILE_DEFAULT_NAME):
            shutil.copy("./" + CUFILE_LOG_FILE_DEFAULT_NAME, temp_dir.name)

    #Get memory peers info for nvidia-fs
    memory_peers_dir = "/sys/kernel/mm/memory_peers/nvidia-fs/*"
    for filename in glob.glob(memory_peers_dir):
        if os.path.isfile(filename):
            cmd = "cat %s"%(filename)
            run_cmd_and_add_output_to_file(cmd, temp_dir.name, "memory_peers_nvidia_fs")

    #Get process IDs using nvidia drivers
    process_ids = subprocess.run(["nvidia-smi --query-compute-apps=pid --format=csv,noheader"], stdout=subprocess.PIPE, universal_newlines=True, shell=True)
    process_ids = process_ids.stdout
    process_ids = process_ids.split("\n")
    process_ids = list(filter(None, process_ids))

    for pid in process_ids:
        pid_dir = temp_dir.name + "/pid_" + pid
        json_path = "/etc/cufile.json" 
        os.mkdir(pid_dir)
        json_paths = []
        cufile_log_path = []
        json_paths.append("/etc/cufile.json")
        cmd = "xargs -0 -L1 -a /proc/{}/environ ".format(pid)
        output = subprocess.run([cmd], stdout=subprocess.PIPE, universal_newlines=True, shell=True)
        content = output.stdout
        content = content.split("\n")
        content = list(filter(None, content))
        #Search for cufile.json 
        for c in content:
            line = re.search(r"CUFILE_ENV_PATH_JSON*", c)
            
            if line :
                x = line.string.split("=")
                json_path = x[1]
        #Copy json under pid_<pid number>
        shutil.copy(json_path, pid_dir)
        
        #Search for cufile.log
        for c in content:
            line = re.search(r"CUFILE_LOGFILE_PATH*", c)
            if line :
                x = line.string.split("=")
                cufile_log_path.append(x[1])
        
        if not cufile_log_path :
            pattern1 = "//\"dir\":"
            pattern2 = "\"dir\":"
            file = open(json_path, "r")
            for line in file:
                if re.search(pattern1,line):
                    break
                elif re.search(pattern2,line):
                    x = line.split(":")
                    p = x[1].strip()
                    p = p.rstrip(",")
                    p = re.sub('\"', '', p)
                    search_path = p + "/cufile_"+ pid + "*"
                    for filename in glob.glob(search_path):
                        cufile_log_path.append(filename)

        if cufile_log_path: 
            for filename in cufile_log_path:
                if os.path.isfile(filename) :
                    #copy cufile under pid_<pid number>
                    shutil.copy(filename, pid_dir)

        # Get gds stats
        cmd = GDS_TOOLS_PATH + "/gds_stats -p {} -l 3 ".format(pid)
        run_cmd_and_add_output_to_file(cmd, pid_dir, "gds_stats_output")

        #Get the stack pointer
        cmd = "cat /proc/{}/task/*/stack".format(pid)
        run_cmd_and_add_output_to_file(cmd, pid_dir, "stack_pointer_output")

        #Get the command arguments
        cmd = "ps -p {}  -o args".format(pid)
        run_cmd_and_add_output_to_file(cmd, pid_dir, "process_cmd_args")

    tar_filename = tar_file_location + log_file_prefix + ".tar.xz"
    add_to_tarfile(tar_filename, temp_dir.name)
    print("GDS Log file saved under ", tar_filename)

# Using the special variable 
# __name__
if __name__=="__main__":
    main(sys.argv[1:])
