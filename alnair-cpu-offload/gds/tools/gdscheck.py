#!/usr/bin/env python3
import os
import subprocess
import os.path
from os import path
import argparse

def ofed_check():
    print("ofed_info:")
    if path.exists("/usr/bin/ofed_info"):
        try:
                proc = subprocess.Popen(['ofed_info', '-s'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
                if proc:
                    ofed_info_str = proc.stdout.read()
                    if ofed_info_str:
                        ofed_info_str = ofed_info_str.decode().strip(' \n\t')
                    supported = False
                    ofed_info_fields = ofed_info_str.split("-")
                    if (len(ofed_info_fields) == 3 and ofed_info_fields[0] == "MLNX_OFED_LINUX" ):
                            ofed_version_fields = ofed_info_fields[1].split(".")
                            if (len(ofed_version_fields) == 2 and int(ofed_version_fields[0]) >= 5):
                                if (int(ofed_version_fields[1]) >= 1):
                                        supported = True
                            elif (len(ofed_version_fields) == 2 and int(ofed_version_fields[0]) >= 4):
                                if (int(ofed_version_fields[1]) >= 6):
                                        supported = True
                                        print("Note: WekaFS support needs MLNX_OFED_LINUX installed with --upstream-libs")
                    if(supported):
                         print("current version: " + ofed_info_str + " (Supported)")
                    else:
                         print("current version: " + ofed_info_str + " (Unsupported)")
                else:
                     print("current version: Unknown")
        except ValueError:
            print("failed to obtain OFED version information")
    print("min version supported: " + "MLNX_OFED_LINUX-4.6-1.0.1.1")

def weka_check():
    if path.exists("/usr/bin/weka"):
        print("WekaFS:")
        try:
                proc = subprocess.Popen(['weka', 'version', 'current'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
                if proc:
                    weka_ver_str = proc.stdout.read()
                    if weka_ver_str:
                        weka_ver_str = weka_ver_str.decode().strip(' \n\t')
                    supported = False
                    weka_version_fields = weka_ver_str.split(".")
                    if (int(weka_version_fields[0]) >= 3):
                        if (int(weka_version_fields[1]) >= 8):
                            if (int(weka_version_fields[2]) >= 0):
                                supported = True
                                print("GDS RDMA read: supported")
                                print("GDS RDMA write: experimental")
                    if(supported):
                         print("current version: " + weka_ver_str + " (Supported)")
                    else:
                         print("current version: " + weka_ver_str + " (Unsupported)")
                else:
                     print("current version: Unknown")
        except ValueError:
             print("failed to obtain weka version information")
        print("min version supported: " + "3.8.0")

def lustre_check():
     if path.exists("/usr/sbin/lctl"):
        print("Lustre:")
        try:
            proc = subprocess.Popen(['lctl', 'get_param', 'version'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
            if proc:
                supported = False
                lustre_out = proc.stdout.read()
                if lustre_out:
                   lustre_out = lustre_out.decode().strip(' \n\t')
                lustre_ver_str = lustre_out.split('=')
                if(len(lustre_ver_str) == 2 and lustre_ver_str[1].find("ddn")):
                    lustre_version_fields = lustre_ver_str[1].split(".")
                    if (len(lustre_version_fields) ==3 and int(lustre_version_fields[0]) >= 2):
                        if (int(lustre_version_fields[1]) >= 12):
                            patch_ver = lustre_version_fields[2].split('_')
                            if (patch_ver and int(patch_ver[0]) >= 3):
                               supported = True

                    if(supported):
                        print("current version: " + lustre_ver_str[1] + " (Supported)")
                    else:
                        print("current version: " + lustre_ver_str[1] + " (Unsupported)")
                else:
                        print("current version: " + "Unknown")
            else:
                 print("current version: " + "Unknown")
        except ValueError:
            print("failed to obtain lustre version information")
        print("min version supported: " + "2.12.3_ddn28")

def main():
    parser = argparse.ArgumentParser(description='GPUDirectStorage platform checker')
    parser.add_argument('-p', action='store_true', dest='platform',  help='gds platform check')
    parser.add_argument('-f', dest='file', help='gds file check')
    parser.add_argument('-v', action='store_true', dest='versions', help='gds version checks')
    parser.add_argument('-V', action='store_true', dest='fs_versions', help='gds fs checks')

    args = parser.parse_args()

    # Get the gds tools install path, gdscheck would be installed in the same
    # directory where gdscheck.py resides.
    gds_tools_path = (os.path.dirname(os.path.realpath(__file__)))
    gdscheck_path = gds_tools_path+"/gdscheck"
    gds_check=False
    
    cmd = [gdscheck_path]
    if (args.platform) :
        gds_check = True
        cmd.append('-p')
    if (args.file) :
        gds_check = True 
        cmd.append('-f')
        cmd.append(args.file)
    if (args.versions) :
        gds_check = True 
        cmd.append('-v')

    if gds_check and path.exists(gdscheck_path):
        proc = subprocess.Popen(cmd,
                   stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT)
        if proc:
            print(proc.stdout.read().decode())
    elif not args.fs_versions:
        parser.print_help()

    if args.fs_versions:
        print("FILESYSTEM VERSION CHECK:")
        lustre_check()
        weka_check()
        ofed_check()

if __name__== "__main__":
   main()
