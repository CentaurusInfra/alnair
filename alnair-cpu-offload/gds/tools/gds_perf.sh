#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


usage()
{
  echo "Usage: $0   -d | --devices <num_devices>  [ -c | --cross_sock ] -p | --mnt_prefix <mount path prefix> [ -P | --populate] [ -i | --iterations <max_iters>] [ -o | --output-dir <directory name> ] [-q -quick]"
  exit 2
}

GDSIO=/usr/local/gds/tools/gdsio
CROSS_SOCK=0
STAT_TIME=30
POPULATED=1
TEST_TIME=$(expr $STAT_TIME + 15)
MAX_ITERS=3
RESULTS="./"
logdate=$(date +%Y%m%d_%H%M)

PARSED_ARGUMENTS=$(getopt -a -n alphabet -o i:cPp:d:m:o:q --long iterations,cross_sock,populate,mnt_prefix:,devices:,mounts:,output-dir:,quick -- "$@")
VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
  usage
  exit 1
fi

eval set -- "$PARSED_ARGUMENTS"
while :
do
  case "$1" in
    -i | --iterations)    MAX_ITERS="$2"       ; shift 2  ;;
    -c | --cross_sock)    CROSS_SOCK=1       ; shift   ;;
    -P | --populate)    POPULATED=0       ; shift   ;;
    -p | --mnt_prefix) PATH_PREFIX="$2" ; shift 2 ;;
    -d | --devices)   DEVICES="$2"   ; shift 2 ;;
    -m | --mounts)   MOUNTS="$2"   ; shift 2 ;;
    -o | --output-dir)   RESULTS="$2"   ; shift 2 ;;
    -q | --quick)   QUICK=1   ; shift ;;
    --) shift; break ;;
    *) echo "Unexpected option: $1"
       usage ;;
  esac
done

if [ -z $PATH_PREFIX ]
then
        echo "path prefix missing"
	usage
fi

if [ ! -d $RESULTS ]
then
        echo "invalid output directory specified"
	usage
fi

#Find the maximum number of devices
if [ -z $NVIDIA_VISIBLE_DEVICES ] || [ "x$NVIDIA_VISIBLE_DEVICES" = "xall" ] ; then
	MAX_DEVICES=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
else
	MAX_DEVICES=$NVIDIA_VISIBLE_DEVICES
fi

echo "Max allowed GPUS: $MAX_DEVICES"
#Get number of GPUs in system
if [ -z $DEVICES ] || [ "x$DEVICES" == "xall" ]
then
	DEVICES=$MAX_DEVICES
elif [ $DEVICES -gt $MAX_DEVICES ]
then
	DEVICES=$MAX_DEVICES
	echo "restricting the test to max available GPUS $MAX_DEVICES vs specified $DEVICES"
fi


if [ -z $MOUNTS ]
then
    MOUNTS=1
else
	for j in $(seq 0 $((MOUNTS -1)))
	do
	   mountpoint ${PATH_PREFIX}/${j} >/dev/null 2>&1
	   if [ $? -ne 0  ] ; then
		echo "path ${PATH_PREFIX}/${j} is not a mount point"
	   fi
	done
fi

declare -a MOUNT_LIST=()
declare -a DEV_LIST=()
declare -a NUMA_LIST=()
if [ -z $QUICK ] ; then
declare -a XFER_LIST=("0" "2" "1" "NOREG" "NVLINK" "COMPAT")
declare -a IO_SIZE_KiB_LIST=("4" "8" "16" "32" "64" "128" "256" "512" "1024" "2048" "4096" "8192" "16384")
#declare -a THREAD_LIST=("128" "128" "96" "96" "64" "64" "48" "48" "48" "48" "48" "48" "48")
declare -a THREAD_LIST=("128" "128" "96" "96" "96" "96" "96" "96" "96" "96" "96" "96" "96")
MAX_THREADS=128
MAX_FILESIZE=4096
else
echo "running quick test"
MAX_ITERS=1
MAX_THREADS=32
MAX_FILESIZE=2048
declare -a XFER_LIST=("0" "2")
declare -a IO_SIZE_KiB_LIST=("64" "128" "512" "1024" "4096")
declare -a THREAD_LIST=("32" "32" "24" "24" "24")
fi

if [ ! -z $CUDA_VISIBLE_DEVICES ] ;then
	IFS=',' read -r -a CUDA_DEV_LIST <<< "$CUDA_VISIBLE_DEVICES"
	CUDA_DEVICES=${#CUDA_DEV_LIST[@]}

	if [ $CUDA_DEVICES -lt $MAX_DEVICES ] ;then
		echo "restricting the test to CUDA_VISIBLE GPUS: $CUDA_DEVICES vs max specified: $MAX_DEVICES"
		MAX_DEVICES=$CUDA_DEVICES
		DEVICES=$MAX_DEVICES
	fi
fi


MPSTAT=`which mpstat`
if [ $? -ne 0 ]
then
	echo "mpstat not found"
	exit 1
fi

NUMA_NODES=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`


for j in $(seq 0 $((MOUNTS -1)))
do
	if [ $DEVICES -le $(expr $MAX_DEVICES / 2 ) ]
	then
		dev=1
	else
		dev=0
	fi

	for i in $(seq 0 $((DEVICES -1)))
	do
		mountpoint ${PATH_PREFIX}/${j} >/dev/null 2>&1
		if [ $? -eq 0  ] ; then
			MOUNT_LIST+=("${PATH_PREFIX}/${j}/GPU${i}")
		else
			if [ "x$MOUNTS" = "x1" ] ; then
				MOUNT_LIST+=("${PATH_PREFIX}/GPU${i}")
			else
				MOUNT_LIST+=("${PATH_PREFIX}/${j}/GPU${i}")
			fi
			echo "mount ${PATH_PREFIX}/${j} not found, creating directory ${MOUNT_LIST[-1]}"
		fi
		mkdir -p ${MOUNT_LIST[-1]}/gds
		if [ ! -d  ${MOUNT_LIST[-1]}/gds ] ; then
			echo "failed to create dir ${MOUNT_LIST[-1]}/gds"
			exit 1
		fi
		if [ "x$CROSS_SOCK" == "x1" ] ;then
			DEV_LIST=("$dev" "${DEV_LIST[@]}")
		else
			DEV_LIST+=("$dev")
		fi
		if [ $? -ne 0 ]; then
		  echo "Failed to create directory ${MOUNT_LIST[$i]}/gds "
		  exit 1
		fi

		nvidia-smi topo -mp | grep 'NUMA Affinity' 2>&1 >/dev/null
		if [ $? -eq 0 ]; then
			if [ $NUMA_NODES -ge 2 ]; then
				if [ -z $CUDA_DEV_LIST ] ;then
					numa_node=`nvidia-smi topo -mp | grep -v NUMA | egrep "^GPU${dev}"$'\t' | awk '{print $(NF)}'`
				else 
					cuda_dev=${CUDA_DEV_LIST[$dev]}
					numa_node=`nvidia-smi topo -mp | grep -v NUMA | egrep "^GPU${cuda_dev}"$'\t' | awk '{print $(NF)}'`
				fi
				if [ ! -z $numa_node ]; then
					NUMA_LIST+=("-n $numa_node")
				else
					NUMA_LIST+=("-n 0")
				fi
			fi
		else
				NUMA_LIST+=(" ")
		fi

		if [ $DEVICES -le $(expr $MAX_DEVICES / 2 ) ]
		then
			#skip itermeditate GPUs
			dev=$(expr $dev + 2)
		else
			dev=$(expr $dev + 1)
		fi

	   done   # end of DEVICES LOOP
done # end of MOUNTS LOOP


for i in $(seq 0 $((DEVICES -1)))
do
    if [ -z $CUDA_DEV_LIST ] ;then
	    echo "mount path: " ${MOUNT_LIST[$i]} " -> GPU device: "${DEV_LIST[$i]}
    else
	    echo "mount path: " ${MOUNT_LIST[$i]} " -> GPU device: "${CUDA_DEV_LIST[$i]}
    fi
done

rm -rf gdsio.txt
rm -rf results.txt
rm -rf gdsio_populated.txt
touch results.txt

for IOTYPE in 0 1
do
for XFERTYPE in ${XFER_LIST[@]}
do
	echo "Iteration XferType Iosize IoType XferMode TotalThreads TransferSize/RequestedSize(KiB) Throughput(GiB/sec) latency(usec) total_ops TOTALTIME(sec) CPU_USR(%) CPU_SYS(%) CPU_IRQ(%)" >> results.txt
        k=0
	for IOSIZE in ${IO_SIZE_KiB_LIST[@]}
	do
		for iter in $(seq 1 ${MAX_ITERS})
		do
			if [ "$XFERTYPE" = "NOREG" ] ; then
				XFER="-x 0 -b"
			elif [ "$XFERTYPE" = "COMPAT" ] ; then
				XFER="-x 0"
			elif [ "$XFERTYPE" = "NVLINK" ] ; then
				XFER="-x 0 -p"
				if [ "$CROSS_SOCK" != 1 ]; then
					"skipping NVLINK for affinity test"
					continue
				fi
			elif [[ "$CROSS_SOCK" == 1 && "$XFERTYPE" = "0" ]] ; then
				XFER="-x 0"
				XFERTYPE="CROSS_SOCK"
			else
				XFER="-x $XFERTYPE"
			fi

                        THREADS=${THREAD_LIST[$k]}
			if [ "$IOSIZE" -lt "16" ] ; then
				FILESIZE=$(expr $MAX_FILESIZE / 8)
			elif [ "$IOSIZE" -lt "64" ] ; then
				FILESIZE=$(expr $MAX_FILESIZE / 8)
			elif [ "$IOSIZE" -lt "128" ] ; then
				FILESIZE=$(expr $MAX_FILESIZE / 4)
			elif [ "$IOSIZE" -lt "256" ] ; then
				FILESIZE=$(expr $MAX_FILESIZE / 4)
			elif [ "$IOSIZE" -lt "512" ] ; then
				FILESIZE=$(expr $MAX_FILESIZE / 2)
			elif [ "$IOSIZE" -lt "1024" ] ; then
				FILESIZE=${MAX_FILESIZE}
			else
				FILESIZE=${MAX_FILESIZE}
			fi

			if [ "$XFERTYPE" = "COMPAT" ] ; then
				pkill gdsio

				lsmod | grep nvidia_fs
				if [ $? -eq 0 ]; then
					sudo rmmod nvidia_fs
					if [ $? -ne 0 ]; then
						echo " Failed to remove nvidia_fs driver for compat mode"
						exit 2
					fi
				fi
                        elif [ "$XFERTYPE" != "1" ] && [ "$XFERTYPE" != "2" ]; then
				sudo modprobe nvidia_fs
				if [ $? -ne 0 ]; then
					echo " Failed to modprobe nvidia_fs driver"
					exit 2
				fi
			fi

			JOBS=""
			for i in ${!MOUNT_LIST[@]}
			do
				JOBS+=" -D ${MOUNT_LIST[$i]}/gds -w $THREADS -d ${DEV_LIST[$i]} ${NUMA_LIST[$i]}"
			done

			if [ "x$POPULATED" == "x0" ] ; then
				echo "populating files:"

				PJOBS=""
				for i in ${!MOUNT_LIST[@]}
				do
					PJOBS+=" -D ${MOUNT_LIST[$i]}/gds -w $MAX_THREADS -d ${DEV_LIST[$i]} ${NUMA_LIST[$i]}"
				done

			        echo ${GDSIO} -s ${MAX_FILESIZE}M -V -I 1 -x 0 \
				$PJOBS -i 1M

			        ${GDSIO} -s ${MAX_FILESIZE}M -V -I 1 -x 0 \
				$PJOBS -i 1M > /results/gdsio_populated_${logdate}.txt
				if [ $? -ne 0 ] ; then
					echo "${GDSIO} population failed"
					exit 1
				else
					echo "Done populating"
					rm /results/gdsio_populated_${logdate}.txt
					POPULATED=1
				fi
			fi

			pkill gdsio
			if [ $? -eq 0 ]; then
				echo "WARNING: background gdsio process running. killed"
			fi
			echo "Running iter $iter for IOTYPE: $IOTYPE for XFERTYPE: $XFER IOSIZE: $IOSIZE kb with threads: $THREADS"
		        echo ${GDSIO} -T ${TEST_TIME} -s ${FILESIZE}M -I $IOTYPE $XFER\
				$JOBS \
				-i ${IOSIZE}k $OPTS

		        ${GDSIO} -T ${TEST_TIME} -s ${FILESIZE}M -I $IOTYPE $XFER\
				$JOBS \
				-i ${IOSIZE}k $OPTS > gdsio.txt &
			PID=$!
			sleep 5
			MPSTATOUT=`${MPSTAT} -P ALL 1 ${STAT_TIME} | grep Average | grep 'all' | awk '{print $3, $5, $7}'`
			if wait $PID; then #checks if process executed successfully or not
				echo -n "$iter " >> results.txt
				echo -n "$XFERTYPE " >> results.txt
				echo -n "${IOSIZE}KiB " >> results.txt
				GDSSTAT=`cat gdsio.txt | awk '{print $2, $4, $6, $8, $12, $15, $18, $20, $22}'`
				echo -n "$GDSSTAT ">> results.txt
				echo $MPSTATOUT >>results.txt
			else    #process terminated abnormally
				echo "failed (returned $?)"
				mv gdsio.txt /results/gdsio_failed_${logdate}.txt
			fi
                        rm gdsio.txt
                        
		done
		k=$(expr $k + 1)
	done
done

if [ "$CROSS_SOCK" != 1 ]; then
	mv results.txt ${RESULTS}/results_affinity_${IOTYPE}_${logdate}.csv
else
	mv results.txt ${RESULTS}/results_cross_sock_${IOTYPE}_${logdate}.csv
fi

touch results.txt
done
rm results.txt
