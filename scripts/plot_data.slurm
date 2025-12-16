#!/bin/bash
#SBATCH --job-name="Plot data"
#SBATCH --export=ALL
#SBATCH --ntasks=1

module load apps/python3

source activate coherence

## declare array of data directories. This is to allow for parallel processing of multiple datasets.
## Example: declare -a arr=("/path/to/data1" "/path/to/data2" "/path/to/data3")
# declare -a arr=("/u/st/by/aissah/scratch/data/BradysHotspring/03_13_2016" "/u/st/by/aissah/scratch/data/BradysHotspring/03_14_2016")
data_dir="/u/st/by/aissah/scratch/data/BradysHotspring"
## declare array of methods to be used.
declare -a arr=("qr")

for method in "${arr[@]}";
do
   echo "Working on directory: $data_dir"
   srun --exclusive --ntasks=1 python ../coherence_analysis/plot_raw_data.py "$data_dir" "('03/14/2016 08:38:18', '03/14/2016 08:40:18')" -ch "[5000, 6500]" -ds 3
done
wait

echo "All jobs finished"
