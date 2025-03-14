for task in beta binloc fluorescence human soluability ss stability yeast; do
    sbatch run_esm.slrm --c /homefs/home/lux70/code/PEER_Benchmark/config/single_task/ESM/${task}_ESM_fix.yaml
done

# sbatch run_esm.slrm --c /homefs/home/lux70/code/PEER_Benchmark/config/single_task/ESM/fold_ESM_fix.yaml

