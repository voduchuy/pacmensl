#!/bin/bash

cd build/

np_array=(32 16 8 4 2)
model_array=(repressilator)
partition_opts=(Block Graph)
repart_opts=(Repart)


for num_procs in "${np_array[@]}"
do
  for model in "${model_array[@]}"
  do
    for partitioning in "${partition_opts[@]}"
    do
      for repartition in "${repart_opts[@]}"
      do
        mpirun -np $num_procs --bind-to core --map-by socket $model -fsp_verbosity 2 -fsp_partitioning_type $partitioning -fsp_repart_approach $repartition -fsp_output_marginal 1 -fsp_log_events 1 -log_view
      wait
    done
    done
  done
done

for model in "${model_array[@]}"
do
      mpirun -np 1 --bind-to core --map-by socket $model -fsp_verbosity 0 -fsp_partitioning_type Block -fsp_repart_approach Repartition -fsp_output_marginal 1 -fsp_log_events 1 -log_view
      wait
done
