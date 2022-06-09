#!/bin/bash

outdir=./output/
maxiter=800
penalty_anneal_iters=200
i=0
for j in {2..6}
do
    penalty_weight=$((10 ** $j))
    for method in irm girm
    do
        filename="$outdir$method$i"_"maxiter$maxiter"_"anneal$penalty_anneal_iters"_"weight$penalty_weight"
        out_result=$filename.pkl
        out_model=$filename.pt
        out_print=$filename.out
        echo "sbatch run_script.sbatch $i $method $maxiter $out_result $out_model $penalty_anneal_iters $penalty_weight $out_print"
        sbatch run_script.sbatch $i $method $maxiter $out_result $out_model $penalty_anneal_iters $penalty_weight $out_print
    done
done

method=erm
filename="$outdir$method$i"_"maxiter$maxiter"
out_result=$filename.pkl
out_model=$filename.pt
out_print=$filename.out
echo "sbatch run_script.sbatch $i $method $maxiter $out_result $out_model 0 0 $out_print"
sbatch run_script.sbatch $i $method $maxiter $out_result $out_model 0 0 $out_print
