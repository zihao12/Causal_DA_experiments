#!/bin/bash

outdir=./output/
maxiter=200
for i in {13..16}
do
    for method in erm irm girm
    do
        out_result=$outdir$method$i.pkl
        out_model=$outdir$method$i.pt
        out_print=$outdir$method$i.out
        echo "sbatch run_script.sbatch $i $method $maxiter $out_result $out_model $out_print"
        sbatch run_script.sbatch $i $method $maxiter $out_result $out_model $out_print
    done
done

