#!/bin/bash
#int-or-string.sh


# updating repo
cd ../from_git
git fetch --all
git pull
echo "Fetched latest files from git"
cd ../pr_paramSearch

echo $PWD

PARAMS_FILE="pr_paramSearch_params.csv"
LOG_DIR="/home/sidrees/scratch/RetinaPredictors/grid_scripts/pr_paramSearch/logs"

if [ ! -d "$LOG_DIR" ]; then 
	mkdir $LOG_DIR
fi

expDate_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f1) )
path_mdl_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f2) )
trainingDataset_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f3) )
testingDataset_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f4) )
mdl_name_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f5) )
path_excel_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f6) )
path_perFiles_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f7) )
lightLevel_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f8) )
pr_type_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f9) )
samps_shift_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f10) )
r_sigma_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f11) )
r_phi_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f12) )
r_eta_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f13) )
r_k_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f14) )
r_h_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f15) )
r_beta_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f16) )
r_hillcoef_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f17) )
r_gamma_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f18) )


numParams=${#r_sigma_all[@]}
echo "Number of parameter combinations: $numParams"



for ((i=0; i<$numParams; i++));
#for ((i=0; i<1; i++));
do
 path_mdl=${path_mdl_all[i]}
 trainingDataset=${trainingDataset_all[i]}
 testingDataset=${testingDataset_all[i]}
 mdl_name=${mdl_name_all[i]}
 expDate=${expDate_all[i]}
 path_excel=${path_excel_all[i]}
 path_perFiles=${path_perFiles_all[i]}
 lightLevel=${lightLevel_all[i]}
 pr_type=${pr_type_all[i]}
 samps_shift=${samps_shift_all[i]}
 r_sigma=${r_sigma_all[i]}
 r_phi=${r_phi_all[i]}
 r_eta=${r_eta_all[i]}
 r_k=${r_k_all[i]}
 r_h=${r_h_all[i]}
 r_beta=${r_beta_all[i]}
 r_hillcoef=${r_hillcoef_all[i]}
 r_gamma=${r_gamma_all[i]}
 
 
 

 echo "path_mdl: $path_mdl"
 echo "trainingDataset: $trainingDataset"
 echo "testingDataset: $trainingDataset"
 echo "expDate: $expDate"
 echo "mdl_name: $mdl_name"
 echo "path_excel: $path_excel"
 echo "path_perFiles: $path_perFiles"
 echo "lightLevel: $lightLevel"
 echo "pr_type: $pr_type"
 echo "samps_shift: $samps_shift"
 echo "r_sigma: $r_sigma"
 echo "r_phi: $r_phi"
 echo "r_eta: $r_eta"
 echo "r_k: $r_k"
 echo "r_h: $r_h"
 echo "r_beta: $r_beta"
 echo "r_hillcoef: $r_hillcoef"
 echo "r_gamma: $r_gamma"


 JOB_ID=$(sbatch --export=LOG_DIR=$LOG_DIR,path_mdl=$path_mdl,trainingDataset=$trainingDataset,testingDataset=$testingDataset,expDate=$expDate,mdl_name=$mdl_name,path_excel=$path_excel,path_perFiles=$path_perFiles,lightLevel=$lightLevel,pr_type=$pr_type,samps_shift=$samps_shift,r_sigma=$r_sigma,r_phi=$r_phi,r_eta=$r_eta,r_k=$r_k,r_h=$r_h,r_beta=$r_beta,r_hillcoef=$r_hillcoef,r_gamma=$r_gamma pr_paramSearch_launcher.sh)

echo $JOB_ID
JOB_ID=$(echo "$JOB_ID" | grep -Eo '[0-9]{1,8}')
 
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t' $JOB_ID $expDate $path_mdl $trainingDataset $testingDataset $mdl_name $path_excel $path_perFiles $lightLevel $pr_type $samps_shift,$r_sigma $r_phi $r_eta $r_k $r_h $r_beta $r_hillcoef $r_gamma | paste -sd '\t' >> job_list.csv
 
 
done
