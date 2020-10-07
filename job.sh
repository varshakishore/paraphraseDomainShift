#!/bin/bash
SWEEP_NAME=bert_finetune_12000_1
JOBSCRIPTS=sweep_scripts/$SWEEP_NAME
mkdir -p $JOBSCRIPTS

QUEUE=kilian,default_gpu

echo_append (){
    echo "$@" >> ${SCRIPT}
}

echo_prepare (){
    SCRIPT=$JOBSCRIPTS/$jobname.sh
    SAVE=save/${SWEEP_NAME}/${jobname}
    mkdir -p $SAVE
    echo "#!/bin/bash" > ${SCRIPT}
    echo_append "#SBATCH -J ${jobname}"
    echo_append "#SBATCH -o ${SAVE}/stdout.%j"
    echo_append "#SBATCH -e ${SAVE}/stderr.%j"
    echo_append "#SBATCH --time=96:00:00"
    echo_append "#SBATCH --partition=${QUEUE}"
    echo_append "#SBATCH --mem=50G"
    echo_append "#SBATCH --cpus-per-task=1"
    echo_append "#SBATCH --gres=gpu:1"
    # echo_append "#SBATCH --exclude=nikola-compute02,nikola-compute04"
}

tasks=('msr' 'quora' 'twitter' 'paws' 'paws_qqp')
max_i=`expr ${#tasks[*]} - 1`

for i in $(seq 0 $max_i); do
    jobname="finetune_${tasks[i]}"
    echo_prepare 
    echo_append stdbuf -eL -oL \
        python finetune-bert.py --save_path="/home/vk352/paraphraseDomainShift/savedModels/${SWEEP_NAME}/" --epochs=25 --task=${tasks[i]} --train_task=${tasks[i]}
    sbatch $SCRIPT
done
