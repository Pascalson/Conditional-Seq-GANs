#datadir=results
#datadir=/home/pascalson/research/MyOpenSource/advanced-sequence-generator
#datadir=/home/pascalson/research/backup_Warship/17000
datadir=/home/pascalson/research/MyOpenSource/chatbot-data
gpu=$1
testtype=$2
opt=
mdl=$3
task=$4
if [ $# == 5 ]; then
  opt=$5
elif [ $# == 8 ]; then
  lr=$5
  lrdecay=$6
  Gstep=$7
  Dstep=$8
  fix_steps=5000
elif [ $# != 4 ]; then
    echo "usage: $0 <gpu-id> <test-type> <used-model> <used-task> [optional-flag]"
    echo "e.g.: $0 0 None MLE OpenSubtitles"
    exit 1;
else
  lr=0.5
  lrdecay=0.99
  Gstep=1
  Dstep=5
  fix_steps=0
fi

if [[ ("$testtype" == 'accuracy') || ("$testtype" == 'KLD') \
    || ("$testtype" == 'RKLD') || ("$testtype" == 'BLEU') \
    || ("$testtype" == 'batch_test_D') ]]; then
  batchsize=1000
else
  batchsize=64
fi

if [ "$task" == 'OpenSubtitles' ]; then
  size=512
  numlayers=1
  dsize=512
elif [ "$task" == 'Counting' ]; then
  size=64
  numlayers=1
  dsize=32
fi

config=""
mdlname=$mdl

if [[ ("$mdl" == 'SeqGAN') || ("$mdl" == 'REGS') \
    || ("$mdl" == 'StepGAN') || ("$mdl" == 'MaliGAN') || ("$mdl" == 'StepGAN-W') \
    || ("$mdl" == 'MC-SeqGAN') || ("$mdl" == 'MC-MaliGAN') \
    || ("$mdl" == 'MaskGAN') ]]; then
    gantype=$mdl
    mdlname=$mdl\_G$Gstep\_D$Dstep\_lr$lr\_lrd$lrdecay
    config="--D-lr=$lr --D-lr-decay-factor=$lrdecay --gan-type=$gantype --gan-size=$dsize --gan-num-layers=$numlayers --G-step=$Gstep --D-step=$Dstep --fix-steps=$fix_steps --option=$opt"
fi

echo "TestType:$testtype"
echo "BatchSize:$batchsize"
echo "GanType: $gantype; DSize: $dsize"

if [ "$task" == 'OpenSubtitles' ]; then
  datapath=$datadir/opensubtitles/opensubtitles.txt
  modeldir=$datadir/OpenSubtitles_Exp/$mdl
  preDmodeldir=$modeldir/D
  premodeldir=$datadir/OpenSubtitles_Exp/MLE
  vocabsize=4000
  buckets='[(8,8),(10,10),(15,15),(20,20)]'
  config="--pre-model-dir=$premodeldir --pre-D-model-dir=$preDmodeldir $config"
elif [ "$task" == 'Counting' ]; then
  datapath=$datadir/data/counting/counting.txt
  modeldir=$datadir/Counting_Exp/$mdlname
  preDmodeldir=None
  premodeldir=$datadir/Counting_Exp/MLE
  vocabsize=14
  buckets='[(5,5),(10,5)]'
  config="--pre-model-dir=$premodeldir --pre-D-model-dir=$preDmodeldir --test-critic="Counting_Task" $config"
fi
echo "TASK:$task; VOCABSIZE:$vocabsize; MODELSIZE:$size;"

CUDA_VISIBLE_DEVICES=$gpu python3 main.py \
  --lr=$lr \
  --lr-decay=$lrdecay \
  --model-dir=$modeldir \
  --data-path=$datapath \
  --size=$size \
  --num-layers=$numlayers \
  --vocab-size=$vocabsize \
  --buckets=$buckets \
  --batch-size=$batchsize \
  --test-type=$testtype \
  $config
