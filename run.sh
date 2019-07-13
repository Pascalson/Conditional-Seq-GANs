#datadir=/home/pascalson/research/MyOpenSource/chatbot-data/Synthetics2
datadir=/home/pascalson/research/MyOpenSource/chatbot-data/English
opensubtitlesdir=/home/pascalson/handover/StepGAN/model_ckpts/OpenSubtitles_Exp
countingdir=./experiments/Counting_Exp


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


if [[ ("$testtype" == 'accuracy') ]]; then
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
gantype=$mdl
mdlname=Test_$mdl\_G$Gstep\_D$Dstep\_lr$lr\_lrd$lrdecay
config="--D-lr=$lr --D-lr-decay-factor=$lrdecay --gan-type=$gantype --gan-size=$dsize --gan-num-layers=$numlayers --G-step=$Gstep --D-step=$Dstep --fix-steps=$fix_steps --option=$opt"

echo "TestType:$testtype"
echo "BatchSize:$batchsize"
echo "GanType: $gantype; DSize: $dsize"



if [ "$task" == 'OpenSubtitles' ]; then
  datapath=$datadir/opensubtitles/opensubtitles.txt
  modeldir=$opensubtitlesdir/Test_$mdl
  premodeldir=$opensubtitlesdir/MLE
  vocabsize=4000
  buckets='[(8,8),(10,10),(15,15),(20,20)]'
  config="--pre-model-dir=$premodeldir $config"
elif [ "$task" == 'Counting' ]; then
  datapath=$datadir/counting/counting.txt
  modeldir=$countingdir/$mdlname
  premodeldir=$countingdir/MLE
  vocabsize=14
  buckets='[(5,5),(10,5)]'
  config="--pre-model-dir=$premodeldir --test-critic="Counting_Task" $config"
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
