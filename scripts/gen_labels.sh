folder=$1


for i in `ls $folder`;do
  python gen_labels.py \
    --cfg cfg/yolov3.cfg \
    --weights weights/yolov3.weights \
    --img-size 608 \
    --source $1/$i
done
