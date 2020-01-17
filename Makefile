H=416
W=416


export:
	python jit_trace.py \
		--cfg cfg/final.cfg \
		--weights weights/final.weights \
		--device cuda \
		--half \
		--input_height ${H} \
		--input_width ${W}


anchor:
	python -c "from utils.utils import kmeans_targets; kmeans_targets('data/coco-person-bag/train.txt')"


train:
	python train.py \
	  --cfg cfg/yolov3.cfg \
	  --data data/coco.data \
	  --weights weights/yolov3.weights \
	  --epochs 100 \
	  --batch-size 60


train2:
	python train.py \
		--cfg cfg/yolov3-spp-4cls.cfg \
		--data data/coco-person-bag.data \
		--epochs 100 \
		--batch-size 64 \
		-sr \
		--s 0.00001 \
		--prune 1


CFG=yolov3-hand.cfg


reduce_class:
	python simplify_det_head.py cfg/${CFG} weights/best.pt weights/person.weights 0


DATA=coco-person.data


reduce_data:
	python utils/extract_coco_person_data.py


sparsify:
	python train.py \
	  --cfg cfg/${CFG} \
	  --data data/${DATA} \
	  --weights weights/person.weights \
	  --epochs 300 \
	  --batch-size 88 \
	  -sr \
	  --s 0.001 \
	  --prune 1


CHANNEL=0.74375
KEEP=0.1


channel:
	python slim_prune.py \
	  --cfg cfg/${CFG} \
	  --data data/${DATA} \
	  --weights weights/last.pt \
	  --global_percent ${CHANNEL} \
	  --layer_keep ${KEEP}


LAYER=3


layer:
	python layer_prune.py \
	  --cfg cfg/prune_${CHANNEL}_keep_${KEEP}_${CFG} \
	  --data data/${DATA} \
	  --weights weights/prune_${CHANNEL}_keep_${KEEP}_last.weights \
	  --shortcuts ${LAYER}


finetune:
	python train.py \
	  --cfg cfg/prune_${LAYER}_shortcut_prune_${CHANNEL}_keep_${KEEP}_${CFG} \
	  --data data/${DATA} \
	  --weights weights/prune_${LAYER}_shortcut_prune_${CHANNEL}_keep_${KEEP}_last.weights \
	  --epochs 100 \
	  --batch-size 96


finalize:
	python simplify_det_head.py \
		cfg/prune_${LAYER}_shortcut_prune_${CHANNEL}_keep_${KEEP}_${CFG} \
		weights/best.pt \
		weights/final.weights \
		-1
