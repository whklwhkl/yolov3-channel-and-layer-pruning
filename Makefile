export:
	for h in `seq 128 64 320`;do \
		for w in `seq 64 64 320`;do \
			python jit_trace.py \
				--cfg cfg/yolov3.cfg \
				--weights weights/best.pt \
				--device cuda \
				--half \
				--input_height $$h \
				--input_width $$w; \
		done; \
	done;


train:
	python train.py \
	  --cfg cfg/yolov3.cfg \
	  --data data/coco.data \
	  --weights weights/yolov3.weights \
	  --epochs 100 \
	  --batch-size 60


sparsify:
	python train.py \
	  --cfg cfg/yolov3-hand.cfg \
	  --data data/coco-person.data \
	  --weights weights/person.weights \
	  --epochs 300 \
	  --batch-size 88 \
	  -sr \
	  --s 0.001 \
	  --prune 1


channel:
	python slim_prune.py \
	  --cfg cfg/yolov3.cfg \
	  --data data/coco.data \
	  --weights weights/last.pt \
	  --global_percent 0.8 \
	  --layer_keep 0.01


layer:
	python layer_prune.py \
	  --cfg cfg/yolov3.cfg \
	  --data data/coco.data \
	  --weights weights/last.pt \
	  --shortcuts 12


finetune:
	python train.py \
	  --cfg cfg/prune_0.85_my_cfg.cfg \
	  --data data/coco.data \
	  --weights weights/prune_0.85_last.weights \
	  --epochs 100 \
	  --batch-size 32
