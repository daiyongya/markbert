python run_ner.py \
    --ptm_path ../ckpt230000/  \
    --ptm_name ../ckpt230000/ \
    --dataset msra_ner \
    --lr 3e-5 \
    --use_crf 0 \
    --warmup_step 1000 \
    --sampler bucket  \
    --batch_size 10 \
    --after_bert linear \
    --fix_ptm_epoch -1 \
    --weight_decay 0.1 \
    --crf_lr_rate 5 \
    --gradient_clip_norm_other 5 \
    --gradient_clip_norm_bert 1 \
    --warmup_schedule inverse_square \
    --encoding_type bmeso \
    --epoch 3