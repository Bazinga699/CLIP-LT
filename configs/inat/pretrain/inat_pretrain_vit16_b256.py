# 8 GPU
cfg = dict(
    model='CVLP_vit16',
    desc_path='/data/datasets/inaturalist2018/annotations',
    pretrained_clip='/data/pretrained_checkpoint/ViT-B-16.pt',
    context_length=75,
    pretrain_cvlp=True,
    loss_type="smoothCE",

    data_set='INAT',
    drop_last=True,
    eval_pretrain=True,

    weight_sample=True,
    use_sqrt_freq=True,

    lr=3.5e-4,
    epochs=100,
    batch_size=32,

    repeated_aug=False,
    mixup=0.,
    cutmix=0.,
    clip_ms=True,
    distillation_beta=0.5,
    distillation_type='logits',
)
