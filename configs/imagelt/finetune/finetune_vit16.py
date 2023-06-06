# 8 GPU
cfg = dict(
    model='LGR_vit16',
    desc_path='/data/datasets/ImageNet_LT',
    pretrained_clip='/data/pretrained_checkpoint/ViT-B-16.pt',
    context_length=75,
    pretrain_cvlp=False,
    pretrain_cvlp_path='/data/VL-LTR/test/imagenet_LT_pretrain_vit16_2',
    loss_type="CE",
    two_branch=True,

    data_set='IMNET_LT',
    drop_last=True,

    weight_sample=True,
    use_sqrt_freq=True,

    lr=1e-3,
    min_lr=0,
    warmup_epochs=0,
    text_lr=1e-6,
    
    epochs=50,
    batch_size=128,

    repeated_aug=False,
    clip_ms=True,
    test=False,
)
