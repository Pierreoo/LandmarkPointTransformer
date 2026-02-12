_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 4  # bs: total bs in all gpus
num_worker = 20
mix_prob = 0.0
empty_cache = False
enable_amp = True
num_keypoints = 22  # Here no background !!!
num_points = 8192

# scheduler settings
epoch = 500
eval_epoch = 50

# find_unused_parameters = True
# clip_grad = 3.0

# model settings
model = dict(type="DefaultKeypointer",
             backbone=dict(
                 type="PT-v2m2",
                 in_channels=3,
                 num_classes=num_keypoints,
                 patch_embed_neighbours=16,
                 enc_neighbours=(32, 32, 64, 64),
                 dec_neighbours=(16, 32, 32, 64),
                 grid_sizes=(0.02, 0.04, 0.08, 0.16),
                 attn_drop_rate=.2,
                 drop_path_rate=.3,
                 pe_multiplier=False
             ),
             criteria=[
                 dict(type="CorrespondenceLoss", loss_weight=1.0, ignore_index=-1),
             ],
             )


optimizer = dict(type="AdamW", lr=0.0003, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.0003,
    pct_start=0.4,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
dataset_type = "KeyboneNetDataset"
data_root = "data/KeyboneNetPlus"
category = 'FB'
ignore_index = -1
names = ["Background"] + [f"KP{i + 1}" for i in range(num_keypoints - 1)]

data = dict(
    ignore_index=ignore_index,
    num_classes=num_keypoints,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        num_points=num_points,
        data_root=data_root,
        category=category,
        save_record=True,
        transform=[
            dict(type='NormalizeCoord'),
            dict(type='RandomJitter', sigma=0.002, clip=0.005),
            dict(type='RandomFlip', p=0.5, dim=1),
            dict(type='RandomRotate'),
            dict(type='RandomScale'),
            dict(
                type='RandomShift',
                shift=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))),
            dict(type='ToTensor'),
            dict(
                type="Collect",
                keys=("coord", "kp_index", "name", "cls_token"),
                feat_keys=("coord",),
                offset_keys_dict=dict(offset="coord"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        num_points=num_points,
        data_root=data_root,
        category=category,
        save_record=True,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "kp_index", "name", "cls_token"),
                feat_keys=("coord",),
                offset_keys_dict=dict(offset="coord")
            ),

        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        num_points=num_points,
        data_root=data_root,
        category=category,
        save_record=True,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "kp_index", "name", "cls_token"),
                feat_keys=("coord",),
                offset_keys_dict=dict(offset="coord"),
            ),
        ],
        test_mode=False,
    ),
)

# hooks
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CorrespondenceEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# tester
test_norm_geo = False
test = dict(type="CorrespondenceTester", verbose=True)