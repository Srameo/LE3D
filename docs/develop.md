# Develop

> This repository borrows heavily from [BasicSR](https://github.com/XPixelGroup/BasicSR).
>
> If you are a beginner of BasicSR, please refer to [BasicSR](https://github.com/XPixelGroup/BasicSR).

- [Develop](#develop)
  - [More Than BasicSR!](#more-than-basicsr)

## More Than BasicSR!

Compared with BasicSR, the option files could be more simple with a `base` key:

```yaml
base:
- options/base/dataset/pretrain/SID_raw_gt.yaml           # train dataset
- options/base/dataset/test/SID_SonyA7S2_val_split.yaml   # test dataset
- options/base/network_g/repnr_unet.yaml                  # network_g
- options/base/noise_g/noise_g_virtual.yaml               # noise_g
- options/base/pretrain/MM22_PMN.yaml                     # train
- options/base/val_and_logger.yaml                        # val + logger

name: LED_Pretrain_MM22_PMN_Setting
model_type: RAWImageDenoisingModel
scale: 1
num_gpu: 1
manual_seed: 2022

path:
  pretrain_network_g: ~
  predefined_noise_g: ~
  strict_load_g: true
  resume_state: ~
  CRF: datasets/EMoR

val:
  illumination_correct: true
```

Just use the relative path to project root directory for a cleaner and simpler config files!
