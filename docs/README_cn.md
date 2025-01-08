> The translation is all done by Cursor with claude-3-5-sonnet.

<p align="center">
  <a href="https://srameo.github.io/projects/le3d/">
    <img src='/.assets/logo.svg' alt='NIPS2024_LE3D_LOGO' width='250px'/><br/>
  </a>
</p>

## <div align="center"><a href="https://srameo.github.io/projects/le3d/">网页预览</a> | <a href="https://srameo.github.io/projects/le3d/intro.html">项目主页</a> | <a href="https://arxiv.org/abs/2406.06216">论文</a> | Google Drive (待定) | <a href="/docs/editor.md">编辑器</a> | <a href="/README.md">英文版</a>
<div align="center">

:newspaper: [**新闻**](#newspaper-新闻) | :wrench: [**安装**](#wrench-依赖和安装) | :tv:[**快速演示**](https://srameo.github.io/projects/le3d/) | :camera: [训练与编辑](#camera-训练与编辑) | :construction: [**贡献**](docs/develop.md) | :scroll: [**许可证**](#scroll-许可证) | :question:[**常见问题**](https://github.com/Srameo/LE3D/issues?q=label%3AFAQ+)

</div>

<span style="font-size: 18px;"><b>注意：</b>本仓库同时也是一个<span style="color: orange;"><a href="https://github.com/XPixelGroup/BasicSR">BasicSR</a>风格</span>的3D高斯散射代码库！欢迎在您自己的项目中使用！如果这个仓库对您有帮助，请考虑给我们一个:star2:！</span>

<b>简介：</b> LE3D是一个项目，用于从带噪RAW图像快速训练和实时渲染HDR视图合成的3DGS模型。

本仓库包含以下论文的官方实现：
> <b>L</b>ighting <b>E</b>very Darkness with <b>3D</b>GS: Fast Training and Real-Time Rendering for HDR View Synthesis<br/>
> [Xin Jin](https://srameo.github.io)<sup>\*</sup>, [Pengyi Jiao](https://github.com/VictorJiao)<sup>\*</sup>, [Zheng-Peng Duan](https://mmcheng.net/dzp/), [Xingchao Yang](https://scholar.google.com/citations?user=OkhB4Y8AAAAJ&hl=zh-CN), [Chongyi Li](https://li-chongyi.github.io/), [Chunle Guo](https://mmcheng.net/clguo/)<sup>\#</sup>, [Bo Ren](http://ren-bo.net/)<sup>\#</sup><br/>
> (\* denotes equal contribution. \# denotes the corresponding author.)<br/>
> arxiv preprint, \[[Homepage](https://srameo.github.io/projects/le3d/)\], \[[Paper Link](https://arxiv.org/abs/2406.06216)\]

<span style="font-size: 18px;"><b>请注意：</b>这些视频使用10位HDR色彩的HEVC编码，在支持HDR的兼容显示器上观看效果最佳，例如最新的苹果设备。</span>
<details>
  <summary>这是我们制作演示视频的方法。</summary>

  https://github.com/user-attachments/assets/9e2a9755-14d5-4788-9393-7d0c7ae95486

  只需设置`关键帧`！所有的插值将自动完成！（另外，您可以在`插值`中设置相机运动的加速度！）
</details>

https://github.com/user-attachments/assets/050f1c37-2667-4f9a-927f-5fffe1de9c9e

https://github.com/user-attachments/assets/440aa492-ed06-4519-8509-7d74c75f3275

想制作您自己的3D视频故事板？请参考[LE3D编辑器](docs/editor.md)获取更多详情。

- 首先，[:wrench: 依赖和安装](#wrench-依赖和安装)。
- 对于**快速预览**，请参考我们的[网页预览器](https://srameo.github.io/projects/le3d/)。
- 对于**使用您自己的数据进行训练和编辑**，请参考[:camera: 训练与编辑](#camera-训练与编辑)。
- 对于**进一步开发**，请参考[:construction: 进一步开发](#construction-进一步开发)。

## :newspaper: 新闻

> 未来工作可以在[todo.md](docs/todo.md)中找到。

<ul>
  <li><b>2025年1月8日</b>：代码发布。</li>
  <li><b>2025年1月3日</b>：发布LE3D的<a href="https://srameo.github.io/projects/le3d/">网页演示</a>！您可以实时查看自己重建的HDR场景！代码在<a href="https://github.com/Srameo/hdr-splat">hdr-splat</a>。</li>
  <li><b>2024年10月10日</b>：LE3D被NIPS 2024接收！</li>
</ul>
<details>
  <summary>历史记录</summary>
  <ul>
  </ul>
</details>

## :wrench: 依赖和安装

### 前提条件

> 注意：我们仅在Ubuntu 20.04、CUDA 11.8、Python 3.10和Pytorch 1.12.1上进行了测试。

- 至少12GB显存的NVIDIA GPU。（由于RAW图像的分辨率相对较高，我们建议至少16GB显存。）
- 已安装Python 3.10。
- 已安装CUDA 11.8。

### install.sh

我们提供了一个便于安装的脚本。

```bash
> ./install.sh -h
用法：./install.sh [选项]
选项：
  -i|--interactive             交互式安装
  -cuda|--install-cuda            安装CUDA
  -colmap|--install-colmap          安装COLMAP
      cuda_enabled          启用CUDA支持（必须跟在--install-colmap后面）
  -env|--create-env              创建并激活conda环境
默认情况下，只安装Python包
```

对于交互式安装，您可以运行并选择您想要的选项：

```bash
> ./install.sh -i
是否要安装CUDA？需要root权限。(y/N)：Y
是否要安装COLMAP？需要root权限。(y/N)：Y
是否要启用CUDA支持？(y/N)：Y
是否要创建并激活名为'basicgs'的conda环境？(y/N)：Y
----------------------------------------
INSTALL_CUDA：true
INSTALL_COLMAP：true
  COLMAP_CUDA_ENABLED：true
CREATE_ENV：true
INTERACTIVE：true
INSTALL_PYTHON_PACKAGES：true
----------------------------------------
...
```

或者您可以运行`./install.sh -cuda -colmap cuda_enabled -env`来安装CUDA、COLMAP、创建conda环境，然后安装所有Python包。

这将帮助您安装所有依赖项和Python包，以及所有[子模块](submodules/README.md)。

## :camera: 训练与编辑

### 数据采集

为了减轻数据收集的负担，我们建议用户只使用前向摄像机拍摄场景。就像[LLFF](https://github.com/Fyusion/LLFF)一样，在[Youtube](https://youtu.be/LY6MgDUzS3M?si=TW-OKOSrm2w9Wiqy&t=83)上提供了**相机放置**的指南。

对于相机设置，我们建议您**将ISO和光圈固定在合理的值**。曝光值（EV）可以设置为较低的值，例如-2。光圈应该设置得尽可能大以避免散焦模糊。

如果您想拍摄多重曝光图像，您可以固定ISO和光圈，然后改变曝光值（EV）来拍摄不同曝光的图像。

> 注意：我们建议您固定ISO和光圈的原因是：
> 1) 固定ISO以保持噪声水平一致。<br/>
> 2) 固定光圈以避免散焦模糊。

对于拍摄工具，我们建议使用那些可以拍摄DNG文件的工具。对于IOS设备，我们使用[Halide](https://halide.cam/)进行拍摄。对于其他设备，我们建议使用[DNGConverter](https://helpx.adobe.com/camera-raw/using/adobe-dng-converter.html)将RAW图像转换为DNG文件。

### 训练

1. 对于训练，我们首先需要准备好数据，如下所示：
    ```bash
    数据路径
    |-- images
    |  |-- IMG_2866.JPG
    |  |-- IMG_2867.JPG
    |  `-- ...
    `-- raw
      |-- IMG_2866.dng
      |-- IMG_2866.json
      |-- IMG_2867.dng
      |-- IMG_2867.json
      `-- ...
    ```
2. 如果您没有json文件，请使用`scripts/data/extract_exif_as_json.sh`从raw图像中提取exif信息。您可以在数据集目录中运行此脚本：`bash scripts/data/extract_exif_as_json.sh path/to/your/dataset/raw`。<br/>
然后您可以使用COLMAP校准相机姿态。
    ```bash
    USE_GPU=1 bash scripts/data/local_colmap.sh path/to/your/dataset PINHOLE
    ```
    > <b style="color: red;">注意：</b> `PINHOLE`是必须用于3DGS校准的相机模型。
3. 校准后，您可以编写自己的yaml文件进行训练！
    ```yaml
    base: options/le3d/base.yaml     # 用于没有多重曝光图像的场景
    base: options/le3d/base_wme.yaml # 用于有多重曝光图像的场景

    name: le3d/bikes # 实验名称

    datasets:
      train:
        name: rawnerf_bikes_train
        scene_root: datasets/rawnerf/scenes/bikes_pinhole # 更改为您的数据集路径

      val:
        name: rawnerf_bikes_val
        scene_root: datasets/rawnerf/scenes/bikes_pinhole # 更改为您的数据集路径

    network_g:
      # 更改为您的稀疏点云路径，此文件将在数据集初始化期间创建
      init_ply_path: datasets/rawnerf/scenes/bikes_pinhole/sparse/0/points3D.ply
    ```

### 查看/编辑

我们提供两种方式在社交媒体上分享您重建的HDR场景。

1. 使用[hdr-splat](https://github.com/Srameo/hdr-splat)用纯JavaScript代码部署您自己的HDR场景查看器。您可以使用以下命令将您的`.ply`文件转换为`.splat`文件。
    ```bash
    bash scripts/export_splat.sh path/to/your/experiment [ITERATION]
    # 例如
    bash scripts/export_splat.sh output/le3d/bikes        latest
    ```
    选定的场景可以在我们的[网页预览器](https://srameo.github.io/projects/le3d/)中找到。
2. 使用[LE3D编辑器](docs/editor.md)创建视频故事板并在社交媒体上分享。

## :construction: 进一步开发

如果您想在您的项目中开发/使用LE3D，欢迎告诉我们。我们会在这个仓库中列出您的项目。

## :book: 引用

如果您觉得我们的仓库对您的研究有用，请考虑引用我们的论文：

```bibtex
@inproceedings{jin2024le3d,
  title={Lighting Every Darkness with 3DGS: Fast Training and Real-Time Rendering for HDR View Synthesis},
  author={Jin, Xin and Jiao, Pengyi and Duan, Zheng-Peng and Yang, Xingchao and Li, Chong-Yi and Guo, Chun-Le and Ren, Bo},
  booktitle={NIPS},
  year={2024}
}
```

## :scroll: 许可证

本代码根据[知识共享署名-非商业性使用4.0国际许可协议](https://creativecommons.org/licenses/by-nc/4.0/)仅供非商业用途使用。
请注意，任何商业用途都需要在使用前获得正式许可。

## :postbox: 联系方式

技术问题，请联系`xjin[AT]mail.nankai.edu.cn`。

商业许可，请联系`cmm[AT]nankai.edu.cn`。

## :handshake: 致谢

本仓库大量借鉴了[BasicSR](https://github.com/XPixelGroup/BasicSR)和[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)。<br/>
我们要特别感谢[李女士](https://xinruli418.github.io/)为我们的项目设计了精美的logo。

我们也感谢所有的贡献者。

<a href="https://github.com/Srameo/LE3D/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Srameo/LE3D" />
</a>
