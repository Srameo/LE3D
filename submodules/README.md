## Submodules

Here contains the submodules of BasicGS.

- Rasterizations
    - **base-rasterization**: Base rasterization module for BasicGS, which could output a rendered image from a given camera pose (from [https://github.com/graphdeco-inria/diff-gaussian-rasterization/](https://github.com/graphdeco-inria/diff-gaussian-rasterization/)).
    - **gcnt-rasterization**: Gaussian count rasterization module for BasicGS, which could output a rendered image from a given camera pose as well as the Gaussian count and the first/last top k (5) Gaussian count.
    - **full-rasterization**: Full rasterization module for BasicGS, which could output a rendered image from a given camera pose, the depth map and the final opacity map (from [https://github.com/ashawkey/diff-gaussian-rasterization](https://github.com/ashawkey/diff-gaussian-rasterization)).
    - **hist-rasterization**: Histogram rasterization module for BasicGS, which could output a rendered historgram with 32 bins from a given camera pose. this is the key module for defocus in BasicGS.
- third_party/glm: glm package for all of our rasterizations.
- Colmap: in case you want to build from source.
- hdr-splat: a totally js implementation of hdr splatting, enable to preview the rendered image in real-time using your browser.
