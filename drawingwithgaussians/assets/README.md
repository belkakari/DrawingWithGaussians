# LPIPS-Alex weights

`lpips_alex.npz` contains only the five convolutional feature layers from the
TorchVision AlexNet ImageNet checkpoint plus the five LPIPS-Alex linear heads;
classifier weights are not included.

- AlexNet source SHA-256: `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`
- LPIPS-Alex source SHA-256: `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`
- Converted NPZ SHA-256: `e5e20342cc19498845cfd13187ef04ea893822daa3d8dc32668e7488011b7962`

The LPIPS implementation derives from the BSD-licensed PerceptualSimilarity
project as carried by Apache-2.0 TorchMetrics; TorchVision is BSD-3-Clause.
